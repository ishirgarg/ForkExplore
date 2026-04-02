"""Go Explore agent.

Two-phase training loop:
  - Go phase   (phase == 0): deterministic GCP tries to reach a proposed goal.
  - Explore phase (phase == 1): stochastic explore policy maximises epistemic
    uncertainty about reaching states from the go-phase start.

Phase management is handled by ``GoExploreWrapper`` (see ``jaxgcrl/envs/wrappers.py``).
"""

import functools
import logging
import pickle
import random
import time
from typing import Any, Callable, Literal, NamedTuple, Optional, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from etils import epath
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    GoExploreWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization

from .types import Actor, Critic, TrainingState, Transition, GoalProposerState
from .algorithms import get_algorithm, get_explore_policy
from .algorithms_utils import reconstruct_full_critic_params
from .utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
)
from .losses import update_alpha_sac, update_critic_sac, update_actor_sac
from .visualization import all_visualizations, visualize_go_explore_phases
from .goal_proposers import (
    create_goal_proposer,
    create_random_env_goals_proposer,
    create_explore_reward_fn,
)

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class GoExplore:
    """Go Explore agent with dual-policy training (GCP + explore policy).
    
    The goal-conditioned policy (GCP) trains on *all* transitions while the
    non-goal-conditioned explore policy trains on *explore-phase* transitions
    only, using the epistemic-uncertainty reward.
    """

    # Algorithm type for the goal-conditioned policy
    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 20000
    min_replay_size: int = 1000
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    repr_dim: int = 64
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    tau: float = 0.005
    n_critics: int = 2
    use_her: bool = True

    goal_proposer_name: Literal["random_env_goals", "rb", "q_epistemic", "ucgr", "max_critic_to_env", "mega", "omega"] = "random_env_goals"
    ep_goal_proposer_name: Literal["nearest_env_goal_to_gcp_goal"] = "nearest_env_goal_to_gcp_goal"
    num_candidates: int = 512

    # ── Go Explore specific parameters ──────────────────────────────────────
    num_gcp_steps: int = 100      # max steps in go phase before forcing explore
    num_ep_steps: int = 50        # steps in explore phase before reset to go
    explore_reward_type: Literal["q_epistemic"] = "q_epistemic"
    explore_policy_type: Literal["sac"] = "sac"
    explore_noise_scale: float = 0.1   # kept for future use

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env
        
        action_size = train_env.action_size
        state_size  = train_env.state_dim
        goal_size   = len(train_env.goal_indices)
        obs_size    = state_size + goal_size
        
        # ── Eval env (unchanged from baseline) ──────────────────────────────
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # ── Train env: GoExploreWrapper replaces TrainAutoResetWrapper ───────
        # No TrajectoryIdWrapper here – GoExploreWrapper manages traj_id.
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        train_env = GoExploreWrapper(
            train_env,
            num_gcp_steps=self.num_gcp_steps,
            num_ep_steps=self.num_ep_steps,
            state_size=state_size,
            goal_size=goal_size,
            goal_indices=unwrapped_env.goal_indices,
        )

        # ── Step count bookkeeping ───────────────────────────────────────────
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps    = self.min_replay_size * config.num_envs
        num_prefill_actor_steps  = int(np.ceil(self.min_replay_size / self.unroll_length))
        
        available_env_steps          = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch          = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps:          %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps:         %d", num_prefill_actor_steps)
        logging.info("env_steps_per_epoch:             %d", env_steps_per_epoch)
        logging.info("num_training_steps_per_epoch:    %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)
        key, explore_actor_key, explore_critic_key, explore_alpha_key = jax.random.split(key, 4)

        # ── GCP (goal-conditioned policy) ────────────────────────────────────
        gcp_actor, gcp_critic = get_algorithm(
            agent_type=self.agent_type,
            action_size=action_size,
            obs_size=obs_size,
            state_size=state_size,
            goal_indices=train_env.goal_indices,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            repr_dim=self.repr_dim,
            discounting=self.discounting,
            energy_fn=self.energy_fn,
            n_critics=self.n_critics,
        )

        gcp_actor_params  = gcp_actor.init(actor_key,  np.ones([1, obs_size]))
        gcp_critic_params = gcp_critic.init(critic_key, np.ones([1, obs_size]))

        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        gcp_critic_states = gcp_critic.create_critic_states(gcp_critic_params, self.critic_lr)

        target_entropy = -0.5 * action_size
        log_alpha      = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state    = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        target_critic_params = None
        if self.agent_type == "sac":
            target_critic_params = gcp_critic_params

        # ── Explore policy (non-goal-conditioned, takes state_size input) ────
        explore_actor, explore_critic = get_explore_policy(
            explore_policy_type=self.explore_policy_type,
            action_size=action_size,
            state_size=state_size,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            n_critics=self.n_critics,
        )

        explore_actor_params  = explore_actor.init(explore_actor_key,  np.ones([1, state_size]))
        explore_critic_params = explore_critic.init(explore_critic_key, np.ones([1, state_size]))

        explore_actor_state = TrainState.create(
            apply_fn=explore_actor.apply,
            params=explore_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        explore_critic_states = explore_critic.create_critic_states(explore_critic_params, self.critic_lr)

        explore_log_alpha   = jnp.asarray(0.0, dtype=jnp.float32)
        explore_alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": explore_log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )
        # SAC target network for explore critic
        explore_target_critic_params = explore_critic_params

        # ── Combined TrainingState ───────────────────────────────────────────
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=gcp_actor_state,
            critic_states=gcp_critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
            explore_actor_state=explore_actor_state,
            explore_critic_states=explore_critic_states,
            explore_alpha_state=explore_alpha_state,
            explore_target_critic_params=explore_target_critic_params,
        )
        
        # ── Goal proposer ────────────────────────────────────────────────────
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
            self.num_candidates,
            state_size=unwrapped_env.state_dim,
            goal_indices=unwrapped_env.goal_indices,
            actor=gcp_actor,
            critic=gcp_critic,
            discounting=self.discounting,
        )
        
        # ── Explore reward factory ───────────────────────────────────────────
        explore_reward_fn = create_explore_reward_fn(
            reward_type=self.explore_reward_type,
            critic=gcp_critic,
            gcp_actor=gcp_actor,
            state_size=state_size,
            goal_indices=tuple(unwrapped_env.goal_indices),
        )

        # ── Env reset ───────────────────────────────────────────────────────
        random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys      = jax.random.split(env_key, config.num_envs)
        initial_goals = jax.vmap(random_goals_proposer)(env_keys)
        
        env_state = train_env.reset(env_keys, goal=initial_goals)
        # Maintain proposed_goals in info for the goal-proposer cycle
        info = dict(env_state.info)
        info['proposed_goals'] = initial_goals
        env_state = env_state.replace(info=info)

        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # ── Replay buffer ────────────────────────────────────────────────────
        # Transitions include "phase" in state_extras for filtering explore updates.
        # Go Explore always stores next_observation (explore critic is SAC, needs it).
        dummy_transition = create_single_dummy_transition(
            obs_size=obs_size,
            action_size=action_size,
            agent_type="sac",   # always include next_observation for explore critic
            include_phase=True,
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)
        
        dummy_batch_transition = create_dummy_transition_for_buffer(
            unroll_length=self.unroll_length,
            num_envs=config.num_envs,
            obs_size=obs_size,
            action_size=action_size,
            agent_type="sac",   # always include next_observation for explore critic
            include_phase=True,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)
        
        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type="sac",   # always include next_observation for explore critic
            include_phase=True,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=gcp_actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(gcp_critic_states)},
        )

        # ── actor_step ───────────────────────────────────────────────────────
        def actor_step(training_state, env, env_state, key, extra_fields):
            """One env step; selects policy based on phase, computes explore reward."""
            key, action_key_gcp, action_key_exp, env_rng, reward_key = jax.random.split(key, 5)

            phase     = env_state.info['phase']     # (num_envs,) int32
            go_goal   = env_state.info['go_goal']   # (num_envs, goal_size)
            raw_state = env_state.obs[:, :state_size]  # (num_envs, state_size)
            dummy_goal = jnp.full((raw_state.shape[0], goal_size), -1.0)

            # GCP observation: [state, go_goal] in go phase, [state, -1] in explore
            gcp_goal = jnp.where(phase[:, None] == 0, go_goal, dummy_goal)
            gcp_obs  = jnp.concatenate([raw_state, gcp_goal], axis=-1)  # (num_envs, obs_size)

            # Explore observation: state only
            explore_obs = raw_state  # (num_envs, state_size)

            # Sample actions from both policies
            gcp_actions = gcp_actor.sample_actions(
                training_state.actor_state.params, gcp_obs, action_key_gcp, is_deterministic=True
            )
            explore_actions = explore_actor.sample_actions(
                training_state.explore_actor_state.params, explore_obs, action_key_exp, is_deterministic=False
            )

            # Select action based on phase (0=go → GCP, 1=explore → explore policy)
            actions = jnp.where(phase[:, None] == 0, gcp_actions, explore_actions)

            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras['phase'] = phase

            # ── Compute explore reward (every step; zeroed for go phase) ──────
            full_gcp_critic_params = reconstruct_full_critic_params(
                {i: cs.params for i, cs in enumerate(training_state.critic_states)}
            )
            explore_first_obs = env_state.info['explore_first_obs']  # (num_envs, obs_size)

            pre_reset_next_obs = nstate.info['pre_reset_obs']  # (num_envs, obs_size)
            reward_keys = jax.random.split(reward_key, config.num_envs)
            explore_rewards = jax.vmap(
                lambda fo, co, rk: explore_reward_fn(
                    fo, co, training_state.actor_state.params, full_gcp_critic_params, rk
                )
            )(explore_first_obs, pre_reset_next_obs, reward_keys)  # (num_envs,)

            # Use explore reward only in explore phase
            in_explore  = (phase == 1)
            final_reward = jnp.where(in_explore, explore_rewards, nstate.reward)

            # Always store next_observation so the explore critic (SAC) can use it.
            # CRL GCP update ignores next_observation; storing it is harmless.
            next_gcp_obs = jnp.concatenate([nstate.obs[:, :state_size], gcp_goal], axis=-1)

            return nstate, Transition(
                observation=gcp_obs,
                action=actions, 
                reward=final_reward,
                discount=1 - nstate.done,
                next_observation=next_gcp_obs,
                extras={"state_extras": state_extras},
            )

        # ── get_experience ───────────────────────────────────────────────────
        def get_experience(training_state, env_state, buffer_state, key,
                           experience_count, goal_proposer_state):
            buffer_state, transitions_sample = replay_buffer.sample(buffer_state)
            
            goal_proposer_state = goal_proposer_state.replace(
                transitions_sample=transitions_sample,
                actor_params=training_state.actor_state.params,
                critic_params={i: cs.params for i, cs in enumerate(training_state.critic_states)},
            )
            
            num_envs_     = config.num_envs
            episode_length = config.episode_length
            info           = dict(env_state.info)
            
            reset_threshold    = jnp.array(episode_length // (self.unroll_length * 2), dtype=jnp.int32)
            new_experience_count = experience_count + 1
            
            def propose_new_goals(env_state, key, info, experience_count, goal_proposer_state):
                viz_key, goal_key = jax.random.split(key)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs_)
                goal_keys   = jax.random.split(goal_key, num_envs_)
                first_obs   = info['first_obs']

                def propose_single(rng_key, obs, state):
                    goal, updated_state, log_data = goal_proposer(rng_key, obs, state)
                    return goal, log_data
                
                new_goals, log_data_tree = jax.vmap(
                    propose_single, in_axes=(0, 0, None)
                )(goal_keys, first_obs, goal_proposer_state)
                info['proposed_goals'] = new_goals
                
                # Pass env_steps to limit visualization frequency (max once per 1M steps)
                env_steps = training_state.env_steps

                def log_viz(log_data_tree_np, viz_idx, steps):
                    selected = {k: v[viz_idx] for k, v in log_data_tree_np.items()}
                    handle_goal_proposer_visualization(
                        selected, self.goal_proposer_name,
                        unwrapped_env.x_bounds, unwrapped_env.y_bounds, steps
                    )
                    return jnp.array(0, dtype=jnp.int32)
                
                jax.experimental.io_callback(
                    log_viz, jnp.array(0, dtype=jnp.int32),
                    log_data_tree, viz_env_idx, env_steps
                )
                return env_state, info, jnp.array(0, dtype=jnp.int32), goal_proposer_state
            
            def keep_existing_goals(env_state, key, info, experience_count, goal_proposer_state):
                return env_state, info, experience_count + 1, goal_proposer_state
            
            env_state, info, updated_experience_count, updated_goal_proposer_state = jax.lax.cond(
                new_experience_count >= reset_threshold,
                propose_new_goals,
                keep_existing_goals,
                env_state, key, info, experience_count, goal_proposer_state,
            )
            env_state = env_state.replace(info=info)
            
            @jax.jit
            def f(carry, _):
                env_state, buffer_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    k,
                    extra_fields=("truncation", "traj_id", "phase"),
                )
                return (env_state, buffer_state, next_k), transition

            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, key), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            
            return env_state, buffer_state, updated_experience_count, updated_goal_proposer_state
            
        # ── prefill_replay_buffer ────────────────────────────────────────────
        def prefill_replay_buffer(training_state, env_state, buffer_state, key, goal_proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, new_k = jax.random.split(k)
                es, bs, ec, gps = get_experience(ts, es, bs, k, ts.experience_count, gps)
                ts = ts.replace(
                    env_steps=ts.env_steps + config.num_envs * self.unroll_length,
                    experience_count=ec,
                )
                return (ts, es, bs, new_k, gps), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ── update_networks (GCP) ────────────────────────────────────────────
        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            if self.agent_type == "sac":
                key, alpha_key, critic_key, actor_key = jax.random.split(key, 4)
            else:
                key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
            )
            networks = dict(actor=gcp_actor, critic=gcp_critic)

            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
            else:  # sac
                training_state, alpha_metrics  = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)
            
            # Update SAC target network
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                full_cp = {}
                for i, cs in enumerate(training_state.critic_states):
                    for lname, lparams in cs.params.items():
                        full_cp[f"critic_{i}_{lname}"] = lparams
                new_target = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params, full_cp,
                )
                training_state = training_state.replace(target_critic_params=new_target)
            
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
            return (training_state, key), metrics

        # ── update_explore_networks (explore policy, SAC on explore phase) ───
        @jax.jit
        def update_explore_networks(carry, transitions):
            """Update explore actor/critic with phase-masked SAC loss."""
            training_state, key = carry
            key, alpha_key, critic_key, actor_key = jax.random.split(key, 4)

            # Extract phase mask from extras; shape: (batch_size,) after flatten
            phase_flat = transitions.extras["state_extras"]["phase"]  # (batch_size,)
            explore_mask = (phase_flat == 1).astype(jnp.float32)     # 1 for explore, 0 for go

            # Build explore-obs transitions (state_size instead of obs_size).
            # next_observation is always stored (actor_step always fills it).
            state_obs      = transitions.observation[:, :state_size]
            next_state_obs = transitions.next_observation[:, :state_size]
           
            explore_transitions = transitions._replace(
                observation=state_obs,
                next_observation=next_state_obs,
            )

            # Build a temporary TrainingState that has explore states in the
            # standard actor_state / critic_states / alpha_state fields so we
            # can reuse the existing SAC loss functions directly.
            temp_ts = TrainingState(
                env_steps=training_state.env_steps,
                gradient_steps=training_state.gradient_steps,
                experience_count=training_state.experience_count,
                actor_state=training_state.explore_actor_state,
                critic_states=training_state.explore_critic_states,
                alpha_state=training_state.explore_alpha_state,
                target_critic_params=training_state.explore_target_critic_params,
            )

            explore_context = dict(
                discounting=self.discounting,
                target_entropy=target_entropy,
                state_size=state_size,
                action_size=action_size,
                sample_weights=explore_mask,
            )
            explore_networks = dict(actor=explore_actor, critic=explore_critic)

            temp_ts, alpha_metrics  = update_alpha_sac(explore_context, explore_networks, explore_transitions, temp_ts, alpha_key)
            temp_ts, critic_metrics = explore_critic.update(explore_context, explore_networks, explore_transitions, temp_ts, critic_key)
            temp_ts, actor_metrics  = explore_actor.update(explore_context, explore_networks, explore_transitions, temp_ts, actor_key)

            # Update explore target network
            full_exp_cp = {}
            for i, cs in enumerate(temp_ts.critic_states):
                for lname, lparams in cs.params.items():
                    full_exp_cp[f"critic_{i}_{lname}"] = lparams
            new_exp_target = jax.tree_util.tree_map(
                lambda x, y: x * (1 - self.tau) + y * self.tau,
                temp_ts.target_critic_params, full_exp_cp,
            )

            # Write updated explore states back to the full training_state
            training_state = training_state.replace(
                explore_actor_state=temp_ts.actor_state,
                explore_critic_states=temp_ts.critic_states,
                explore_alpha_state=temp_ts.alpha_state,
                explore_target_critic_params=new_exp_target,
            )

            metrics = {}
            metrics.update({f"explore_{k}": v for k, v in alpha_metrics.items()})
            metrics.update({f"explore_{k}": v for k, v in critic_metrics.items()})
            metrics.update({f"explore_{k}": v for k, v in actor_metrics.items()})
            return (training_state, key), metrics

        # ── training_step ────────────────────────────────────────────────────
        @jax.jit
        def training_step(training_state, env_state, buffer_state, key, goal_proposer_state):
            exp_key, process_key, train_key, explore_train_key = jax.random.split(key, 4)

            env_state, buffer_state, updated_ec, updated_gps = get_experience(
                training_state, env_state, buffer_state, exp_key,
                training_state.experience_count, goal_proposer_state,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
                experience_count=updated_ec,
            )

            # ── GCP update (all transitions) ──────────────────────────────────
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            transitions, _ = gcp_actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                state_size, tuple(train_env.goal_indices),
                train_env.goal_reach_thresh, self.use_her,
            )
            (training_state, _), gcp_metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
            )

            # ── Explore update (explore-phase transitions, masked) ────────────
            buffer_state, explore_trans_raw = replay_buffer.sample(buffer_state)
            # Simple reshape/permute (no HER, no flatten_batch) for explore transitions
            explore_trans, _ = explore_actor.process_transitions(
                explore_trans_raw, process_key, self.batch_size, self.discounting,
                state_size, tuple(train_env.goal_indices),
                train_env.goal_reach_thresh, use_her=False,
            )
            (training_state, _), explore_metrics = jax.lax.scan(
                update_explore_networks, (training_state, explore_train_key), explore_trans
            )

            metrics = {}
            metrics.update(gcp_metrics)
            metrics.update(explore_metrics)
            return (training_state, env_state, buffer_state, updated_gps), metrics

        # ── training_epoch ───────────────────────────────────────────────────
        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key, goal_proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, gps = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, gps), metrics = training_step(ts, es, bs, train_key, gps)
                return (ts, es, bs, k, gps), metrics

            (training_state, env_state, buffer_state, key, goal_proposer_state), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_training_steps_per_epoch,
            )

            # Go Explore phase metrics — computed from cumulative counters so the
            # value is independent of which phase each env happens to be in at
            # epoch end (Bug 2 fix: point-in-time snapshot was biased by ~50% of
            # envs being mid-go-phase with success=0).
            total_completions   = jnp.sum(env_state.info['go_completions_total'])
            total_successes     = jnp.sum(env_state.info['go_successes_total'])
            go_success_rate     = jnp.where(total_completions > 0,
                                            total_successes / total_completions,
                                            0.0)
            total_success_steps = jnp.sum(env_state.info['go_success_steps_total'])
            avg_go_steps        = jnp.where(total_successes > 0,
                                            total_success_steps / total_successes,
                                            0.0)
            
            # Broadcast to match scan output shape for consistent aggregation
            scan_shape = jax.tree_util.tree_leaves(metrics)[0].shape if metrics else (1,)
            metrics["go_phase_success_rate"] = jnp.broadcast_to(
                go_success_rate, scan_shape
            )
            metrics["avg_go_phase_steps"]    = jnp.broadcast_to(
                avg_go_steps, scan_shape
            )
            metrics["buffer_current_size"]   = jnp.broadcast_to(
                replay_buffer.size(buffer_state), scan_shape
            )

            return training_state, env_state, buffer_state, goal_proposer_state, metrics

        # ── prefill ──────────────────────────────────────────────────────────
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, goal_proposer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state
        )

        # ── Evaluator ────────────────────────────────────────────────────────
        # Use a simpler eval actor_step that avoids phase / go_goal lookups
        # (eval env has no GoExploreWrapper, so those fields won't exist).
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            actions = gcp_actor.sample_actions(
                training_state.actor_state.params,
                env_state.obs,
                jax.random.PRNGKey(0),
                is_deterministic=True,
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=None,
                extras={"state_extras": state_extras},
            )

        evaluator = ActorEvaluator(
            eval_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ── Main training loop ────────────────────────────────────────────────
        training_walltime      = 0
        last_visualization_step = -1
        logging.info("starting training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, goal_proposer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key, goal_proposer_state
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time  = time.time() - t
            training_walltime   += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            
            # Convert metrics to Python values for logging
            metrics_dict = {}
            for name, value in metrics.items():
                if hasattr(value, 'item'):
                    metrics_dict[f"training/{name}"] = float(value.item())
                elif hasattr(value, '__float__'):
                    metrics_dict[f"training/{name}"] = float(value)
                else:
                    metrics_dict[f"training/{name}"] = value
            
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **metrics_dict,
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            # Visualize trajectories every 1M steps
            if current_step // 1_000_000 > last_visualization_step // 1_000_000:
                key, viz_key = jax.random.split(key)
                buffer_state = all_visualizations(
                    replay_buffer=replay_buffer,
                    buffer_state=buffer_state,
                    env=unwrapped_env,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                    rng_key=viz_key,
                )
                # Separate Go Explore phase breakdown plot
                _, phase_transitions = replay_buffer.sample(buffer_state)
                visualize_go_explore_phases(
                    phase_transitions,
                    unwrapped_env.x_bounds,
                    unwrapped_env.y_bounds,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                )
                last_visualization_step = current_step

            do_render = ne % config.visualization_interval == 0
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: gcp_actor.apply(param, obs)
            else:
                make_policy = lambda param: lambda obs, rng: (
                    gcp_actor.sample_actions(param, obs, rng, is_deterministic=True), {}
                )

            # Build full GCP critic params for checkpointing
            full_critic_params = {}
            for i, cs in enumerate(training_state.critic_states):
                for lname, lparams in cs.params.items():
                    if lname in ["sa_encoder", "g_encoder"]:
                        full_critic_params[f"{lname}_{i}"] = lparams
                    else:
                        full_critic_params[f"critic_{i}_{lname}"] = lparams
            
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                full_critic_params,
            )
            
            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)
        return make_policy, params, metrics
