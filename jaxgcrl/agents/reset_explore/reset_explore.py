"""Reset Explore agent.

Resets to a state provided by a *reset_proposer* (same API as goal proposers)
and rolls out the goal-conditioned policy toward a goal from a
*goal_proposer*.  There are no go/explore phases — just normal policy
rollouts with auto-reset on episode truncation.
"""

import logging
import random
import time
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    ResetExploreAutoResetWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue

# Reuse all shared components from go_explore
from jaxgcrl.agents.go_explore.types import TrainingState, Transition, GoalProposerState
from jaxgcrl.agents.go_explore.algorithms import get_algorithm
from jaxgcrl.agents.go_explore.utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
)
from jaxgcrl.agents.go_explore.losses import update_alpha_sac
from jaxgcrl.agents.go_explore.visualization import (
    all_visualizations,
    handle_goal_proposer_visualization,
)
from jaxgcrl.agents.reset_explore.visualization import (
    handle_reset_explore_visualization,
)
from jaxgcrl.agents.go_explore.goal_proposers import (
    create_goal_proposer,
    create_random_env_goals_proposer,
)

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class ResetExplore:
    """Reset Explore agent.

    At each episode boundary the environment resets to a position chosen by
    ``reset_proposer`` and pursues a goal chosen by ``goal_proposer``.
    Both proposers share the standard goal-proposer API, so any goal proposer
    can serve as a reset proposer and vice-versa.
    """

    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 30000
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

    goal_proposer_name: Literal[
        "random_env_goals", "rb", "q_epistemic", "ucgr",
        "max_critic_to_env", "mega", "omega",
    ] = "random_env_goals"
    reset_proposer_name: Literal[
        "random_env_goals", "rb", "q_epistemic", "ucgr",
        "max_critic_to_env", "mega", "omega",
    ] = "random_env_goals"
    # Probability an env resets from the original initial-state distribution
    # instead of using the reset proposer (per-reset Bernoulli).
    p_initial_reset: float = 0.0
    # Separate goal proposer for initial-distribution resets (defaults to same family)
    goal_proposer_name_initial: Literal[
        "random_env_goals", "rb", "q_epistemic", "ucgr",
        "max_critic_to_env", "mega", "omega",
    ] = "random_env_goals"
    num_candidates: int = 512

    def check_config(self, config):
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0

    # --------------------------------------------------------------------- #
    # train_fn
    # --------------------------------------------------------------------- #
    def train_fn(
        self,
        config,
        train_env,
        eval_env=None,
        randomization_fn=None,
        progress_fn=lambda *args: None,
    ):
        self.check_config(config)
        unwrapped_env = train_env

        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size

        # ── Wrap envs ───────────────────────────────────────────────────
        train_env = TrajectoryIdWrapper(train_env)
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        # ResetExploreAutoResetWrapper added below, after proposers are created

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # ── Step bookkeeping ────────────────────────────────────────────
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        available_env_steps = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step
        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps: %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps: %d", num_prefill_actor_steps)
        logging.info("env_steps_per_epoch: %d", env_steps_per_epoch)
        logging.info("num_training_steps_per_epoch: %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        # ── Networks ────────────────────────────────────────────────────
        actor, critic = get_algorithm(
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

        actor_params = actor.init(actor_key, np.ones([1, obs_size]))
        critic_params = critic.init(critic_key, np.ones([1, obs_size]))

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        critic_states = critic.create_critic_states(critic_params, self.critic_lr)

        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        target_critic_params = None
        if self.agent_type == "sac":
            target_critic_params = critic_params

        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=actor_state,
            critic_states=critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )

        # ── Proposers ──────────────────────────────────────────────────
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name, unwrapped_env, config.num_envs,
            self.num_candidates,
            state_size=state_size,
            goal_indices=unwrapped_env.goal_indices,
            actor=actor, critic=critic, discounting=self.discounting,
        )
        reset_proposer = create_goal_proposer(
            self.reset_proposer_name, unwrapped_env, config.num_envs,
            self.num_candidates,
            state_size=state_size,
            goal_indices=unwrapped_env.goal_indices,
            actor=actor, critic=critic, discounting=self.discounting,
        )
        # Separate goal proposer for initial resets
        init_goal_proposer = create_goal_proposer(
            self.goal_proposer_name_initial, unwrapped_env, config.num_envs,
            self.num_candidates,
            state_size=state_size,
            goal_indices=unwrapped_env.goal_indices,
            actor=actor, critic=critic, discounting=self.discounting,
        )

        # ── Wrap train env with auto-reset ──────────────────────────────
        train_env = ResetExploreAutoResetWrapper(train_env)

        # ── Initial reset ────────────────────────────────────────────────
        random_goals_fn = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys = jax.random.split(env_key, config.num_envs)
        initial_goals = jax.vmap(random_goals_fn)(env_keys)

        # Sample initial start positions using environment's _random_start
        assert hasattr(unwrapped_env, "_random_start"), "Environment must have _random_start method"
        random_start_fn = jax.jit(lambda rng: unwrapped_env._random_start(rng))

        key, start_key = jax.random.split(key)
        start_keys = jax.random.split(start_key, config.num_envs)
        initial_starts = jax.vmap(random_start_fn)(start_keys)

        env_state = train_env.reset(env_keys, goal=initial_goals, start=initial_starts)
        info = dict(env_state.info)
        info['proposed_goals'] = initial_goals
        info['proposed_starts'] = initial_starts
        env_state = env_state.replace(info=info)

        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size

        # ── Replay buffer ────────────────────────────────────────────────
        dummy_transition = create_single_dummy_transition(
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
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
            agent_type=self.agent_type,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)

        dummy_gp_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
        )
        proposer_state = GoalProposerState(
            transitions_sample=dummy_gp_transition,
            actor_params=actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(critic_states)},
        )

        # ── actor_step ──────────────────────────────────────────────────
        def actor_step(actor_state, env, env_state, key, extra_fields,
                       is_deterministic: bool):
            action_key, env_rng = jax.random.split(key)
            actions = actor.sample_actions(
                actor_state.params, env_state.obs, action_key,
                is_deterministic=is_deterministic,
            )
            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=nstate.obs if self.agent_type == "sac" else None,
                extras={"state_extras": state_extras},
            )

        # ── get_experience ──────────────────────────────────────────────
        def get_experience(actor_state, critic_states, env_state, buffer_state,
                           key, experience_count, proposer_state, viz_env_steps,
                           is_deterministic: bool):
            buffer_state, transitions_sample = replay_buffer.sample(buffer_state)
            proposer_state = proposer_state.replace(
                transitions_sample=transitions_sample,
                actor_params=actor_state.params,
                critic_params={i: cs.params for i, cs in enumerate(critic_states)},
            )

            num_envs_ = config.num_envs
            episode_length = config.episode_length
            info = dict(env_state.info)

            reset_threshold = jnp.array(
                episode_length // (self.unroll_length * 2), dtype=jnp.int32,
            )
            new_experience_count = experience_count + 1

            # ---------------------------------------------------------
            def propose_new(env_state, key, info, experience_count, proposer_state):
                viz_key, goal_key, reset_key, prob_key, init_start_key = jax.random.split(key, 5)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs_)
                goal_keys = jax.random.split(goal_key, num_envs_)
                reset_keys = jax.random.split(reset_key, num_envs_)
                init_start_keys = jax.random.split(init_start_key, num_envs_)
                first_obs = info['first_obs']

                # Bernoulli mask for initial-distribution resets
                init_mask = (jax.random.uniform(prob_key, (num_envs_,)) < self.p_initial_reset)

                # Propose goals: both variants (initial/dynamic)
                def _propose_goal_dyn(rng, obs, state):
                    goal, _, log_data = goal_proposer(rng, obs, state)
                    return goal, log_data
                def _propose_goal_init(rng, obs, state):
                    goal, _, log_data = init_goal_proposer(rng, obs, state)
                    return goal, log_data

                dyn_goals, dyn_goal_log = jax.vmap(
                    _propose_goal_dyn, in_axes=(0, 0, None),
                )(goal_keys, first_obs, proposer_state)
                init_goals, init_goal_log = jax.vmap(
                    _propose_goal_init, in_axes=(0, 0, None),
                )(goal_keys, first_obs, proposer_state)

                # Propose starts (reset positions)
                def _propose_start(rng, obs, state):
                    start, _, _ = reset_proposer(rng, obs, state)
                    return start
                dyn_starts = jax.vmap(
                    _propose_start, in_axes=(0, 0, None),
                )(reset_keys, first_obs, proposer_state)
                init_starts = jax.vmap(random_start_fn)(init_start_keys)

                # Select per-env based on init_mask
                new_goals = jnp.where(init_mask[:, None], init_goals, dyn_goals)
                new_starts = jnp.where(init_mask[:, None], init_starts, dyn_starts)

                info['proposed_goals'] = new_goals
                info['proposed_starts'] = new_starts

                # Goal proposer visualisation:
                def log_viz(
                    dyn_log_np, init_log_np, goals_np, init_mask_np, viz_idx_np, viz_steps_np,
                ):
                    handle_reset_explore_visualization(
                        dyn_log_np,
                        init_log_np,
                        goals_np,
                        init_mask_np,
                        viz_idx_np,
                        self.goal_proposer_name,
                        self.goal_proposer_name_initial,
                        np.asarray(unwrapped_env.x_bounds),
                        np.asarray(unwrapped_env.y_bounds),
                        int(viz_steps_np),
                    )
                    return np.int32(0)

                jax.experimental.io_callback(
                    log_viz, jnp.array(0, dtype=jnp.int32),
                    dyn_goal_log, init_goal_log, new_goals, init_mask, viz_env_idx, viz_env_steps,
                )

                return (
                    env_state, info,
                    jnp.array(0, dtype=jnp.int32),
                    proposer_state,
                )

            def keep_existing(env_state, key, info, experience_count, proposer_state):
                return env_state, info, experience_count + 1, proposer_state

            # ---------------------------------------------------------
            env_state, info, updated_ec, updated_ps = jax.lax.cond(
                new_experience_count >= reset_threshold,
                propose_new, keep_existing,
                env_state, key, info, experience_count, proposer_state,
            )
            env_state = env_state.replace(info=info)

            @jax.jit
            def f(carry, _):
                env_state, buffer_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    actor_state, train_env, env_state, k,
                    extra_fields=("truncation", "traj_id"),
                    is_deterministic=is_deterministic,
                )
                return (env_state, buffer_state, next_k), transition

            (env_state, buffer_state, _), data = jax.lax.scan(
                f, (env_state, buffer_state, key), (), length=self.unroll_length,
            )
            buffer_state = replay_buffer.insert(buffer_state, data)
            return env_state, buffer_state, updated_ec, updated_ps

        # ── prefill ─────────────────────────────────────────────────────
        def prefill_replay_buffer(training_state, env_state, buffer_state,
                                  key, proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, ps = carry
                k, new_k = jax.random.split(k)
                es, bs, ec, ps = get_experience(
                    ts.actor_state, ts.critic_states, es, bs, k,
                    ts.experience_count, ps, ts.env_steps, is_deterministic=False,
                )
                ts = ts.replace(
                    env_steps=ts.env_steps + config.num_envs * self.unroll_length,
                    experience_count=ec,
                )
                return (ts, es, bs, new_k, ps), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, proposer_state),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ── update_networks ─────────────────────────────────────────────
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
            networks = dict(actor=actor, critic=critic)

            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics = actor.update(
                    context, networks, transitions, training_state, actor_key,
                )
                training_state, critic_metrics = critic.update(
                    context, networks, transitions, training_state, critic_key,
                )
            else:  # sac
                training_state, alpha_metrics = update_alpha_sac(
                    context, networks, transitions, training_state, alpha_key,
                )
                training_state, critic_metrics = critic.update(
                    context, networks, transitions, training_state, critic_key,
                )
                training_state, actor_metrics = actor.update(
                    context, networks, transitions, training_state, actor_key,
                )
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)

            # SAC target-network EMA
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                full_cp = {}
                for i, cs in enumerate(training_state.critic_states):
                    for lname, lparams in cs.params.items():
                        full_cp[f"critic_{i}_{lname}"] = lparams
                new_target = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params, full_cp,
                )
                training_state = training_state.replace(
                    target_critic_params=new_target,
                )

            training_state = training_state.replace(
                gradient_steps=training_state.gradient_steps + 1,
            )
            return (training_state, key), metrics

        # ── training_step ───────────────────────────────────────────────
        @jax.jit
        def training_step(training_state, env_state, buffer_state, key,
                          proposer_state):
            exp_key, process_key, train_key = jax.random.split(key, 3)

            env_state, buffer_state, updated_ec, updated_ps = get_experience(
                training_state.actor_state, training_state.critic_states,
                env_state, buffer_state, exp_key,
                training_state.experience_count, proposer_state, training_state.env_steps,
                is_deterministic=False,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
                experience_count=updated_ec,
            )

            buffer_state, transitions = replay_buffer.sample(buffer_state)
            transitions, _ = actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting,
                state_size, tuple(train_env.goal_indices),
                train_env.goal_reach_thresh, self.use_her,
            )
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions,
            )
            return (training_state, env_state, buffer_state, updated_ps), metrics

        # ── training_epoch ──────────────────────────────────────────────
        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key,
                           proposer_state):
            @jax.jit
            def f(carry, _):
                ts, es, bs, k, ps = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, ps), metrics = training_step(
                    ts, es, bs, train_key, ps,
                )
                return (ts, es, bs, k, ps), metrics

            (training_state, env_state, buffer_state, key, proposer_state), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, proposer_state),
                (),
                length=num_training_steps_per_epoch,
            )
            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, proposer_state, metrics

        # ── Prefill ─────────────────────────────────────────────────────
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, proposer_state = (
            prefill_replay_buffer(
                training_state, env_state, buffer_state,
                prefill_key, proposer_state,
            )
        )

        # ── Evaluator ──────────────────────────────────────────────────
        evaluator = ActorEvaluator(
            lambda ts, env, es, extra_fields=(): actor_step(
                ts.actor_state, env, es,
                jax.random.PRNGKey(0), extra_fields, is_deterministic=True,
            ),
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ── Main loop ──────────────────────────────────────────────────
        training_walltime = 0
        last_visualization_step = -1
        logging.info("starting training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, proposer_state, metrics = (
                training_epoch(
                    training_state, env_state, buffer_state,
                    epoch_key, proposer_state,
                )
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(
                lambda x: x.block_until_ready(), metrics,
            )

            epoch_time = time.time() - t
            training_walltime += epoch_time
            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_time

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            # Visualise buffer every 1M steps
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
                last_visualization_step = current_step

            do_render = ne % config.visualization_interval == 0

            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            else:
                make_policy = lambda param: lambda obs, rng: (
                    actor.sample_actions(param, obs, rng, is_deterministic=True), {}
                )

            # Build full critic params for checkpoint
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
                current_step, metrics, make_policy,
                training_state.actor_state.params, unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)
        return make_policy, params, metrics
