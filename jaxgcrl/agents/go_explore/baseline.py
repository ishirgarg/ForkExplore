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
    TrainAutoResetWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization

from .types import Actor, Critic, TrainingState, Transition, GoalProposerState
from .algorithms import get_algorithm
from .utils import save_params, create_single_dummy_transition, create_dummy_transition_for_buffer, create_dummy_transition_for_goal_proposer
from .losses import update_alpha_sac, update_critic_sac, update_actor_sac
from .visualization import all_visualizations
from .goal_proposers import create_goal_proposer, create_random_env_goals_proposer

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]

@dataclass
class Baseline:
    """Unified baseline agent supporting both CRL and SAC algorithms.
    
    This agent can be configured to run either CRL (Contrastive Reinforcement Learning)
    or SAC (Soft Actor-Critic) by setting the agent_type parameter.
    """

    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 256

    # gamma
    discounting: float = 0.99

    # forward CRL logsumexp penalty
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

    # phi(s,a) and psi(g) repr dimension
    repr_dim: int = 64

    # layer norm
    use_ln: bool = True # NOTE: for CRL we don't apply layer norm to the actor regardless

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    # SAC specific
    tau: float = 0.005
    n_critics: int = 2 # Also for q epistemic
    use_her: bool = True  # Hindsight Experience Replay

    # goal proposer for training
    goal_proposer_name: Literal["random_env_goals", "rb", "q_epistemic", "ucgr"] = "random_env_goals"
    num_candidates: int = 512  # Number of candidate goals to filter before final selection

    def check_config(self, config):
        """
        episode_length: the maximum length of an episode
            NOTE: `num_envs * (episode_length - 1)` must be divisible by
            `batch_size` due to the way data is stored in replay buffer.
            NOTE: `episode_length` must be divisible by `unroll_length`.
        """
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        # Removed assertion about episode_length / unroll_length since we're using auto-resets with unroll_length

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
        
        # Compute dimensions early (needed for dummy transition before wrapper creation)
        action_size = train_env.action_size
        state_size = train_env.state_dim
        goal_size = len(train_env.goal_indices)
        obs_size = state_size + goal_size
        
       
        # Wrap envs explicitly (mirrors brax.envs.training.wrap), but with TrajectoryIdWrapper innermost:
        # inner -> outer: TrajectoryIdWrapper -> VmapWrapper -> EpisodeWrapper -> (Train/Eval)AutoResetWrapper
        # Note: TrainAutoResetWrapper will be added after actor and critic are created (they're needed for goal_proposer)
        train_env = TrajectoryIdWrapper(train_env)
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)

        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # Use unroll_length directly (original behavior with auto-resets)
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps = self.min_replay_size * config.num_envs
        num_prefill_actor_steps = int(np.ceil(self.min_replay_size / self.unroll_length))
        
        # Calculate training steps per epoch
        # Available env steps for training = total - prefill
        available_env_steps = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0, (
            "total_env_steps too small for given num_envs and episode_length"
        )

        logging.info(
            "num_prefill_env_steps: %d",
            num_prefill_env_steps,
        )
        logging.info(
            "num_prefill_actor_steps: %d",
            num_prefill_actor_steps,
        )
        logging.info(
            "env_steps_per_epoch: %d",
            env_steps_per_epoch,
        )
        logging.info(
            "num_training_steps_per_epoch: %d",
            num_training_steps_per_epoch,
        )

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        # Network setup
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
            n_critics=self.n_critics,  # Number of critics in ensemble (for both CRL and SAC)
        )

        actor_params = actor.init(actor_key, np.ones([1, obs_size]))
        critic_params = critic.init(critic_key, np.ones([1, obs_size]))

        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )

        # Create separate TrainState for each critic using the critic's class method
        critic_states = critic.create_critic_states(critic_params, self.critic_lr)

        # Entropy coefficient
        target_entropy = -0.5 * action_size
        log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        # SAC-specific initialization
        target_critic_params = None
        if self.agent_type == "sac":
            # Initialize target Q-network params (copy of critic params)
            target_critic_params = critic_params

        # Trainstate
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=actor_state,
            critic_states=critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
        )
        
        # Now create goal proposer with actor and critic objects (captured in closure)
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
            self.num_candidates,
            state_size=unwrapped_env.state_dim,
            goal_indices=unwrapped_env.goal_indices,
            actor=actor,
            critic=critic,
            discounting=self.discounting,
        )
        
        # Wrap train_env with TrainAutoResetWrapper (no goal_proposer needed - goals stored in info)
        train_env = TrainAutoResetWrapper(train_env)

        # Propose random environment goals for the first reset
        random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys = jax.random.split(env_key, config.num_envs)

        initial_goals = jax.vmap(random_goals_proposer)(env_keys)
        
        env_state = train_env.reset(env_keys, goal=initial_goals)
        # Initialize the entire info dict with proposed_goals and traj_id
        # to maintain consistent pytree structure in scans
        info = env_state.info.copy() if hasattr(env_state.info, 'copy') else dict(env_state.info)
        info['proposed_goals'] = initial_goals

        env_state = env_state.replace(info=info)
        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # Replay Buffer
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
        
        # Add dummy transition to buffer at initialization so we can always sample
        dummy_batch_transition = create_dummy_transition_for_buffer(
            unroll_length=self.unroll_length,
            num_envs=config.num_envs,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)
        
        # Initialize goal proposer state with dummy transition matching replay buffer sample shape
        # Replay buffer sample returns (num_envs, episode_length, ...), not (unroll_length, num_envs, ...)
        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=self.agent_type,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=actor_state.params,
            critic_params={i: critic_i_state.params for i, critic_i_state in enumerate(critic_states)},
        )

        def actor_step(actor_state, env, env_state, key, extra_fields, is_deterministic: bool):
            # Split key: one for action sampling, one for environment step (for resets)
            action_key, env_rng = jax.random.split(key)
            actions = actor.sample_actions(
                actor_state.params,
                env_state.obs,
                action_key,
                is_deterministic=is_deterministic
            )
            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs, 
                action=actions, 
                reward=nstate.reward, 
                discount=1 - nstate.done,
                next_observation=nstate.obs if self.agent_type == "sac" else None,  # SAC needs next_observation
                extras={"state_extras": state_extras}
            )

        def get_experience(actor_state, critic_states, env_state, buffer_state, key, experience_count, goal_proposer_state, is_deterministic: bool):
            # Always sample from buffer (dummy data was added at initialization)
            buffer_state, transitions_sample = replay_buffer.sample(buffer_state)
            
            # Update goal proposer state with new replay buffer sample and current network params
            goal_proposer_state = goal_proposer_state.replace(
                transitions_sample=transitions_sample,
                actor_params=actor_state.params,
                critic_params={i: critic_i_state.params for i, critic_i_state in enumerate(critic_states)},
            )
            
            num_envs = config.num_envs
            episode_length = config.episode_length
            
            info = dict(env_state.info)
            
            reset_threshold = jnp.array(episode_length // (self.unroll_length * 2), dtype=jnp.int32)
            new_experience_count = experience_count + 1
            
            # Propose new goals if we've reached the threshold
            def propose_new_goals(env_state, key, info, experience_count, goal_proposer_state):
                # Split key: one for selecting viz env, one for goal proposer keys
                viz_key, goal_key = jax.random.split(key)
                # Randomly select one environment for visualization
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs)
                goal_keys = jax.random.split(goal_key, num_envs)
                first_obs = info['first_obs']
                # Call goal proposer to get new goals (vmapped over envs)
                def propose_single_goal(rng_key, obs, state):
                    goal, updated_state, log_data = goal_proposer(rng_key, obs, state)
                    return goal, log_data
                
                new_goals, log_data_tree = jax.vmap(
                    propose_single_goal, in_axes=(0, 0, None)
                )(goal_keys, first_obs, goal_proposer_state)
                info['proposed_goals'] = new_goals
                
                # Log visualization using io_callback (select one environment's log_data)
                # Capture goal_proposer_name from closure (static value, not traced)
                def log_visualization(log_data_tree_np, viz_idx):
                    selected_log_data = {}
                    for key, value in log_data_tree_np.items():
                        selected_log_data[key] = value[viz_idx]
                    handle_goal_proposer_visualization(selected_log_data, self.goal_proposer_name, unwrapped_env.x_bounds, unwrapped_env.y_bounds)
                    return jnp.array(0, dtype=jnp.int32)
                
                # Use io_callback to log visualization
                jax.experimental.io_callback(
                    log_visualization,
                    jnp.array(0, dtype=jnp.int32),
                    log_data_tree,
                    viz_env_idx
                )
                
                # Reset counter to 0 after proposing new goals
                updated_experience_count = jnp.array(0, dtype=jnp.int32)
                # Return the same goal_proposer_state (it's updated elsewhere with new transitions_sample)
                return env_state, info, updated_experience_count, goal_proposer_state
            
            def keep_existing_goals(env_state, key, info, experience_count, goal_proposer_state):
                # Increment counter
                updated_experience_count = experience_count + 1
                return env_state, info, updated_experience_count, goal_proposer_state
            
            # Conditionally propose new goals
            env_state, info, updated_experience_count, updated_goal_proposer_state = jax.lax.cond(
                new_experience_count >= reset_threshold,
                propose_new_goals,
                keep_existing_goals,
                env_state, key, info, experience_count, goal_proposer_state
            )
            
            env_state = env_state.replace(info=info)
            
            @jax.jit
            def f(carry, unused_t):
                env_state, buffer_state, current_key = carry
                current_key, next_key = jax.random.split(current_key)
                env_state, transition = actor_step(
                    actor_state,
                    train_env,
                    env_state,
                    current_key,
                    extra_fields=("truncation", "traj_id"),
                    is_deterministic=is_deterministic,
                )
                # transitions_sample flows through state.info (JAX PyTree, JIT-compatible)
                # When reset() is called during step() (auto-reset), it reads proposed_goals
                # from state.info and uses them for reset
                return (env_state, buffer_state, next_key), transition

            (env_state, buffer_state, _), data = jax.lax.scan(f, (env_state, buffer_state, key), (), length=self.unroll_length)

            # Goal proposing happens before the scan, so new goals are available during the scan
            # Auto reset wrapper will use proposed_goals from info when resets occur
            
            buffer_state = replay_buffer.insert(buffer_state, data)
            
            return env_state, buffer_state, updated_experience_count, updated_goal_proposer_state
            

        def prefill_replay_buffer(training_state, env_state, buffer_state, key, goal_proposer_state):
            # get_experience will update goal_proposer_state before each scan
            @jax.jit
            def f(carry, unused):
                del unused
                training_state, env_state, buffer_state, key, goal_proposer_state = carry
                key, new_key = jax.random.split(key)
                env_state, buffer_state, updated_experience_count, updated_goal_proposer_state = get_experience(
                    training_state.actor_state,
                    training_state.critic_states,
                    env_state,
                    buffer_state,
                    key,
                    training_state.experience_count,
                    goal_proposer_state,
                    is_deterministic=False,
                )
                # Prefill uses unroll_length steps per call
                training_state = training_state.replace(
                    env_steps=training_state.env_steps + config.num_envs * self.unroll_length,
                    experience_count=updated_experience_count,
                )
                return (training_state, env_state, buffer_state, new_key, updated_goal_proposer_state), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_prefill_actor_steps,
            )[0]

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

            networks = dict(
                actor=actor,
                critic=critic,
            )

            # Update order: match original implementations exactly
            # CRL: actor (with alpha) then critic (original CRL order)
            # SAC: alpha, critic, then actor (original SAC order - line 362, 371, 381)
            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics = actor.update(context, networks, transitions, training_state, actor_key)
                training_state, critic_metrics = critic.update(context, networks, transitions, training_state, critic_key)
            elif self.agent_type == "sac":  # SAC
                # SAC updates: alpha first, then critic, then actor (matching original)
                training_state, alpha_metrics = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
                training_state, critic_metrics = critic.update(context, networks, transitions, training_state, critic_key)
                training_state, actor_metrics = actor.update(context, networks, transitions, training_state, actor_key)
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)
            
            # Update target networks for SAC
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                # Reconstruct full critic params from separate critic states (SAC structure)
                full_critic_params = {}
                for i, critic_i_state in enumerate(training_state.critic_states):
                    for layer_name, layer_params in critic_i_state.params.items():
                        full_critic_params[f"critic_{i}_{layer_name}"] = layer_params
                
                new_target_critic_params = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params,
                    full_critic_params,
                )
                training_state = training_state.replace(target_critic_params=new_target_critic_params)
            
            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)

            return (
                training_state,
                key,
            ), metrics

        @jax.jit
        def training_step(training_state, env_state, buffer_state, key, goal_proposer_state):
            experience_key1, process_key, training_key = jax.random.split(key, 3)

            # Collect unroll_length steps (with auto-resets)
            # get_experience will update goal_proposer_state before the scan
            env_state, buffer_state, updated_experience_count, updated_goal_proposer_state = get_experience(
                training_state.actor_state,
                training_state.critic_states,
                env_state,
                buffer_state,
                experience_key1,
                training_state.experience_count,
                goal_proposer_state,
                is_deterministic=False,
            )

            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
                experience_count=updated_experience_count,
            )

            # sample actor-step worth of transitions
            buffer_state, transitions = replay_buffer.sample(buffer_state)
            
            # Process transitions (algorithm-specific: flatten_batch, reshape, permute)
            transitions, _ = actor.process_transitions(
                transitions, process_key, self.batch_size, self.discounting, state_size, 
                tuple(train_env.goal_indices), train_env.goal_reach_thresh, self.use_her
            )
            
            # take actor-step worth of training-step
            (
                (
                    training_state,
                    _,
                ),
                metrics,
            ) = jax.lax.scan(update_networks, (training_state, training_key), transitions)
            
            return (
                training_state,
                env_state,
                buffer_state,
                updated_goal_proposer_state,
            ), metrics


        @jax.jit
        def training_epoch(
            training_state,
            env_state,
            buffer_state,
            key,
            goal_proposer_state,
        ):
            @jax.jit
            def f(carry, unused_t):
                ts, es, bs, k, gps = carry
                k, train_key = jax.random.split(k, 2)
                (
                    (ts, es, bs, updated_gps),
                    metrics,
                ) = training_step(ts, es, bs, train_key, gps)
                return (ts, es, bs, k, updated_gps), metrics

            (training_state, env_state, buffer_state, key, goal_proposer_state), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state),
                (),
                length=num_training_steps_per_epoch,
            )

            metrics["buffer_current_size"] = replay_buffer.size(buffer_state)
            return training_state, env_state, buffer_state, goal_proposer_state, metrics

        key, prefill_key = jax.random.split(key, 2)

        training_state, env_state, buffer_state, _, goal_proposer_state = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state
        )

        """Setting up evaluator"""
        evaluator = ActorEvaluator(
            lambda ts, env, es, extra_fields=(): actor_step(ts.actor_state, env, es, jax.random.PRNGKey(0), extra_fields, is_deterministic=True),
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        training_walltime = 0
        last_visualization_step = -1  # Track last step we visualized at
        logging.info("starting training....")
        for ne in range(config.num_evals):
            t = time.time()

            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, goal_proposer_state, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key, goal_proposer_state
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time = time.time() - t
            training_walltime += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time
            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **{f"training/{name}": value for name, value in metrics.items()},
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            do_render = ne % config.visualization_interval == 0
            # For CRL: return (mean, log_std) from actor.apply (matches CRL's pattern)
            # For SAC: return (action, {}) from actor.sample_actions (matches SAC's pattern)
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: actor.apply(param, obs)
            elif self.agent_type == "sac":  # SAC
                make_policy = lambda param: lambda obs, rng: (actor.sample_actions(param, obs, rng, is_deterministic=True), {})

            # Visualize trajectories every 1M steps (robust check that handles step skips)
            # Check if we've crossed a 1M boundary since last visualization
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

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

            # Prepare params for return (and optionally save if checkpointing)
            # Reconstruct full critic params from separate critic states
            # CRL uses sa_encoder_{i}/g_encoder_{i}, SAC uses critic_{i}_hidden_{j}/critic_{i}_output
            full_critic_params = {}
            for i, critic_i_state in enumerate(training_state.critic_states):
                for layer_name, layer_params in critic_i_state.params.items():
                    # Check if this is CRL structure (sa_encoder/g_encoder) or SAC structure (hidden/output)
                    if layer_name in ["sa_encoder", "g_encoder"]:
                        full_critic_params[f"{layer_name}_{i}"] = layer_params
                    else:
                        # SAC structure: add critic_{i}_ prefix
                        full_critic_params[f"critic_{i}_{layer_name}"] = layer_params
            
            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                full_critic_params,
            )
            
            if config.checkpoint_logdir:
                # Save current policy and critic params.
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

        total_steps = current_step
        # assert total_steps >= config.total_env_steps

        logging.info("total steps: %s", total_steps)

        return make_policy, params, metrics