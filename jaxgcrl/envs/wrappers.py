import jax
from brax.envs import PipelineEnv, State, Wrapper, Env
from jax import numpy as jnp
from jax import tree_util
from typing import Callable, Any, Optional

class TrajectoryIdWrapper(Wrapper):
    def __init__(self, env: PipelineEnv):
        super().__init__(env)

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal)
        # Increment traj_id instead of setting to 0
        if "traj_id" not in state.info:
            state.info["traj_id"] = jnp.zeros(rng.shape[:-1])
        else:
            state.info["traj_id"] = state.info["traj_id"] + 1
        return state

    def step(self, state: State, action: jax.Array) -> State:
        if "steps" in state.info.keys():
            traj_id = state.info["traj_id"] + jnp.where(state.info["steps"], 0, 1)
        else:
            traj_id = state.info["traj_id"]
        state = self.env.step(state, action)
        state.info["traj_id"] = traj_id
        return state


class EvalAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array, rng=None) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)

class VmapWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None,
              start: Optional[jnp.ndarray] = None) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        if goal is None and start is None:
            return jax.vmap(lambda r: self.env.reset(r))(rng)
        elif goal is not None and start is None:
            return jax.vmap(lambda r, g: self.env.reset(r, goal=g))(rng, goal)
        elif goal is None and start is not None:
            return jax.vmap(lambda r, s: self.env.reset(r, start=s))(rng, start)
        else:
            return jax.vmap(lambda r, g, s: self.env.reset(r, goal=g, start=s))(rng, goal, start)

    def step(self, state: State, action: jax.Array) -> State:
        return jax.vmap(self.env.step)(state, action)


class EpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None,
              start: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal, start=start)
        state.info['steps'] = jnp.zeros(rng.shape[:-1])
        state.info['truncation'] = jnp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jnp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jnp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jnp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jnp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jnp.ones_like(state.done)
        zero = jnp.zeros_like(state.done)
        episode_length = jnp.array(self.episode_length, dtype=jnp.int32)
        done = jnp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jnp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['sum_reward'] += jnp.sum(rewards, axis=0)
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
        state.info['episode_done'] = done
        return state.replace(done=done)


class TrainAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array, goal: jnp.ndarray) -> State:
        proposed_goals = goal
        state = self.env.reset(rng, goal=proposed_goals)
        state.info['proposed_goals'] = proposed_goals
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        return state

    def step(self, state: State, action: jax.Array, rng: jax.Array) -> State:
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))
        state = self.env.step(state, action)

        done = state.done
        
        # Reset done environments using goals from info
        def reset_done_envs(state, done, info, rng):
            # Split single rng key into num_envs keys (will be vmapped by VmapWrapper)
            num_envs = done.shape[0] if done.shape else 1
            reset_rng = jax.random.split(rng, num_envs)
            
            # Get proposed goals from info (should always be present)
            # proposed_goals is always maintained in info by baseline
            proposed_goals = info['proposed_goals']
            
            # Reset all envs with goals from info
            reset_state = self.env.reset(reset_rng, goal=proposed_goals)
            
            # Update first_pipeline_state and first_obs for done envs
            def where_done_reset(x_reset, x_current):
                if x_reset.shape and x_reset.shape[0] != done.shape[0]:
                    return x_current
                if done.shape:
                    done_reshaped = jnp.reshape(done, [done.shape[0]] + [1] * (len(x_reset.shape) - 1))
                else:
                    done_reshaped = done
                return jnp.where(done_reshaped, x_reset, x_current)
            
            # Update info dict fields for done envs from reset_state
            info['first_pipeline_state'] = jax.tree.map(
                where_done_reset, reset_state.pipeline_state, info.get('first_pipeline_state', state.info.get('first_pipeline_state', state.pipeline_state))
            )
            info['first_obs'] = jax.tree.map(
                where_done_reset, reset_state.obs, info.get('first_obs', state.info.get('first_obs', state.obs))
            )
            
            # Update traj_id for done envs (handled by TrajectoryIdWrapper in reset)
            info['traj_id'] = jax.tree.map(
                where_done_reset, reset_state.info['traj_id'], info.get('traj_id', state.info.get('traj_id'))
            )

            info['proposed_goals'] = proposed_goals
            
            return state.replace(info=info)
        
        def no_reset(state, done, info, rng):
            # Just preserve the existing info dict (including proposed_goals)
            return state.replace(info=info)
        
        # Only reset if any env is done (JIT-compatible)
        info = dict(state.info)
        state = jax.lax.cond(
            jnp.any(done),
            reset_done_envs,
            no_reset,
            state, done, info, rng
        )

        def where_done(x, y):
            done = state.done
            if done.shape and done.shape[0] != x.shape[0]:
                return y
            if done.shape:
                done = jnp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jnp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        return state.replace(pipeline_state=pipeline_state, obs=obs)


class GoExploreWrapper(Wrapper):
    """Outermost training wrapper for the Go Explore algorithm.

    Replaces ``TrainAutoResetWrapper``.  Responsibilities:
    - Tracks per-environment phase (0 = go, 1 = explore) and phase_step.
    - Increments ``traj_id`` on go→explore (+1) and explore→go / ep_done (+2).
    - Resets obs/pipeline_state to the start-of-training state after the
      explore phase (or on episode done).
    - Updates ``go_goal`` from ``proposed_goals`` when a new go phase begins.
    - Stores ``explore_first_obs`` (obs at go-phase start) for explore reward.
    - Sets ``truncation=1`` at phase boundaries so trajectory boundaries are
      respected in the replay buffer.

    Must sit *outside* ``EpisodeWrapper`` and ``VmapWrapper`` in the env stack.
    The eval env should *not* use this wrapper.

    Note: ``TrajectoryIdWrapper`` should *not* be included in the Go Explore
    training env stack; this wrapper manages ``traj_id`` directly.
    """

    def __init__(self, env: Env, num_gcp_steps: int, num_ep_steps: int,
                 state_size: int, goal_size: int, goal_indices=None):
        super().__init__(env)
        self.num_gcp_steps = num_gcp_steps
        self.num_ep_steps = num_ep_steps
        self.state_size = state_size
        self.goal_size = goal_size
        # Indices into the state portion of obs that select the (x, y) position
        # used as the ant's start position (passed to env.reset(start=...)).
        self.goal_indices = goal_indices

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, rng: jax.Array, goal: Optional[jnp.ndarray] = None) -> State:
        state = self.env.reset(rng, goal=goal)
        num_envs = rng.shape[0]  # rng: (num_envs, 2)

        if goal is not None:
            go_goal = goal  # (num_envs, goal_size)
        else:
            go_goal = state.obs[:, self.state_size:]  # extract from obs

        state.info['phase']               = jnp.zeros(num_envs, dtype=jnp.int32)
        state.info['phase_step']          = jnp.zeros(num_envs, dtype=jnp.int32)
        state.info['go_goal']             = go_goal
        state.info['go_phase_success']    = jnp.zeros(num_envs, dtype=jnp.float32)
        state.info['go_phase_steps']      = jnp.zeros(num_envs, dtype=jnp.float32)
        # Cache the start position (at goal_indices) so env.reset(start=first_start)
        # can restore it across go phases without manual pipeline-state surgery.
        state.info['first_start'] = state.obs[:, self.goal_indices]
        state.info['first_obs']           = state.obs
        state.info['explore_first_obs']   = state.obs
        state.info['pre_reset_obs']       = state.obs
        state.info['traj_id']             = jnp.zeros(num_envs, dtype=jnp.float32)
        state.info['proposed_goals']      = go_goal
        # Cumulative counters for Bug 2 (never reset within step, only in reset)
        state.info['go_completions_total']   = jnp.zeros(num_envs, dtype=jnp.float32)
        state.info['go_successes_total']     = jnp.zeros(num_envs, dtype=jnp.float32)
        state.info['go_success_steps_total'] = jnp.zeros(num_envs, dtype=jnp.float32)
        return state

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, state: State, action: jax.Array, rng: jax.Array = None) -> State:
        # ── 0. Zero `steps` for previously done envs ─────────────────────────────
        if 'steps' in state.info:
            steps = state.info['steps']
            steps = jnp.where(state.done, jnp.zeros_like(steps), steps)
            state.info.update(steps=steps)
        state = state.replace(done=jnp.zeros_like(state.done))

        # ── 1. Step underlying env ────────────────────────────────────────────────
        nstate = self.env.step(state, action)
        info = dict(nstate.info)

        # ── 2. Read phase state from *before* this step ───────────────────────────
        phase      = state.info['phase']       # (num_envs,) int32
        phase_step = state.info['phase_step']  # (num_envs,) int32
        in_go      = (phase == 0)
        in_explore = (phase == 1)

        # ── 3. Detect success (go phase) ──────────────────────────────────────────
        success      = nstate.metrics.get('success', jnp.zeros(phase.shape, dtype=jnp.float32))
        goal_reached = (success > 0.5)

        # ── 4. Phase-transition predicates ────────────────────────────────────────
        # (phase_step + 1) because we are about to increment phase_step.
        ep_done              = nstate.done.astype(bool)
        # Bug 3 fix: ep_done takes priority — mask it out of go→explore so a
        # simultaneous episode termination always routes through should_reset (go phase).
        should_go_to_explore = in_go & (goal_reached | ((phase_step + 1) >= self.num_gcp_steps)) & ~ep_done
        should_explore_to_go = in_explore & ((phase_step + 1) >= self.num_ep_steps)

        # Any condition that resets to the go-phase start state
        should_reset = should_explore_to_go | ep_done

        # ── 5. Go-phase success metrics ───────────────────────────────────────────
        go_success = in_go & goal_reached
        # Point-in-time snapshot fields (kept for backward compat / debugging)
        new_go_phase_success = jnp.where(
            should_go_to_explore,
            go_success.astype(jnp.float32),
            jnp.where(
                should_reset,
                jnp.zeros_like(state.info['go_phase_success']),
                state.info['go_phase_success'],
            ),
        )
        new_go_phase_steps = jnp.where(
            should_go_to_explore & go_success,
            (phase_step + 1).astype(jnp.float32),
            jnp.where(
                should_reset,
                jnp.zeros_like(state.info['go_phase_steps']),
                state.info['go_phase_steps'],
            ),
        )
        # Bug 2 fix: cumulative counters — incremented only when a go phase
        # completes (should_go_to_explore), never reset within step().
        new_go_completions_total   = (state.info['go_completions_total']
                                      + should_go_to_explore.astype(jnp.float32))
        new_go_successes_total     = (state.info['go_successes_total']
                                      + go_success.astype(jnp.float32))
        new_go_success_steps_total = (state.info['go_success_steps_total']
                                      + jnp.where(go_success,
                                                   (phase_step + 1).astype(jnp.float32),
                                                   0.0))

        # ── 6. New phase and phase_step ───────────────────────────────────────────
        new_phase = jnp.where(
            should_go_to_explore,
            jnp.ones_like(phase),
            jnp.where(should_reset, jnp.zeros_like(phase), phase),
        )
        new_phase_step = jnp.where(
            should_go_to_explore | should_reset,
            jnp.zeros_like(phase_step),
            phase_step + 1,
        )

        # ── 7. Trajectory ID increments ───────────────────────────────────────────
        traj_id = state.info['traj_id']
        traj_id = traj_id + jnp.where(should_go_to_explore, 1.0, 0.0)
        traj_id = traj_id + jnp.where(should_reset, 2.0, 0.0)

        # ── 8. Update go_goal on new go phase ─────────────────────────────────────
        proposed_goals = state.info.get('proposed_goals', state.info['go_goal'])
        new_go_goal = jnp.where(should_reset[:, None], proposed_goals, state.info['go_goal'])

        # ── 9. Update explore_first_obs on go→explore ─────────────────────────────
        new_explore_first_obs = jnp.where(
            should_go_to_explore[:, None],
            state.info['first_obs'],
            state.info['explore_first_obs'],
        )

        # ── 10. Restore physics to start state on reset via proper env.reset() ──────
        # Bug 1 fix: instead of restoring cached pipeline state (which has the
        # original goal baked into q), call env.reset(start=first_start, goal=new_goal)
        # so pipeline_init regenerates correct q/qd with the new goal embedded.
        # This matches how TrainAutoResetWrapper resets via the env API.
        first_start = state.info['first_start']   # (num_envs, 2) — ant xy at episode start
        num_envs_local = state.obs.shape[0]
        # Split rng for per-env resets (rng may be a single key from actor_step)
        reset_rng = jax.random.split(rng, num_envs_local)  # (num_envs, 2)
        # Reset ALL envs unconditionally (JAX traces the branch regardless);
        # _where_masked selects results only for envs where should_reset is True.
        reset_state = self.env.reset(reset_rng, goal=new_go_goal, start=first_start)

        def _where_masked(x_reset, x_current):
            if not hasattr(x_reset, 'shape'):
                return x_current
            if x_reset.ndim == 0:
                return jnp.where(should_reset[0], x_reset, x_current)
            if x_reset.shape[0] != should_reset.shape[0]:
                return x_current
            mask = jnp.reshape(should_reset, [should_reset.shape[0]] + [1] * (x_reset.ndim - 1))
            return jnp.where(mask, x_reset, x_current)

        reset_pipeline_state = jax.tree.map(_where_masked, reset_state.pipeline_state, nstate.pipeline_state)
        reset_obs = jnp.where(should_reset[:, None], reset_state.obs, nstate.obs)

        # first_obs for the new go phase now has the correct goal baked in
        new_first_obs = jnp.where(should_reset[:, None], reset_state.obs, state.info['first_obs'])

        # ── 11. Mark phase boundaries as truncations ─────────────────────────────
        phase_boundary = (should_go_to_explore | should_reset).astype(jnp.float32)
        existing_trunc = info.get('truncation', jnp.zeros(phase.shape, dtype=jnp.float32))
        info['truncation'] = jnp.maximum(existing_trunc, phase_boundary)

        # ── 12. Write updated fields back to info ────────────────────────────────
        info['phase']                   = new_phase
        info['phase_step']              = new_phase_step
        info['traj_id']                 = traj_id
        info['go_goal']                 = new_go_goal
        info['go_phase_success']        = new_go_phase_success
        info['go_phase_steps']          = new_go_phase_steps
        info['go_completions_total']    = new_go_completions_total
        info['go_successes_total']      = new_go_successes_total
        info['go_success_steps_total']  = new_go_success_steps_total
        info['explore_first_obs']       = new_explore_first_obs
        info['first_obs']               = new_first_obs
        info['first_start']             = first_start   # unchanged, carry forward
        info['proposed_goals']          = proposed_goals
        info['pre_reset_obs']           = nstate.obs

        return nstate.replace(pipeline_state=reset_pipeline_state, obs=reset_obs, info=info)