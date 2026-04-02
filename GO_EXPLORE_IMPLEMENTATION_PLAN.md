# Go Explore Implementation Plan

## Overview
Implement Go Explore algorithm in `goal_proposers.py` with two phases:
- **Go Phase**: Goal-conditioned policy (no noise) that tries to reach a specific goal
- **Explore Phase**: Non-goal-conditioned policy (with noise) that explores the environment

## Key Requirements

### Phase Management
1. **Go Phase**:
   - Uses goal-conditioned policy (GCP) with `is_deterministic=True` (no noise)
   - Runs until either:
     - Goal is reached (success detected via `state.metrics['success']`)
     - Maximum steps reached (`num_gcp_steps` config parameter)
   - When goal is reached OR max steps reached → switch to Explore Phase
   - Starts at the beginning of training
   
2. **Explore Phase**:
   - Uses separate non-goal-conditioned explore policy (created via factory, similar to goal-conditioned policy)
   - Runs with noise (`is_deterministic=False`)
   - Reward is dynamic and configurable via factory method (default: `q_epistemic_reward`)
   - Runs for `num_ep_steps` steps, then resets to start state
   - After reset → switch back to Go Phase with new goal

### State Management
3. **Environment State Extensions**:
   - Add to `env_state.info`:
     - `phase`: jnp.ndarray indicating current phase (0=go, 1=explore)
     - `phase_step`: jnp.ndarray counting steps in current phase
     - `go_goal`: jnp.ndarray storing current goal for go phase
     - `go_phase_started`: jnp.ndarray boolean indicating if go phase has started
     - `explore_phase_started`: jnp.ndarray boolean indicating if explore phase has started
     - `go_phase_success`: jnp.ndarray boolean tracking if go phase succeeded
     - `go_phase_steps`: jnp.ndarray tracking number of steps in successful go phases

4. **Trajectory IDs**:
   - Go and explore phases must have separate trajectory IDs
   - When switching from go → explore, increment traj_id
   - When resetting after explore phase, increment traj_id again
   - This ensures trajectories are properly segmented in replay buffer

### Wrapper Changes
5. **New Wrapper**: `GoExploreWrapper`
   - Create new wrapper in `jaxgcrl/envs/wrappers.py` (don't modify existing wrappers)
   - Wrapper should:
     - Track phase state in `info` dict
     - Handle phase transitions
     - Manage trajectory ID increments on phase switches
     - Handle goal setting for go phase
     - Handle reward modification for explore phase (if needed)
     - Reset to start state after explore phase completes
   - Must be JIT-compatible (use `jax.lax.cond` for conditionals, not Python if)

### Policy Selection
6. **Policy Selection Logic**:
   - In `actor_step` function (or similar), check `env_state.info['phase']`
   - If phase == 0 (go): use goal-conditioned policy (GCP) with goal from `info['go_goal']`
   - If phase == 1 (explore): use separate explore policy (non-goal-conditioned, created via factory)
   - Go phase: `is_deterministic=True`
   - Explore phase: `is_deterministic=False`
   - **Explore Policy Factory**: Create explore policy similar to `get_algorithm()` factory
     - Should support different explore policy types (e.g., "sac", "random")
     - Takes same network architecture parameters as goal-conditioned policy
     - Returns explore actor (non-goal-conditioned, takes only state as input)

### Goal Proposing
7. **Goal Proposing**:
   - Goals are proposed at the start of each Go Phase
   - Use existing goal proposer infrastructure
   - Store proposed goal in `env_state.info['go_goal']`
   - Goal should be part of observation during go phase

### Reward Handling
8. **Reward Modification**:
   - Go phase: use environment's natural reward (typically goal-reaching reward)
   - Explore phase: use configurable reward function via factory method
   - **Default Explore Reward: `q_epistemic_reward`**
     - For state `w` in explore phase, compute Q-value variance across critic ensemble
     - Use `first_obs` from go phase as start state, and current state `w` as goal
     - Construct observation: `obs = [first_obs_state, w_goal]` where:
       - `first_obs_state = first_obs[:state_size]` (state portion)
       - `w_goal = w[:state_size][goal_indices]` (goal portion from current state)
     - Sample action deterministically from goal-conditioned policy: `action = gcp_actor.sample_actions(params, obs, key, is_deterministic=True)`
     - Compute Q-values using unified critic interface: `q_values = critic.apply(full_critic_params, obs, action)`
     - Shape: `(n_critics,)` - one Q-value per critic in ensemble
     - Reward = standard deviation of Q-values: `reward = jnp.std(q_values)`
     - This measures epistemic uncertainty about reaching state `w` from the go phase start
   - **Reward Factory**: Create `create_explore_reward_fn()` factory method
     - Takes: `reward_type`, `critic`, `gcp_actor`, `state_size`, `goal_indices`
     - Returns: reward function `(first_obs, current_state, gcp_actor_params, critic_params, rng) -> reward`
     - Allows easy extension to other reward types in future
   - Modify reward in wrapper's `step` method based on phase

### Logging and Metrics
9. **Metrics to Track**:
   - `go_phase_success_rate`: Fraction of go phases that succeed
   - `avg_go_phase_steps`: Average number of steps in successful go phases
   - Track these per environment and aggregate
   - Log via `state.metrics` or separate tracking structure
   - Use `jax.experimental.io_callback` for logging if needed

### Replay Buffer
10. **Single Replay Buffer**:
    - Both go and explore phase transitions go into same replay buffer
    - Goal-conditioned policy trains on all transitions regardless of phase
    - **Explore policy trains on explore phase transitions only**
    - Transitions must include phase information in `extras` for filtering
    - Add `phase` field to `extras["state_extras"]` in transitions

## Implementation Details

### Configuration Parameters
Add to `GoExplore` dataclass in `go_explore.py`:
- `num_gcp_steps: int = 100` - Maximum steps in go phase
- `num_ep_steps: int = 50` - Maximum steps in explore phase
- `explore_reward_type: Literal["q_epistemic", ...] = "q_epistemic"` - Type of explore reward (factory-based)
- `explore_policy_type: Literal["sac", "random"] = "sac"` - Type of explore policy (factory-based)
- `explore_noise_scale: float = 0.1` - Noise scale for explore policy (if applicable)

### Phase Transition Logic (JIT-Compatible)

```python
def should_switch_to_explore(env_state, num_gcp_steps):
    """Check if should switch from go to explore phase."""
    phase = env_state.info['phase']
    phase_step = env_state.info['phase_step']
    # Use state.metrics['success'] from underlying Brax env
    success = env_state.metrics.get('success', jnp.zeros_like(phase))
    goal_reached = success > 0.5
    max_steps_reached = phase_step >= num_gcp_steps
    in_go_phase = phase == 0
    return in_go_phase & (goal_reached | max_steps_reached)

def should_reset_after_explore(env_state, num_ep_steps):
    """Check if should reset after explore phase."""
    phase = env_state.info['phase']
    phase_step = env_state.info['phase_step']
    in_explore_phase = phase == 1
    return in_explore_phase & (phase_step >= num_ep_steps)
```

### Trajectory ID Management
- When switching go → explore: `traj_id = traj_id + 1`
- When resetting after explore: `traj_id = traj_id + 2`
- Use `jax.lax.cond` to conditionally increment

### Goal Handling
- During go phase: observation includes goal from `info['go_goal']` → `obs = [state, go_goal]`
- During explore phase: 
  - For goal-conditioned policy: observation includes state with dummy goal (all entries `-1`) → `obs = [state, dummy_goal]`
  - For explore policy: observation is just state (size `state_size`) → `obs = state`
  - This maintains compatibility - goal-conditioned policy still sees full obs_size, explore policy sees state_size
- Need to modify observation construction in wrapper based on phase

### Success Tracking
- When go phase succeeds: set `info['go_phase_success'] = True`
- Track `info['go_phase_steps']` when success occurs
- Aggregate these metrics for logging

## File Structure

### Primary Changes
1. **`goal_proposers.py`**: 
   - Add Go Explore implementation function
   - Integrate with existing goal proposer infrastructure
   - Handle phase management and transitions

2. **`jaxgcrl/envs/wrappers.py`**:
   - Add `GoExploreWrapper` class
   - Handle phase state, transitions, trajectory IDs
   - Modify rewards based on phase

### Minimal Changes to Other Files
3. **`go_explore.py`**:
   - Add config parameters for Go Explore
   - Integrate wrapper into training loop
   - Add metrics logging

## JAX JIT Considerations

### Critical Rules
1. **No Python if statements in JIT-compiled code**: Use `jax.lax.cond` instead
2. **Preserve tree structure**: All branches of conditionals must return same structure
3. **Array shapes must be static**: Phase indicators should be arrays, not Python bools
4. **Use `jnp.where` for element-wise conditionals**: When operating on batched envs
5. **Trajectory ID increments**: Must be done via JAX operations, not Python

### Example Pattern
```python
# BAD (Python if - breaks JIT):
if phase == 0:
    action = go_policy(obs)
else:
    action = explore_policy(obs)

# GOOD (JAX conditional):
action = jax.lax.cond(
    phase == 0,
    lambda: go_policy(obs),
    lambda: explore_policy(obs)
)

# For batched operations:
action = jnp.where(
    phase == 0,
    go_policy(obs),
    explore_policy(obs)
)
```

## Implementation Details - Clarified

### Explore Policy
- **Separate network** created via factory (similar to `get_algorithm()`)
- Factory function: `get_explore_policy(explore_policy_type, **kwargs)`
- Returns explore actor that takes only **state** (input size: `state_size`, not `obs_size`)
- Uses same network architecture as goal-conditioned policy (h_dim, n_hidden, etc.)
- Supports different types: "sac", "random", etc.
- Created and initialized separately from goal-conditioned policy
- **MUST be trained** on explore phase transitions with computed explore rewards
- Needs its own TrainState with optimizer

### Explore Reward
- **Default: `q_epistemic_reward`** (factory-based)
- Implementation:
  ```python
  def q_epistemic_reward(first_obs, current_state, gcp_actor, gcp_actor_params, 
                         critic, full_critic_params, state_size, goal_indices, rng):
      # Extract state from first_obs and current_state
      first_state = first_obs[:state_size]
      current_goal = current_state[:state_size][goal_indices]
      
      # Construct observation: [first_state, current_goal]
      obs = jnp.concatenate([first_state, current_goal], axis=-1)
      
      # Sample action deterministically from GCP
      action = gcp_actor.sample_actions(gcp_actor_params, obs[None, :], rng, is_deterministic=True)
      action = action[0]  # Remove batch dim
      
      # Compute Q-values using unified critic interface
      q_values = critic.apply(full_critic_params, obs[None, :], action[None, :])
      q_values = q_values[0]  # Shape: (n_critics,)
      
      # Return standard deviation across ensemble
      return jnp.std(q_values)
  ```
- Factory: `create_explore_reward_fn(reward_type, critic, gcp_actor, state_size, goal_indices)`
- Returns function that can be called during explore phase

### Observation During Explore
- **For goal-conditioned policy**: Full state with dummy goal (all entries `-1`)
  - Observation: `obs = [state, dummy_goal]` where `dummy_goal = jnp.full(goal_size, -1.0)`
  - Maintains consistent observation size for goal-conditioned policy
- **For explore policy**: Just state (size `state_size`)
  - Observation: `obs = state` (no goal component)
  - Explore policy is non-goal-conditioned, so takes state_size input

### Phase Initialization
- **Start in Go Phase** (phase = 0)

### Success Detection
- **Use `state.metrics['success']`** from underlying Brax environment
- Check: `success > 0.5` to detect goal reaching

### Explore Policy Training
- **Both policies are trained**:
  - Goal-conditioned policy: trains on all transitions (both go and explore phases)
  - Explore policy: trains on explore phase transitions only (filtered by phase)
- Both policies share the same replay buffer
- Transitions must include phase information in `extras["state_extras"]["phase"]`
- Filter transitions by phase when updating explore policy

### Reset Behavior
- **Reset to same start state** as go phase (whichever is easier to implement)
- Store `first_pipeline_state` and `first_obs` from go phase start
- Use these for reset after explore phase

## Implementation Steps

1. **Create Explore Policy Factory** (in `algorithms.py` or new file)
   - Implement `get_explore_policy(explore_policy_type, **kwargs)`
   - Create explore actor classes (non-goal-conditioned, takes `state_size` input)
   - Use same network architecture as goal-conditioned policy
   - Support "sac" type initially (can extend later)
   - Returns explore actor that implements same interface as goal-conditioned actor

2. **Create Explore Reward Factory** (in `goal_proposers.py`)
   - Implement `create_explore_reward_fn(reward_type, critic, gcp_actor, state_size, goal_indices)`
   - Implement `q_epistemic_reward` function
   - Use unified critic interface: `critic.apply(full_params, obs, action)`
   - Return reward function that takes `(first_obs, current_state, gcp_actor_params, critic_params, rng)`

3. **Create GoExploreWrapper** in `wrappers.py`
   - Implement phase tracking in `info` (phase, phase_step, go_goal, etc.)
   - Implement phase transition logic (go→explore, explore→reset)
   - Implement trajectory ID management (increment on phase switches: go→explore: +1, explore→reset: +2)
   - Implement reward modification (call explore reward function every step during explore phase)
   - Implement observation construction:
     - Go phase: `obs = [state, go_goal]` (full obs_size)
     - Explore phase for goal-conditioned policy: `obs = [state, dummy_goal]` where `dummy_goal = jnp.full(goal_size, -1.0)`
     - Explore phase for explore policy: `obs = state` (state_size only)
   - Store `first_pipeline_state` and `first_obs` in `info` for resets (already compatible)
   - Add phase information to state.info for transition tracking

4. **Update GoExplore class** in `go_explore.py`
   - Add config parameters (num_gcp_steps, num_ep_steps, explore_reward_type, explore_policy_type)
   - Create explore policy via factory (takes `state_size` input, not `obs_size`)
   - Initialize explore policy TrainState (with optimizer)
   - Create explore reward function via factory
   - Integrate wrapper into training loop
   - Pass explore policy and reward function to wrapper
   - Add metrics tracking (go_phase_success_rate, avg_go_phase_steps)
   - Add explore policy TrainState to TrainingState
   - Update explore policy during training (filter transitions by phase from extras)
   - Update goal-conditioned policy on all transitions (as before)
   - Implement metrics tracking and reset every eval

5. **Update actor_step logic** in `go_explore.py`
   - Check phase from `env_state.info['phase']`
   - Use goal-conditioned policy if phase == 0 (go) with full obs (state + goal)
   - Use explore policy if phase == 1 (explore) with state only (state_size)
   - Set `is_deterministic` appropriately (True for go, False for explore)
   - Add phase information to transition extras

6. **Update training loop** in `go_explore.py`
   - Add explore policy update step
   - Filter transitions by phase when updating explore policy (only explore phase transitions)
   - Update goal-conditioned policy on all transitions (as before)
   - Track and reset metrics every eval

7. **Testing**
   - Verify phase transitions work correctly
   - Verify trajectory IDs are separate (go: traj_id, explore: traj_id+1, reset: traj_id+2)
   - Verify metrics are tracked correctly (per-env, aggregated, reset every eval)
   - Verify explore reward computation (q_epistemic) every step
   - Verify dummy goal (-1) is used during explore phase for goal-conditioned policy
   - Verify explore policy receives state_size input (not obs_size)
   - Verify explore policy training (only on explore phase transitions)
   - Verify goal-conditioned policy training (on all transitions)
   - Verify JIT compilation works

## Edge Cases to Handle

1. **Multiple environments**: Each env can be in different phase - handle per-env
2. **Concurrent phase switches**: Some envs switching go→explore while others explore→reset
3. **Goal proposer called during explore**: Should not happen, but handle gracefully
4. **Metrics aggregation**: Track per-env and aggregate correctly
5. **Buffer sampling**: Ensure both phase types are sampled for training

## Key Implementation Summary

### Critical Points
1. **Two Policies**: 
   - Goal-conditioned policy (GCP): takes `obs_size` input, trains on all transitions
   - Explore policy: takes `state_size` input, trains on explore phase transitions only

2. **Phase Transitions**:
   - Go → Explore: when goal reached OR `num_gcp_steps` reached
   - Explore → Reset: when `num_ep_steps` reached
   - Reset → Go: with new goal from goal proposer

3. **Trajectory IDs**:
   - Go phase: `traj_id`
   - Explore phase: `traj_id + 1` (increment on go→explore switch)
   - After reset: `traj_id + 2` (increment on explore→reset)

4. **Observations**:
   - Go phase: `obs = [state, go_goal]` (obs_size)
   - Explore phase (GCP): `obs = [state, dummy_goal]` where `dummy_goal = jnp.full(goal_size, -1.0)` (obs_size)
   - Explore phase (explore policy): `obs = state` (state_size)

5. **Rewards**:
   - Go phase: environment's natural reward
   - Explore phase: `q_epistemic_reward` computed every step using unified critic interface

6. **Training**:
   - Both policies share same replay buffer
   - GCP trains on all transitions
   - Explore policy trains on explore phase transitions only (filtered by phase in extras)

7. **Metrics**:
   - Track per-environment, aggregate globally
   - Reset every eval in training loop

## Implementation Details - Final Clarifications

### Explore Policy
1. **Input Size**: Explore policy takes `state_size` as input (not `obs_size`) since it's non-goal-conditioned
2. **Architecture**: Uses same network architecture (h_dim, n_hidden, etc.) as goal-conditioned policy
3. **Training**: **Explore policy MUST be trained** on the computed explore rewards
   - Explore policy needs its own TrainState
   - Explore policy trains on transitions from explore phase
   - Goal-conditioned policy trains on all transitions (both phases)
   - Both policies share the same replay buffer, but filter by phase when training

### Dummy Goal
3. **Dummy Goal Value**: Use all entries `-1` for dummy goal during explore phase
   - This ensures it doesn't affect environment states
   - Can be any value as long as it doesn't interfere with environment

### Explore Reward
4. **Computation Frequency**: Compute explore reward **every step** during explore phase

### State Storage
5. **First Obs Storage**: Store `first_obs` in `info` (already done for compatibility with existing code)

### Metrics
6. **Metrics Aggregation**: 
   - Track metrics per-environment
   - Aggregate across environments for logging
   - Reset metrics every eval in the training loop

### Training Structure
7. **Explore Policy Training**:
   - Explore policy has its own TrainState (with optimizer)
   - **Update TrainingState dataclass** to include `explore_actor_state: TrainState`
   - Explore policy trains on explore phase transitions only
   - Need to track phase information in transitions (via `extras["state_extras"]["phase"]`)
   - Filter transitions by phase when updating explore policy
   - Goal-conditioned policy continues to train on all transitions
   - Both policies use same learning rate (or can be configurable)
