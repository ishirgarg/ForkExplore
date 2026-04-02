import functools
import pickle
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from etils import epath


def load_params(path: str):
    with epath.Path(path).open("rb") as fin:
        buf = fin.read()
    return pickle.loads(buf)


def save_params(path: str, params: Any):
    """Saves parameters in flax format."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


@functools.partial(jax.jit, static_argnames=("buffer_config"))
def flatten_batch(buffer_config, transition, sample_key):
    gamma, state_size, goal_indices = buffer_config

    # Because it's vmaped transition.obs.shape is of shape (episode_len, obs_dim)
    seq_len = transition.observation.shape[0]
    arrangement = jnp.arange(seq_len)
    is_future_mask = jnp.array(
        arrangement[:, None] < arrangement[None], dtype=jnp.float32
    )  # upper triangular matrix of shape seq_len, seq_len where all non-zero entries are 1
    discount = gamma ** jnp.array(arrangement[None] - arrangement[:, None], dtype=jnp.float32)
    probs = is_future_mask * discount

    # probs is an upper triangular matrix of shape seq_len, seq_len of the form:
    #    [[0.        , 0.99      , 0.98010004, 0.970299  , 0.960596 ],
    #    [0.        , 0.        , 0.99      , 0.98010004, 0.970299  ],
    #    [0.        , 0.        , 0.        , 0.99      , 0.98010004],
    #    [0.        , 0.        , 0.        , 0.        , 0.99      ],
    #    [0.        , 0.        , 0.        , 0.        , 0.        ]]
    # assuming seq_len = 5
    # the same result can be obtained using probs = is_future_mask * (gamma ** jnp.cumsum(is_future_mask, axis=-1))

    single_trajectories = jnp.concatenate(
        [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
        axis=0,
    )
    # array of seq_len x seq_len where a row is an array of traj_ids that correspond to the episode index from which that time-step was collected
    # timesteps collected from the same episode will have the same traj_id. All rows of the single_trajectories are same.

    probs = probs * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
    # ith row of probs will be non zero only for time indices that
    # 1) are greater than i
    # 2) have the same traj_id as the ith time index

    goal_index = jax.random.categorical(sample_key, jnp.log(probs))
    future_state = jnp.take(
        transition.observation, goal_index[:-1], axis=0
    )  # the last goal_index cannot be considered as there is no future.
    future_action = jnp.take(transition.action, goal_index[:-1], axis=0)
    goal = future_state[:, goal_indices]
    future_state = future_state[:, :state_size]
    state = transition.observation[:-1, :state_size]  # all states are considered
    new_obs = jnp.concatenate([state, goal], axis=1)

    extras = {
        "policy_extras": {},
        "state_extras": {
            "truncation": jnp.squeeze(transition.extras["state_extras"]["truncation"][:-1]),
            "traj_id": jnp.squeeze(transition.extras["state_extras"]["traj_id"][:-1]),
        },
        "state": state,
        "future_state": future_state,
        "future_action": future_action,
    }

    return transition._replace(
        observation=jnp.squeeze(new_obs),  # this has shape (num_envs, episode_length-1, obs_size)
        action=jnp.squeeze(transition.action[:-1]),
        reward=jnp.squeeze(transition.reward[:-1]),
        discount=jnp.squeeze(transition.discount[:-1]),
        extras=extras,
    )


def sample_trajectories_from_buffer(
    replay_buffer,
    buffer_state,
    state_size: int,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
) -> Tuple[Any, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample trajectories from the replay buffer and extract positions.
    
    Args:
        replay_buffer: The replay buffer instance
        buffer_state: Current buffer state (will be modified by sampling)
        state_size: Size of state dimension
        goal_indices: Indices for x, y positions (typically [0, 1])
        rng_key: Random key for sampling
        
    Returns:
        Tuple of (buffer_state, all_positions, final_positions, goal_positions) where:
        - buffer_state: Updated buffer state after sampling
        - all_positions: (N, 2) array of [x, y] positions from all states
        - final_positions: (M, 2) array of [x, y] positions from final states
        - goal_positions: (N, 2) array of [x, y] goal positions from all observations
    """
    # Check buffer size
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return buffer_state, np.array([]).reshape(0, 2), np.array([]).reshape(0, 2), np.array([]).reshape(0, 2)
    
    # Sample from buffer - use whatever it gives us
    current_buffer_state, transitions = replay_buffer.sample(buffer_state)
    
    # transitions.observation shape: (num_envs, episode_length, obs_size)
    # transitions.extras["state_extras"]["traj_id"] shape: (num_envs, episode_length)
    # transitions.extras["state_extras"]["truncation"] shape: (num_envs, episode_length)
    
    # Flatten to (num_envs * episode_length, obs_size)
    obs_flat = jnp.reshape(transitions.observation, (-1, transitions.observation.shape[-1]))
    traj_id_flat = jnp.reshape(transitions.extras["state_extras"]["traj_id"], (-1,))
    truncation_flat = jnp.reshape(transitions.extras["state_extras"]["truncation"], (-1,))
    
    # Extract x, y positions from observations (first state_size elements contain state)
    positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, 2)
    
    # Extract goal positions from latter part of observation
    # Goals are at state_size:state_size+goal_size (observation = [state, goal])
    goal_size = len(goal_indices)
    goal_positions = obs_flat[:, -goal_size:]  # (N, goal_size)
    
    # Convert to numpy for easier processing
    positions_np = np.array(positions)
    goal_positions_np = np.array(goal_positions)
    traj_ids_np = np.array(traj_id_flat)
    truncations_np = np.array(truncation_flat)
    
    # Get all positions (for all states plot)
    all_positions = positions_np
    
    # Get final positions (where truncation is True, or last state of each trajectory)
    final_positions = []
    unique_traj_ids = np.unique(traj_ids_np)
    
    for traj_id in unique_traj_ids:
        traj_mask = traj_ids_np == traj_id
        traj_positions = positions_np[traj_mask]
        traj_truncations = truncations_np[traj_mask]
        
        # Find final state: either where truncation is True, or last state
        final_idx = np.where(traj_truncations)[0]
        if len(final_idx) > 0:
            # Use first truncation point as final state
            final_positions.append(traj_positions[final_idx[0]])
        else:
            # Use last state if no truncation found
            if len(traj_positions) > 0:
                final_positions.append(traj_positions[-1])
    
    if len(final_positions) == 0:
        final_positions = np.array([]).reshape(0, 2)
    else:
        final_positions = np.array(final_positions)
    
    # Randomly sample 512 points from all states (keep all final states)
    if len(all_positions) > 512:
        rng = np.random.RandomState(seed=42)  # Deterministic sampling
        indices = rng.choice(len(all_positions), 512, replace=False)
        all_positions = all_positions[indices]
        goal_positions_np = goal_positions_np[indices]
    
    return current_buffer_state, all_positions, final_positions, goal_positions_np



def create_dummy_transition_for_buffer(
    unroll_length: int,
    num_envs: int,
    obs_size: int,
    action_size: int,
    agent_type: str = "crl",
    include_phase: bool = False,
) -> Any:
    """
    Create a dummy transition object for replay buffer insertion.
    Shape is (unroll_length, num_envs, ...) to match what's normally inserted.
    
    Args:
        unroll_length: Length of unroll (first dimension)
        num_envs: Number of parallel environments (second dimension)
        obs_size: Size of observation dimension
        action_size: Size of action dimension
        agent_type: Type of agent ("crl" or "sac") - SAC needs next_observation
        include_phase: If True, include "phase" in state_extras (for Go Explore)
        
    Returns:
        A Transition object with zero-filled arrays of shape (unroll_length, num_envs, ...).
    """
    from .types import Transition
    
    dummy_obs = jnp.zeros((unroll_length, num_envs, obs_size))
    dummy_action = jnp.zeros((unroll_length, num_envs, action_size))
    dummy_reward = jnp.zeros((unroll_length, num_envs))
    dummy_discount = jnp.zeros((unroll_length, num_envs))
    dummy_next_obs = jnp.zeros((unroll_length, num_envs, obs_size)) if agent_type == "sac" else None
    state_extras = {
            "traj_id": jnp.zeros((unroll_length, num_envs), dtype=jnp.float32),
            "truncation": jnp.zeros((unroll_length, num_envs), dtype=jnp.float32),
        }
    if include_phase:
        state_extras["phase"] = jnp.zeros((unroll_length, num_envs), dtype=jnp.int32)
    dummy_extras = {"state_extras": state_extras}
    return Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=dummy_reward,
        discount=dummy_discount,
        next_observation=dummy_next_obs,
        extras=dummy_extras
    )


def create_dummy_transition_for_goal_proposer(
    num_envs: int,
    episode_length: int,
    obs_size: int,
    action_size: int,
    agent_type: str = "crl",
    include_phase: bool = False,
) -> Any:
    """
    Create a dummy transition object with shape (num_envs, episode_length, ...) for goal proposer state.
    This matches the shape returned by the replay buffer's sample method.
    
    Args:
        num_envs: Number of parallel environments
        episode_length: Length of episode
        obs_size: Size of observation dimension
        action_size: Size of action dimension
        agent_type: Type of agent ("crl" or "sac") - SAC needs next_observation
        include_phase: If True, include "phase" in state_extras (for Go Explore)
        
    Returns:
        A Transition object with zero-filled arrays of shape (num_envs, episode_length, ...).
    """
    from .types import Transition
    
    dummy_obs = jnp.zeros((num_envs, episode_length, obs_size))
    dummy_action = jnp.zeros((num_envs, episode_length, action_size))
    dummy_reward = jnp.zeros((num_envs, episode_length))
    dummy_discount = jnp.zeros((num_envs, episode_length))
    dummy_next_obs = jnp.zeros((num_envs, episode_length, obs_size)) if agent_type == "sac" else None
    state_extras = {
            "traj_id": jnp.zeros((num_envs, episode_length), dtype=jnp.float32),
            "truncation": jnp.zeros((num_envs, episode_length), dtype=jnp.float32),
        }
    if include_phase:
        state_extras["phase"] = jnp.zeros((num_envs, episode_length), dtype=jnp.int32)
    dummy_extras = {"state_extras": state_extras}
    return Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=dummy_reward,
        discount=dummy_discount,
        next_observation=dummy_next_obs,
        extras=dummy_extras
    )


def create_single_dummy_transition(
    obs_size: int,
    action_size: int,
    agent_type: str = "crl",
    include_phase: bool = False,
) -> Any:
    """
    Create a single dummy transition object (not batched) for replay buffer initialization.
    
    Args:
        obs_size: Size of observation dimension
        action_size: Size of action dimension
        agent_type: Type of agent ("crl" or "sac") - SAC needs next_observation
        include_phase: If True, include "phase" in state_extras (for Go Explore)
        
    Returns:
        A Transition object with zero-filled arrays of shape (obs_size,) and (action_size,).
    """
    from .types import Transition
    
    dummy_obs = jnp.zeros((obs_size,))
    dummy_action = jnp.zeros((action_size,))
    dummy_reward = 0.0
    dummy_discount = 0.0
    dummy_next_obs = jnp.zeros((obs_size,)) if agent_type == "sac" else None
    state_extras = {
            "truncation": 0.0,
            "traj_id": 0.0,
        }
    if include_phase:
        state_extras["phase"] = jnp.zeros((), dtype=jnp.int32)
    dummy_extras = {"state_extras": state_extras}
    return Transition(
        observation=dummy_obs,
        action=dummy_action,
        reward=dummy_reward,
        discount=dummy_discount,
        next_observation=dummy_next_obs,
        extras=dummy_extras
    )


def sample_trajectory_sequences(
    replay_buffer,
    buffer_state,
    state_size: int,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
    num_trajectories: int = 4,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Sample full trajectory sequences from the replay buffer.
    
    Args:
        replay_buffer: The replay buffer instance
        buffer_state: Current buffer state (will be modified by sampling)
        state_size: Size of state dimension
        goal_indices: Indices for x, y positions (typically [0, 1])
        rng_key: Random key for sampling
        num_trajectories: Number of trajectories to extract (default: 4)
        
    Returns:
        Tuple of (buffer_state, trajectory_states, trajectory_goals) where:
        - buffer_state: Updated buffer state after sampling
        - trajectory_states: (num_trajectories, 8, 2) array of [x, y] positions
          [start, 6 intermediate states, final]
        - trajectory_goals: (num_trajectories, 2) array of [x, y] goal positions
    """
    # Check buffer size
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return buffer_state, np.array([]).reshape(0, 8, 2), np.array([]).reshape(0, 2)
    
    # Sample from buffer
    current_buffer_state, transitions = replay_buffer.sample(buffer_state)
    
    # Flatten to (num_envs * episode_length, obs_size)
    obs_flat = jnp.reshape(transitions.observation, (-1, transitions.observation.shape[-1]))
    traj_id_flat = jnp.reshape(transitions.extras["state_extras"]["traj_id"], (-1,))
    truncation_flat = jnp.reshape(transitions.extras["state_extras"]["truncation"], (-1,))
    
    # Extract positions and goals
    positions = obs_flat[:, :state_size][:, list(goal_indices)]  # (N, 2)
    goal_size = len(goal_indices)
    goal_positions = obs_flat[:, -goal_size:]  # (N, goal_size)
    
    # Convert to numpy
    positions_np = np.array(positions)
    goal_positions_np = np.array(goal_positions)
    traj_ids_np = np.array(traj_id_flat)
    truncations_np = np.array(truncation_flat)
    
    # Group by trajectory ID and extract sequences
    unique_traj_ids = np.unique(traj_ids_np)
    num_trajectories = min(num_trajectories, len(unique_traj_ids))
    
    trajectory_states = []
    trajectory_goals = []
    
    for i in range(num_trajectories):
        traj_id = unique_traj_ids[i]
        traj_mask = traj_ids_np == traj_id
        traj_positions = positions_np[traj_mask]
        traj_goals = goal_positions_np[traj_mask]
        traj_truncations = truncations_np[traj_mask]
        
        if len(traj_positions) == 0:
            continue
        
        # Get goal (should be constant within trajectory, take first)
        goal = traj_goals[0]
        
        # Find final state index
        final_idx = np.where(traj_truncations)[0]
        if len(final_idx) > 0:
            final_idx = final_idx[0]
        else:
            final_idx = len(traj_positions) - 1
        
        # Extract states: start, 6 intermediate (evenly spaced), final
        traj_length = final_idx + 1
        if traj_length < 8:
            # If trajectory is shorter than 8, pad with final state
            states = np.zeros((8, 2))
            states[:traj_length] = traj_positions[:traj_length]
            states[traj_length:] = traj_positions[final_idx]
        else:
            # Evenly sample 6 intermediate states between start and final
            indices = np.linspace(0, final_idx, 8, dtype=int)
            states = traj_positions[indices]
        
        trajectory_states.append(states)
        trajectory_goals.append(goal)
    
    if len(trajectory_states) == 0:
        return current_buffer_state, np.array([]).reshape(0, 8, 2), np.array([]).reshape(0, 2)
    
    return current_buffer_state, np.array(trajectory_states), np.array(trajectory_goals)


def geometric_sample_one_triple(
    gamma, state_size, goal_idx_array, all_obs, all_acts, all_traj_ids, env_idx, t, key
):
    """
    Sample a single (s_t, a_t, g_{t+k}) triple from a pre-sampled buffer.
 
    Given a specific (env_idx, t), samples a future timestep t' > t with
    probability proportional to γ^(t'-t) within the same trajectory, then
    returns obs = [state_t, goal_{t'}] and action_t.
 
    Args:
        gamma:           Discount factor.
        state_size:      Elements in the state portion of obs.
        goal_idx_array:  JAX array of indices selecting goal dims from state.
        all_obs:         (num_envs, episode_length, obs_size) full buffer obs.
        all_acts:        (num_envs, episode_length, action_size) full buffer acts.
        all_traj_ids:    (num_envs, episode_length) trajectory IDs.
        env_idx:         Scalar int — which env to draw from.
        t:               Scalar int — which anchor timestep.
        key:             JAX random key.
 
    Returns:
        obs:    (obs_size,)    [state_t, geom-sampled goal]
        action: (action_size,)
    """
    ep_len = all_obs.shape[1]
    arrangement = jnp.arange(ep_len)
 
    traj_obs  = all_obs[env_idx]       # (ep_len, obs_size)
    traj_acts = all_acts[env_idx]      # (ep_len, action_size)
    traj_ids  = all_traj_ids[env_idx]  # (ep_len,)
 
    # Geometric weights: only future steps in the same trajectory
    is_future  = (arrangement > t).astype(jnp.float32)
    discount   = gamma ** (arrangement - t).astype(jnp.float32)
    same_traj  = jnp.equal(traj_ids, traj_ids[t]).astype(jnp.float32)
    probs      = is_future * discount * same_traj
    # Fallback: if no valid future (e.g. last step), self-sample
    probs      = probs + (arrangement == t).astype(jnp.float32) * 1e-5
 
    future_t  = jax.random.categorical(key, jnp.log(probs))  # scalar
    state     = traj_obs[t, :state_size]                      # (state_size,)
    goal      = traj_obs[future_t, :state_size][goal_idx_array]  # (goal_dim,)
    obs       = jnp.concatenate([state, goal])                # (obs_size,)
    action    = traj_acts[t]                                  # (action_size,)
 
    return obs, action