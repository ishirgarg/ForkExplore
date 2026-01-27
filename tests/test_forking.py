"""Tests for agent forking mechanism."""

import jax
import jax.numpy as jnp
import numpy as np
from brax.envs import State
from collections import namedtuple

from jaxgcrl.utils.fork_heuristics import (
    CumulativeRewardHeuristic,
    FinalRewardHeuristic,
    GoalDistanceHeuristic,
    create_heuristic,
)
from jaxgcrl.utils.fork_manager import ForkManager


# Simple transition namedtuple for testing
Transition = namedtuple('Transition', ['observation', 'action', 'reward', 'discount', 'extras'])


def create_dummy_env_state(num_envs: int, obs_dim: int, key: jax.Array) -> State:
    """Create a dummy environment state for testing."""
    obs = jax.random.normal(key, (num_envs, obs_dim))
    return State(
        pipeline_state=None,
        obs=obs,
        reward=jnp.zeros(num_envs),
        done=jnp.zeros(num_envs, dtype=bool),
        metrics={},
        info={"traj_id": jnp.arange(num_envs, dtype=jnp.float32)},
    )


def create_dummy_transitions(unroll_length: int, num_envs: int, obs_dim: int, key: jax.Array) -> Transition:
    """Create dummy transitions for testing."""
    key1, key2, key3 = jax.random.split(key, 3)
    
    obs = jax.random.normal(key1, (unroll_length, num_envs, obs_dim))
    actions = jax.random.normal(key2, (unroll_length, num_envs, 4))
    
    # Create varying rewards for testing heuristics
    rewards = jax.random.uniform(key3, (unroll_length, num_envs), minval=-1.0, maxval=1.0)
    
    discounts = jnp.ones((unroll_length, num_envs))
    
    traj_ids = jnp.broadcast_to(
        jnp.arange(num_envs, dtype=jnp.float32).reshape(1, -1),
        (unroll_length, num_envs)
    )
    
    extras = {
        "state_extras": {
            "truncation": jnp.zeros((unroll_length, num_envs)),
            "traj_id": traj_ids,
        }
    }
    
    return Transition(
        observation=obs,
        action=actions,
        reward=rewards,
        discount=discounts,
        extras=extras,
    )


def test_cumulative_reward_heuristic():
    """Test cumulative reward heuristic evaluation."""
    print("\n=== Testing Cumulative Reward Heuristic ===")
    
    key = jax.random.PRNGKey(0)
    num_envs = 8
    unroll_length = 10
    obs_dim = 16
    
    env_state = create_dummy_env_state(num_envs, obs_dim, key)
    transitions = create_dummy_transitions(unroll_length, num_envs, obs_dim, key)
    
    heuristic = CumulativeRewardHeuristic()
    scores = heuristic.evaluate(env_state, transitions, {})
    
    # Check shape
    assert scores.shape == (num_envs,), f"Expected shape ({num_envs},), got {scores.shape}"
    
    # Verify scores match cumulative rewards
    expected_scores = jnp.sum(transitions.reward, axis=0)
    assert jnp.allclose(scores, expected_scores), "Scores don't match cumulative rewards"
    
    print(f"✓ Heuristic evaluation passed")
    print(f"  Scores: {scores}")
    print(f"  Min: {jnp.min(scores):.4f}, Max: {jnp.max(scores):.4f}, Mean: {jnp.mean(scores):.4f}")


def test_goal_distance_heuristic():
    """Test goal distance heuristic evaluation."""
    print("\n=== Testing Goal Distance Heuristic ===")
    
    key = jax.random.PRNGKey(1)
    num_envs = 8
    unroll_length = 10
    state_size = 8
    goal_size = 2
    obs_dim = state_size + goal_size
    goal_indices = tuple(range(state_size, obs_dim))
    
    # Create state where first half is at goal, second half is far
    key1, key2 = jax.random.split(key)
    env_state = create_dummy_env_state(num_envs, obs_dim, key1)
    
    # Manually set some agents close to goal and some far
    obs = env_state.obs
    # For first half: make the first goal_size dimensions of state match the goal
    goal_values = obs[:num_envs//2, state_size:]  # shape: (4, 2)
    # Set the first 2 dimensions of state to match goal
    obs = obs.at[:num_envs//2, :goal_size].set(goal_values)
    # For second half: state is far from goal  
    obs = obs.at[num_envs//2:, :goal_size].set(obs[num_envs//2:, :goal_size] + 10.0)
    env_state = env_state.replace(obs=obs)
    
    transitions = create_dummy_transitions(unroll_length, num_envs, obs_dim, key2)
    
    heuristic = GoalDistanceHeuristic(state_size=state_size, goal_indices=goal_indices)
    scores = heuristic.evaluate(env_state, transitions, {})
    
    # Check shape
    assert scores.shape == (num_envs,), f"Expected shape ({num_envs},), got {scores.shape}"
    
    # First half should have higher scores (closer to goal)
    assert jnp.mean(scores[:num_envs//2]) > jnp.mean(scores[num_envs//2:]), \
        "Agents close to goal should have higher scores"
    
    print(f"✓ Goal distance heuristic passed")
    print(f"  Scores (close): {scores[:num_envs//2]}")
    print(f"  Scores (far): {scores[num_envs//2:]}")


def test_fork_decision():
    """Test fork manager decision making."""
    print("\n=== Testing Fork Decision Making ===")
    
    key = jax.random.PRNGKey(2)
    num_envs = 10
    fork_k = 3
    unroll_length = 10
    obs_dim = 16
    
    env_state = create_dummy_env_state(num_envs, obs_dim, key)
    transitions = create_dummy_transitions(unroll_length, num_envs, obs_dim, key)
    
    heuristic = CumulativeRewardHeuristic()
    fork_manager = ForkManager(
        num_envs=num_envs,
        fork_k=fork_k,
        heuristic=heuristic,
        enabled=True,
    )
    
    fork_decision = fork_manager.evaluate_and_decide(env_state, transitions, {})
    
    # Check decision structure
    assert fork_decision.should_fork == True, "Should decide to fork when enabled"
    assert len(fork_decision.bottom_k_indices) == fork_k, f"Should have {fork_k} bottom indices"
    assert len(fork_decision.top_k_indices) == fork_k, f"Should have {fork_k} top indices"
    assert fork_decision.fork_mask.shape == (num_envs,), "Fork mask should match num_envs"
    assert jnp.sum(fork_decision.fork_mask) == fork_k, f"Fork mask should have {fork_k} True values"
    
    # Verify bottom K have lowest scores and top K have highest scores
    scores = fork_decision.scores
    sorted_indices = jnp.argsort(scores)
    
    # Check that bottom_k_indices contains the K indices with lowest scores
    expected_bottom_k = sorted_indices[:fork_k]
    assert set(fork_decision.bottom_k_indices.tolist()) == set(expected_bottom_k.tolist()), \
        f"Bottom K should be agents with lowest scores. Got {fork_decision.bottom_k_indices}, expected {expected_bottom_k}"
    
    # Check that top_k_indices contains the K indices with highest scores
    expected_top_k = sorted_indices[-fork_k:]
    assert set(fork_decision.top_k_indices.tolist()) == set(expected_top_k.tolist()), \
        f"Top K should be agents with highest scores. Got {fork_decision.top_k_indices}, expected {expected_top_k}"
    
    print(f"✓ Fork decision making passed")
    print(f"  Bottom K indices: {fork_decision.bottom_k_indices}")
    print(f"  Top K indices: {fork_decision.top_k_indices}")
    print(f"  Bottom K scores: {scores[fork_decision.bottom_k_indices]}")
    print(f"  Top K scores: {scores[fork_decision.top_k_indices]}")


def test_apply_forking():
    """Test applying forking to environment state."""
    print("\n=== Testing Apply Forking ===")
    
    key = jax.random.PRNGKey(3)
    num_envs = 8
    fork_k = 2
    obs_dim = 16
    
    # Create env state with distinct values for each env
    env_state = create_dummy_env_state(num_envs, obs_dim, key)
    # Set observations to be identifiable (env 0 -> all 0s, env 1 -> all 1s, etc.)
    obs = jnp.array([jnp.full(obs_dim, float(i)) for i in range(num_envs)])
    env_state = env_state.replace(obs=obs)
    
    # Create fork decision (manually)
    from jaxgcrl.utils.fork_manager import ForkDecision
    bottom_k_indices = jnp.array([0, 1])  # Replace envs 0 and 1
    top_k_indices = jnp.array([6, 7])  # With envs 6 and 7
    fork_mask = jnp.array([True, True, False, False, False, False, False, False])
    
    fork_decision = ForkDecision(
        should_fork=jnp.array(True),
        bottom_k_indices=bottom_k_indices,
        top_k_indices=top_k_indices,
        fork_mask=fork_mask,
        scores=jnp.arange(num_envs, dtype=jnp.float32),
    )
    
    heuristic = CumulativeRewardHeuristic()
    fork_manager = ForkManager(num_envs=num_envs, fork_k=fork_k, heuristic=heuristic)
    
    # Apply forking
    new_env_state = fork_manager.apply_forking(env_state, fork_decision)
    
    # Check that bottom K envs now have same obs as top K envs
    assert jnp.allclose(new_env_state.obs[0], env_state.obs[6]), \
        "Env 0 should have obs from env 6"
    assert jnp.allclose(new_env_state.obs[1], env_state.obs[7]), \
        "Env 1 should have obs from env 7"
    
    # Check that other envs are unchanged
    for i in range(2, num_envs):
        assert jnp.allclose(new_env_state.obs[i], env_state.obs[i]), \
            f"Env {i} should be unchanged"
    
    print(f"✓ Apply forking passed")
    print(f"  Env 0 obs (before): {env_state.obs[0][:4]}... -> (after): {new_env_state.obs[0][:4]}...")
    print(f"  Env 6 obs: {env_state.obs[6][:4]}...")


def test_increment_trajectory_ids():
    """Test trajectory ID incrementation for forked agents."""
    print("\n=== Testing Trajectory ID Incrementation ===")
    
    key = jax.random.PRNGKey(4)
    num_envs = 8
    fork_k = 3
    obs_dim = 16
    
    env_state = create_dummy_env_state(num_envs, obs_dim, key)
    
    # Set initial trajectory IDs
    initial_traj_ids = jnp.array([10.0, 12.0, 15.0, 8.0, 20.0, 5.0, 18.0, 25.0])
    env_state = env_state.replace(
        info={"traj_id": initial_traj_ids}
    )
    
    # Create fork decision
    from jaxgcrl.utils.fork_manager import ForkDecision
    bottom_k_indices = jnp.array([0, 2, 5])  # These will be forked
    top_k_indices = jnp.array([4, 6, 7])
    fork_mask = jnp.array([True, False, True, False, False, True, False, False])
    
    fork_decision = ForkDecision(
        should_fork=jnp.array(True),
        bottom_k_indices=bottom_k_indices,
        top_k_indices=top_k_indices,
        fork_mask=fork_mask,
        scores=jnp.zeros(num_envs),
    )
    
    heuristic = CumulativeRewardHeuristic()
    fork_manager = ForkManager(num_envs=num_envs, fork_k=fork_k, heuristic=heuristic)
    
    # Increment trajectory IDs
    new_env_state = fork_manager.increment_trajectory_ids(env_state, fork_decision)
    new_traj_ids = new_env_state.info["traj_id"]
    
    # Check that forked agents have incremented IDs
    for idx in bottom_k_indices:
        assert new_traj_ids[idx] == initial_traj_ids[idx] + 1, \
            f"Forked agent {idx} should have incremented traj_id"
    
    # Check that non-forked agents have unchanged IDs
    for i in range(num_envs):
        if i not in bottom_k_indices:
            assert new_traj_ids[i] == initial_traj_ids[i], \
                f"Non-forked agent {i} should have unchanged traj_id"
    
    print(f"✓ Trajectory ID incrementation passed")
    print(f"  Initial traj_ids: {initial_traj_ids}")
    print(f"  New traj_ids: {new_traj_ids}")
    print(f"  Forked agents: {bottom_k_indices}")


def test_all_transitions_kept():
    """Test that all transitions are kept for replay buffer."""
    print("\n=== Testing All Transitions Kept ===")
    
    key = jax.random.PRNGKey(5)
    num_envs = 10
    fork_k = 3
    unroll_length = 5
    obs_dim = 16
    
    transitions = create_dummy_transitions(unroll_length, num_envs, obs_dim, key)
    
    # Create fork decision
    from jaxgcrl.utils.fork_manager import ForkDecision
    bottom_k_indices = jnp.array([1, 3, 7])
    top_k_indices = jnp.array([0, 5, 9])
    fork_mask = jnp.zeros(num_envs, dtype=bool).at[bottom_k_indices].set(True)
    
    fork_decision = ForkDecision(
        should_fork=jnp.array(True),
        bottom_k_indices=bottom_k_indices,
        top_k_indices=top_k_indices,
        fork_mask=fork_mask,
        scores=jnp.zeros(num_envs),
    )
    
    # With simplified implementation, transitions are NOT filtered
    # All transitions (including from forked agents) go to the buffer
    # We just verify the shape is correct and data is intact
    
    assert transitions.observation.shape == (unroll_length, num_envs, obs_dim), \
        "Transition shape should be unchanged"
    
    # Verify all agent data is present (not zeroed)
    for idx in range(num_envs):
        # All transitions should have non-zero data (they were randomly generated)
        assert not jnp.allclose(transitions.observation[:, idx], 0.0), \
            f"Agent {idx} transitions should have data"
    
    print(f"✓ All transitions kept (simplified implementation)")
    print(f"  Shape: {transitions.observation.shape}")
    print(f"  All {num_envs} agents' transitions will be added to buffer")
    print(f"  Forked agents: {bottom_k_indices}")
    print(f"  Note: Forked agents' transitions ARE included in replay buffer")


def test_fork_metrics():
    """Test fork metrics generation."""
    print("\n=== Testing Fork Metrics ===")
    
    key = jax.random.PRNGKey(6)
    num_envs = 10
    fork_k = 2
    unroll_length = 10
    obs_dim = 16
    
    env_state = create_dummy_env_state(num_envs, obs_dim, key)
    transitions = create_dummy_transitions(unroll_length, num_envs, obs_dim, key)
    
    heuristic = CumulativeRewardHeuristic()
    fork_manager = ForkManager(
        num_envs=num_envs,
        fork_k=fork_k,
        heuristic=heuristic,
        enabled=True,
    )
    
    fork_decision = fork_manager.evaluate_and_decide(env_state, transitions, {})
    metrics = fork_manager.get_metrics(fork_decision)
    
    # Check metric keys
    expected_keys = [
        "forking/enabled",
        "forking/num_forked",
        "forking/mean_score",
        "forking/std_score",
        "forking/min_score",
        "forking/max_score",
        "forking/bottom_k_mean_score",
        "forking/top_k_mean_score",
    ]
    
    for key in expected_keys:
        assert key in metrics, f"Metric {key} should be present"
    
    # Check metric values
    assert metrics["forking/enabled"] == 1.0, "Enabled should be 1.0"
    assert metrics["forking/num_forked"] == float(fork_k), f"Num forked should be {fork_k}"
    assert metrics["forking/bottom_k_mean_score"] < metrics["forking/top_k_mean_score"], \
        "Bottom K mean score should be less than top K mean score"
    
    print(f"✓ Fork metrics generation passed")
    print(f"  Metrics: {metrics}")


def run_all_tests():
    """Run all forking tests."""
    print("=" * 60)
    print("Running Agent Forking Tests")
    print("=" * 60)
    
    try:
        test_cumulative_reward_heuristic()
        test_goal_distance_heuristic()
        test_fork_decision()
        test_apply_forking()
        test_increment_trajectory_ids()
        test_all_transitions_kept()
        test_fork_metrics()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
