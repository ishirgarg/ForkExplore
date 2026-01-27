"""Heuristics for evaluating agent performance in forking mechanism.

This module provides a flexible interface for implementing different heuristics
to evaluate agent performance and determine which agents should be forked.
"""

from abc import ABC, abstractmethod
from typing import Literal, Dict, Any, Union, Callable

import jax
import jax.numpy as jnp
import flax.linen as nn
from brax import envs
from brax.v1 import envs as envs_v1

# Import energy function from CRL losses
from jaxgcrl.agents.crl.losses import energy_fn


class ForkHeuristic(ABC):
    """Base class for fork heuristics.
    
    A heuristic evaluates the current state/performance of each agent
    and returns a score. Higher scores indicate better performance.
    """
    
    @abstractmethod
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate agent performance.
        
        Args:
            env_state: Current environment state for each agent (shape: num_envs)
            transitions: Recent transitions collected during unroll
            env_info: Additional environment information
            
        Returns:
            scores: Performance score for each agent (shape: num_envs)
                   Higher scores indicate better performance
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the heuristic."""
        pass


class CumulativeRewardHeuristic(ForkHeuristic):
    """Heuristic based on cumulative reward over the unroll period."""
    
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate based on cumulative reward.
        
        Args:
            env_state: Current environment state
            transitions: Transitions with shape (unroll_length, num_envs, ...)
            env_info: Additional environment information
            
        Returns:
            cumulative_rewards: Sum of rewards over unroll period (num_envs,)
        """
        # transitions.reward has shape (unroll_length, num_envs)
        # Sum over the unroll dimension to get cumulative reward per env
        cumulative_rewards = jnp.sum(transitions.reward, axis=0)
        return cumulative_rewards
    
    @property
    def name(self) -> str:
        return "cumulative_reward"


class FinalRewardHeuristic(ForkHeuristic):
    """Heuristic based on the most recent reward."""
    
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate based on final reward in the unroll.
        
        Args:
            env_state: Current environment state
            transitions: Transitions with shape (unroll_length, num_envs, ...)
            env_info: Additional environment information
            
        Returns:
            final_rewards: Most recent reward (num_envs,)
        """
        # Get the last reward in the unroll
        final_rewards = transitions.reward[-1]
        return final_rewards
    
    @property
    def name(self) -> str:
        return "final_reward"


class GoalDistanceHeuristic(ForkHeuristic):
    """Heuristic based on distance to goal (for goal-conditioned tasks).
    
    Lower distance is better, so we negate the distance to get higher = better.
    """
    
    def __init__(self, state_size: int, goal_indices: tuple):
        """Initialize goal distance heuristic.
        
        Args:
            state_size: Size of the state portion of observation
            goal_indices: Indices corresponding to goal in observation
        """
        self.state_size = state_size
        self.goal_indices = jnp.array(goal_indices)
    
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate based on negative distance to goal.
        
        Args:
            env_state: Current environment state
            transitions: Transitions with shape (unroll_length, num_envs, ...)
            env_info: Additional environment information
            
        Returns:
            negative_distances: Negative L2 distance to goal (num_envs,)
        """
        # Get current observation for each env
        current_obs = env_state.obs  # shape: (num_envs, obs_dim)
        
        # Extract state and goal
        current_state = current_obs[:, :self.state_size]  # (num_envs, state_size)
        goal = current_obs[:, self.goal_indices]  # (num_envs, goal_size)
        
        # For distance computation, assume the first goal_size dimensions of state
        # correspond to the achieved goal that we compare with the desired goal
        achieved_goal = current_state[:, :goal.shape[-1]]  # (num_envs, goal_size)
        
        # Compute L2 distance between achieved goal and desired goal
        distances = jnp.linalg.norm(achieved_goal - goal, axis=-1)  # (num_envs,)
        
        # Return negative distance (so closer = higher score)
        return -distances
    
    @property
    def name(self) -> str:
        return "goal_distance"


class MinCriticDistanceHeuristic(ForkHeuristic):
    """Heuristic based on minimum critic value to any environment goal.
    
    This heuristic uses the learned critic to evaluate how "close" an agent
    is to achieving any of the environment's possible goals. For each agent's
    final state, it computes the critic value f(s, a, g) for all possible goals
    and takes the maximum (since higher critic value = closer).
    
    The "distance" here is measured by the critic using the repo's energy_fn,
    not L2 norm.
    """
    
    def __init__(
        self,
        state_size: int,
        goal_indices: tuple,
        possible_goals: jnp.ndarray,
        actor_apply_fn: Callable,
        actor_params: Any,
        sa_encoder_apply_fn: Callable,
        g_encoder_apply_fn: Callable,
        critic_params: Any,
        energy_fn: str = "norm",
    ):
        """Initialize min critic distance heuristic.
        
        Args:
            state_size: Size of the state portion of observation
            goal_indices: Indices corresponding to goal in observation
            possible_goals: Array of possible goals, shape (num_goals, goal_size)
            actor_apply_fn: Function to apply actor network
            actor_params: Actor network parameters
            sa_encoder_apply_fn: Function to apply state-action encoder
            g_encoder_apply_fn: Function to apply goal encoder
            critic_params: Critic network parameters (contains sa_encoder and g_encoder)
            energy_fn: Energy function type ("norm", "l2", "dot", "cosine")
        """
        self.state_size = state_size
        self.goal_indices = jnp.array(goal_indices)
        self.possible_goals = possible_goals
        self.actor_apply_fn = actor_apply_fn
        self.actor_params = actor_params
        self.sa_encoder_apply_fn = sa_encoder_apply_fn
        self.g_encoder_apply_fn = g_encoder_apply_fn
        self.critic_params = critic_params
        self.energy_fn = energy_fn
        self.num_goals = possible_goals.shape[0]
    
    
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate based on minimum critic distance to any goal.
        
        For each agent:
        1. Get the final state from the trajectory
        2. For each possible goal:
           a. Sample an action from the policy conditioned on (final_state, goal)
           b. Compute critic value f(final_state, action, goal)
        3. Take the minimum (best) critic value across all goals
        
        Args:
            env_state: Current environment state
            transitions: Transitions with shape (unroll_length, num_envs, ...)
            env_info: Additional environment information (should contain actor_params and critic_params)
            
        Returns:
            scores: Minimum critic value to any goal (num_envs,)
                   Higher scores indicate closer to at least one goal
        """
        # Use current params from env_info if available, otherwise use stored params
        actor_params = env_info.get("actor_params", self.actor_params)
        critic_params = env_info.get("critic_params", self.critic_params)
        
        # Get final observation for each env
        # transitions.observation has shape (unroll_length, num_envs, obs_dim)
        final_obs = transitions.observation[-1]  # (num_envs, obs_dim)
        
        # Extract final state (not the goal part)
        final_state = final_obs[:, :self.state_size]  # (num_envs, state_size)
        
        num_envs = final_state.shape[0]
        
        # For each env, compute critic values for all possible goals
        # Vectorize efficiently over both envs and goals
        
        def compute_critic_for_env(state: jnp.ndarray) -> jnp.ndarray:
            """Compute critic values for one state across all goals.
            
            Args:
                state: State, shape (state_size,)
                
            Returns:
                max_critic: Maximum critic value across all goals (scalar)
            """
            # Create observations for all goals: (num_goals, obs_size)
            # Broadcast state to all goals
            states_broadcast = jnp.tile(state[None, :], (self.num_goals, 1))  # (num_goals, state_size)
            observations = jnp.concatenate([states_broadcast, self.possible_goals], axis=-1)  # (num_goals, obs_size)
            
            # Sample actions from policy for all (state, goal) pairs in one batch
            means, log_stds = self.actor_apply_fn(actor_params, observations)  # (num_goals, action_size)
            # Use mean actions (deterministic) for evaluation
            actions = nn.tanh(means)  # (num_goals, action_size)
            
            # Compute critic values: f(s, a, g) for all goals
            # Encode all state-action pairs
            sa_pairs = jnp.concatenate([states_broadcast, actions], axis=-1)  # (num_goals, state_size + action_size)
            phi_sa = self.sa_encoder_apply_fn(
                critic_params["sa_encoder"],
                sa_pairs
            )  # (num_goals, repr_dim)
            
            # Encode all goals
            psi_g = self.g_encoder_apply_fn(
                critic_params["g_encoder"],
                self.possible_goals
            )  # (num_goals, repr_dim)
            
            # Compute energy (critic values) using the repo's energy function
            # energy_fn can handle batched inputs
            energies = energy_fn(self.energy_fn, phi_sa, psi_g)  # (num_goals,)
            
            # Return maximum (best) critic value
            # Higher critic value = closer to goal
            return jnp.max(energies)
        
        # Compute for all envs
        scores = jax.vmap(compute_critic_for_env)(final_state)  # (num_envs,)
        
        return scores
    
    @property
    def name(self) -> str:
        return f"min_critic_distance_{self.energy_fn}"


class WeightedRewardDistanceHeuristic(ForkHeuristic):
    """Combined heuristic using both reward and goal distance.
    
    Score = alpha * cumulative_reward + beta * (-distance_to_goal)
    """
    
    def __init__(
        self,
        state_size: int,
        goal_indices: tuple,
        reward_weight: float = 1.0,
        distance_weight: float = 1.0,
    ):
        """Initialize weighted heuristic.
        
        Args:
            state_size: Size of the state portion of observation
            goal_indices: Indices corresponding to goal in observation
            reward_weight: Weight for reward component
            distance_weight: Weight for distance component
        """
        self.state_size = state_size
        self.goal_indices = jnp.array(goal_indices)
        self.reward_weight = reward_weight
        self.distance_weight = distance_weight
    
    def evaluate(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> jnp.ndarray:
        """Evaluate based on weighted combination of reward and distance.
        
        Args:
            env_state: Current environment state
            transitions: Transitions with shape (unroll_length, num_envs, ...)
            env_info: Additional environment information
            
        Returns:
            scores: Weighted combination score (num_envs,)
        """
        # Cumulative reward component
        cumulative_rewards = jnp.sum(transitions.reward, axis=0)
        
        # Distance component
        current_obs = env_state.obs
        current_state = current_obs[:, :self.state_size]
        goal = current_obs[:, self.goal_indices]
        # Compare first goal_size dimensions of state with goal
        achieved_goal = current_state[:, :goal.shape[-1]]
        distances = jnp.linalg.norm(achieved_goal - goal, axis=-1)
        
        # Combine with weights
        scores = (
            self.reward_weight * cumulative_rewards
            - self.distance_weight * distances
        )
        return scores
    
    @property
    def name(self) -> str:
        return f"weighted_reward_distance_r{self.reward_weight}_d{self.distance_weight}"


def create_heuristic(
    heuristic_type: Literal[
        "cumulative_reward",
        "final_reward",
        "goal_distance",
        "weighted_reward_distance",
        "min_critic_distance",
    ],
    **kwargs,
) -> ForkHeuristic:
    """Factory function to create heuristics.
    
    Args:
        heuristic_type: Type of heuristic to create
        **kwargs: Additional arguments for heuristic initialization
        
    Returns:
        Instantiated heuristic
        
    Examples:
        >>> h1 = create_heuristic("cumulative_reward")
        >>> h2 = create_heuristic("goal_distance", state_size=8, goal_indices=(0,1))
        >>> h3 = create_heuristic(
        ...     "weighted_reward_distance",
        ...     state_size=8,
        ...     goal_indices=(0,1),
        ...     reward_weight=0.5,
        ...     distance_weight=2.0,
        ... )
    """
    if heuristic_type == "cumulative_reward":
        return CumulativeRewardHeuristic()
    elif heuristic_type == "final_reward":
        return FinalRewardHeuristic()
    elif heuristic_type == "goal_distance":
        return GoalDistanceHeuristic(
            state_size=kwargs["state_size"],
            goal_indices=kwargs["goal_indices"],
        )
    elif heuristic_type == "weighted_reward_distance":
        return WeightedRewardDistanceHeuristic(
            state_size=kwargs["state_size"],
            goal_indices=kwargs["goal_indices"],
            reward_weight=kwargs.get("reward_weight", 1.0),
            distance_weight=kwargs.get("distance_weight", 1.0),
        )
    elif heuristic_type == "min_critic_distance":
        return MinCriticDistanceHeuristic(
            state_size=kwargs["state_size"],
            goal_indices=kwargs["goal_indices"],
            possible_goals=kwargs["possible_goals"],
            actor_apply_fn=kwargs["actor_apply_fn"],
            actor_params=kwargs["actor_params"],
            sa_encoder_apply_fn=kwargs["sa_encoder_apply_fn"],
            g_encoder_apply_fn=kwargs["g_encoder_apply_fn"],
            critic_params=kwargs["critic_params"],
            energy_fn=kwargs.get("energy_fn", "norm"),
        )
    else:
        raise ValueError(f"Unknown heuristic type: {heuristic_type}")
