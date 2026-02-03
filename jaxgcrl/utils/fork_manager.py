"""Fork manager for agent forking mechanism.

This module handles the logic for forking agents: evaluating performance,
determining which agents to fork, and managing state transfers.
"""

from typing import Any, Dict, NamedTuple, Tuple, Union, Optional

import jax
import jax.experimental.io_callback as io_callback
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from brax import envs
from brax.v1 import envs as envs_v1
import wandb

from .fork_heuristics import ForkHeuristic


class ForkDecision(NamedTuple):
    """Contains information about a forking decision.
    
    Attributes:
        should_fork: Whether forking should occur (bool)
        bottom_k_indices: Indices of bottom K agents to be replaced (shape: K)
        top_k_indices: Indices of top K agents to fork from (shape: K)
        fork_mask: Boolean mask indicating which envs are being forked (shape: num_envs)
        scores: Performance scores for all agents (shape: num_envs)
    """
    should_fork: jnp.ndarray  # scalar bool
    bottom_k_indices: jnp.ndarray  # (K,)
    top_k_indices: jnp.ndarray  # (K,)
    fork_mask: jnp.ndarray  # (num_envs,) bool
    scores: jnp.ndarray  # (num_envs,)


class ForkManager:
    """Manages agent forking logic.
    
    The ForkManager evaluates agent performance using a heuristic and determines
    which agents should be forked (replaced with copies from top performers).
    """
    
    def __init__(
        self,
        num_envs: int,
        fork_k: int,
        heuristic: ForkHeuristic,
        enabled: bool = True,
    ):
        """Initialize ForkManager.
        
        Args:
            num_envs: Total number of parallel environments
            fork_k: Number of agents to fork (bottom K replaced by top K)
            heuristic: Heuristic for evaluating agent performance
            enabled: Whether forking is enabled
        """
        self.num_envs = num_envs
        self.fork_k = fork_k
        self.heuristic = heuristic
        self.enabled = enabled
        
        # Validate parameters
        if fork_k < 0:
            raise ValueError(f"fork_k must be non-negative, got {fork_k}")
        if fork_k > num_envs // 2:
            raise ValueError(
                f"fork_k ({fork_k}) must be <= num_envs // 2 ({num_envs // 2})"
            )
    
    
    def evaluate_and_decide(
        self,
        env_state: Union[envs.State, envs_v1.State],
        transitions: Any,
        env_info: Dict[str, Any],
    ) -> ForkDecision:
        """Evaluate agents and decide which to fork.
        
        Args:
            env_state: Current environment state for all agents
            transitions: Recent transitions from unroll
            env_info: Additional environment information
            
        Returns:
            ForkDecision containing forking information
        """
        if not self.enabled or self.fork_k == 0:
            # No forking
            return ForkDecision(
                should_fork=jnp.array(False),
                bottom_k_indices=jnp.array([], dtype=jnp.int32),
                top_k_indices=jnp.array([], dtype=jnp.int32),
                fork_mask=jnp.zeros(self.num_envs, dtype=bool),
                scores=jnp.zeros(self.num_envs),
            )
        
        # Evaluate agent performance
        scores = self.heuristic.evaluate(env_state, transitions, env_info)
        
        # Get bottom K and top K indices
        bottom_k_indices, top_k_indices = self._select_fork_indices(scores)
        
        # Create fork mask (True for agents being forked)
        fork_mask = self._create_fork_mask(bottom_k_indices)
        
        fork_decision = ForkDecision(
            should_fork=jnp.array(True),
            bottom_k_indices=bottom_k_indices,
            top_k_indices=top_k_indices,
            fork_mask=fork_mask,
            scores=scores,
        )
        
        # Log visualization if needed (using io_callback for JIT compatibility)
        assert "env_steps" in env_info, "env_steps must be provided in env_info"
        assert "goal_indices" in env_info, "goal_indices must be provided in env_info"

        env_steps = env_info.get("env_steps", jnp.array(0))
        goal_indices = env_info.get("goal_indices", None)
        
        # Use io_callback to call visualization from JIT
        self._maybe_log_visualization(
            env_state=env_state,
            fork_decision=fork_decision,
            env_steps=env_steps,
            goal_indices=goal_indices,
            log_interval=1_000_000,
        )
        
        return fork_decision
    
    def _select_fork_indices(
        self,
        scores: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Select bottom K and top K agents based on scores.
        
        Args:
            scores: Performance scores (num_envs,)
            
        Returns:
            bottom_k_indices: Indices of bottom K agents (K,)
            top_k_indices: Indices of top K agents (K,)
        """
        # Get sorted indices (ascending order)
        sorted_indices = jnp.argsort(scores)
        
        # Bottom K are the first K (lowest scores)
        bottom_k_indices = sorted_indices[:self.fork_k]
        
        # Top K are the last K (highest scores)
        top_k_indices = sorted_indices[-self.fork_k:]
        
        return bottom_k_indices, top_k_indices
    
    def _create_fork_mask(self, bottom_k_indices: jnp.ndarray) -> jnp.ndarray:
        """Create boolean mask indicating which envs are being forked.
        
        Args:
            bottom_k_indices: Indices of agents being forked (K,)
            
        Returns:
            fork_mask: Boolean mask (num_envs,)
        """
        mask = jnp.zeros(self.num_envs, dtype=bool)
        mask = mask.at[bottom_k_indices].set(True)
        return mask
    
    def apply_forking(
        self,
        env_state: Union[envs.State, envs_v1.State],
        fork_decision: ForkDecision,
    ) -> Union[envs.State, envs_v1.State]:
        """Apply forking by copying state from top K to bottom K agents.
        
        Args:
            env_state: Current environment state
            fork_decision: Decision containing fork indices
            
        Returns:
            new_env_state: Updated environment state with forking applied
        """
        if not fork_decision.should_fork:
            return env_state
        
        # For each attribute in env_state, copy from top K to bottom K
        def fork_array(arr: jnp.ndarray) -> jnp.ndarray:
            """Copy values from top K indices to bottom K indices."""
            if arr.ndim == 0:
                # Scalar, no forking needed
                return arr
            
            # arr has shape (num_envs, ...) or just (num_envs,)
            forked_values = arr[fork_decision.top_k_indices]
            return arr.at[fork_decision.bottom_k_indices].set(forked_values)
        
        # Apply forking to all arrays in env_state
        new_env_state = jax.tree_util.tree_map(fork_array, env_state)
        
        return new_env_state
    
    def increment_trajectory_ids(
        self,
        env_state: Union[envs.State, envs_v1.State],
        fork_decision: ForkDecision,
    ) -> Union[envs.State, envs_v1.State]:
        """Increment trajectory IDs for forked agents.
        
        When agents are forked, they start new trajectories, so their
        trajectory IDs should be incremented.
        
        Args:
            env_state: Environment state after forking
            fork_decision: Decision containing fork mask
            
        Returns:
            new_env_state: State with incremented trajectory IDs for forked agents
        """
        if not fork_decision.should_fork:
            return env_state
        
        # Increment traj_id for forked agents
        current_traj_ids = env_state.info["traj_id"]
        new_traj_ids = jnp.where(
            fork_decision.fork_mask,
            current_traj_ids + 1,  # Increment for forked agents
            current_traj_ids,  # Keep same for non-forked agents
        )
        
        # Update env_state with new trajectory IDs
        new_info = env_state.info.copy()
        new_info["traj_id"] = new_traj_ids
        new_env_state = env_state.replace(info=new_info)
        
        return new_env_state
    
    
    def get_metrics(self, fork_decision: ForkDecision) -> Dict[str, jnp.ndarray]:
        """Get metrics about the forking decision.
        
        Args:
            fork_decision: Decision to extract metrics from
            
        Returns:
            metrics: Dictionary of metric name -> value
        """
        if not fork_decision.should_fork:
            return {
                "forking/enabled": jnp.array(0.0),
                "forking/num_forked": jnp.array(0.0),
            }
        
        metrics = {
            "forking/enabled": jnp.array(1.0),
            "forking/num_forked": jnp.array(float(self.fork_k)),
            "forking/mean_score": jnp.mean(fork_decision.scores),
            "forking/std_score": jnp.std(fork_decision.scores),
            "forking/min_score": jnp.min(fork_decision.scores),
            "forking/max_score": jnp.max(fork_decision.scores),
            "forking/bottom_k_mean_score": jnp.mean(
                fork_decision.scores[fork_decision.bottom_k_indices]
            ),
            "forking/top_k_mean_score": jnp.mean(
                fork_decision.scores[fork_decision.top_k_indices]
            ),
        }
        
        return metrics
    
    def _plot_fork_visualization_callback(
        self,
        positions: np.ndarray,
        scores: np.ndarray,
        top_k_indices: np.ndarray,
        bottom_k_indices: np.ndarray,
        env_steps: np.ndarray,
    ) -> None:
        """Python callback function to create and log wandb visualization.
        
        This is called from JIT-compiled code via io_callback.
        
        Args:
            positions: Agent positions (num_envs, 2)
            scores: Performance scores (num_envs,)
            top_k_indices: Indices of top K agents (K,)
            bottom_k_indices: Indices of bottom K agents (K,)
            env_steps: Current number of environment steps (numpy array, will be converted to int)
        """
        # Convert env_steps to int (it comes as a numpy array from io_callback)
        env_steps_int = int(env_steps.item() if hasattr(env_steps, 'item') else env_steps)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot colored by raw scores
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=scores,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='none',
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Score', rotation=270, labelpad=15)
        
        # Outline top K agents in lime green
        if len(top_k_indices) > 0:
            top_positions = positions[top_k_indices]
            ax.scatter(
                top_positions[:, 0],
                top_positions[:, 1],
                s=200,
                facecolors='none',
                edgecolors='lime',
                linewidths=3,
                label=f'Top {self.fork_k}',
            )
        
        # Outline bottom K agents in red
        if len(bottom_k_indices) > 0:
            bottom_positions = positions[bottom_k_indices]
            ax.scatter(
                bottom_positions[:, 0],
                bottom_positions[:, 1],
                s=200,
                facecolors='none',
                edgecolors='red',
                linewidths=3,
                label=f'Bottom {self.fork_k}',
            )
        
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Final Env States by Score (Step: {env_steps_int:,})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Log to wandb
        wandb.log({
            "forking/final_states_visualization": wandb.Image(fig),
        }, step=env_steps_int)
        
        plt.close(fig)
    
    def _maybe_log_visualization(
        self,
        env_state: Union[envs.State, envs_v1.State],
        fork_decision: ForkDecision,
        env_steps: jnp.ndarray,
        goal_indices: jnp.ndarray,
        log_interval: int = 1_000_000,
    ) -> None:
        """JIT-compatible method to conditionally log visualization.
        
        This method can be called from JIT-compiled code and uses io_callback
        to execute the plotting in Python.
        
        Args:
            env_state: Current environment state for all agents
            fork_decision: Fork decision containing scores and indices
            env_steps: Current number of environment steps (JAX array)
            goal_indices: Indices to use for extracting goal positions
            log_interval: Only log every N environment steps (default: 1M)
        """
        # Check if we should log (using JAX operations)
        should_log = (env_steps % log_interval == 0) & fork_decision.should_fork
        
        if not should_log:
            return
        
        # Extract positions using goal_indices
        x_pos = env_state.pipeline_state.x.pos
        goal_indices_array = jnp.array(goal_indices)
        
        assert len(goal_indices_array) == 2, "Goal indices must be 2D for plotting"
        positions = x_pos[:, goal_indices_array]
        
        # Convert JAX arrays to numpy for the callback
        # io_callback will automatically convert JAX arrays to numpy
        positions_np = positions
        scores_np = fork_decision.scores
        top_k_indices_np = fork_decision.top_k_indices
        bottom_k_indices_np = fork_decision.bottom_k_indices
        
        # Use io_callback to call Python plotting function from JIT
        # The callback will receive numpy arrays and can convert env_steps to int
        io_callback.call(
            self._plot_fork_visualization_callback,
            None,  # Return value shape (None = no return)
            positions_np,
            scores_np,
            top_k_indices_np,
            bottom_k_indices_np,
            env_steps,  # Pass as JAX array, convert to int in callback
        )