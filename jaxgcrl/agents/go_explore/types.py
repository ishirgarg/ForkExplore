from typing import Any, NamedTuple, Optional, Tuple

import flax.linen as nn
import jax.numpy as jnp
from flax.struct import dataclass
from flax.training.train_state import TrainState

@dataclass
class Actor:
    @nn.compact
    def __call__(self, x):
        pass

    def init(self, key, x):
        pass

    def sample_actions(self, params, obs, key, is_deterministic: bool):
        pass

    def update(self,  context, networks, transitions, training_state, actor_key):
        pass

    def process_transitions(self, transitions, process_key, batch_size, discounting, state_size, goal_indices, goal_reach_thresh, use_her):
        pass

class Critic:
    @nn.compact
    def __call__(self, obs, actions):
        pass

    def init(self, key, x):
        pass

    def update(self,  context, networks, transitions, training_state, critic_key):
        pass
    
    def create_critic_states(self, critic_params: dict, learning_rate: float) -> Tuple[TrainState, ...]:
        """Create separate TrainState for each critic from full critic params.
        
        Args:
            critic_params: Full critic parameters (structure depends on critic type)
            learning_rate: Learning rate for optimizer
            
        Returns:
            Tuple of TrainState objects, one for each critic
        """
        raise NotImplementedError("Subclasses must implement create_critic_states")

@dataclass
class TrainingState:
    """Contains training state for the learner"""

    env_steps: jnp.ndarray
    gradient_steps: jnp.ndarray
    experience_count: jnp.ndarray  # Counter for number of experiences collected
    actor_state: TrainState
    critic_states: Tuple[TrainState, ...]  # Separate TrainState for each critic
    alpha_state: Optional[TrainState] = None
    # SAC-specific fields
    target_critic_params: Optional[Any] = None  # Params for target Q-network
    normalizer_params: Optional[Any] = None  # Running statistics for observations
    policy_optimizer_state: Optional[Any] = None  # For SAC's policy optimizer
    q_optimizer_state: Optional[Any] = None  # For SAC's Q-network optimizer
    target_policy_params: Optional[Any] = None  # For SAC's target policy (if TD3-like)


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: Optional[jnp.ndarray] = None  # Required for SAC
    extras: jnp.ndarray = ()


@dataclass
class GoalProposerState:
    """State for goal proposers that can be read from and written to."""
    
    transitions_sample: Any  # Transition sample from replay buffer
    actor_params: Optional[Any] = None  # Actor network parameters (for q_epistemic)
    critic_params: Optional[Any] = None  # Critic network parameters (for q_epistemic)