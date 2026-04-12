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
    
    def create_critic_states(self, critic_params: dict, learning_rate: float, grad_clip: float = 0.0) -> Tuple[TrainState, ...]:
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
    # Explore policy fields (None when explore_policy_type is None)
    explore_actor_state: Optional[TrainState] = None
    explore_critic_states: Optional[Tuple[TrainState, ...]] = None
    explore_alpha_state: Optional[TrainState] = None
    explore_target_critic_params: Optional[Any] = None
    # TLDR fields (None when explore_reward_type != "tldr")
    te_state: Optional[TrainState] = None
    dual_lam_state: Optional[TrainState] = None
    pbe_rms_state: Optional[Any] = None
    # PEG fields (None when neither explore_reward_type="peg" nor goal_proposer_name="peg")
    wm_ensemble_states: Optional[Tuple[TrainState, ...]] = None
    peg_rms_state: Optional[Any] = None
    # PEG latent-space fields (None when use_peg_latent_space=False)
    obs_encoder_state: Optional[TrainState] = None
    obs_decoder_state: Optional[TrainState] = None
    # RSSM fields (None when use_rssm=False)
    rssm_state: Optional[TrainState] = None
    disag_state: Optional[TrainState] = None


class Transition(NamedTuple):
    """Container for a transition"""

    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    discount: jnp.ndarray
    next_observation: Optional[jnp.ndarray] = None  # Required for SAC
    extras: jnp.ndarray = ()


@dataclass
class ExploreRewardState:
    """Carrier for mutable state used by explore reward functions.

    Pack from ``TrainingState`` before calling the reward fn, then unpack the
    returned state back into ``TrainingState``.  Fields unused by the active
    reward type will be ``None`` throughout.
    """

    # q_uncertainty: current explore policy params (read-only inside reward fn)
    explore_actor_params: Optional[Any] = None
    explore_critic_states: Optional[Any] = None
    # tldr: traj encoder + dual lambda + PBE running stats
    te_state: Optional[TrainState] = None
    dual_lam_state: Optional[TrainState] = None
    pbe_rms_state: Optional[Any] = None
    # peg: world model ensemble + PEG running stats + optional encoder
    wm_ensemble_states: Optional[Any] = None
    peg_rms_state: Optional[Any] = None
    obs_encoder_params: Optional[Any] = None


@dataclass
class GoalProposerState:
    """State for goal proposers that can be read from and written to."""

    transitions_sample: Any  # Transition sample from replay buffer
    actor_params: Optional[Any] = None  # Actor network parameters (for q_epistemic)
    critic_params: Optional[Any] = None  # Critic network parameters (for q_epistemic)
    te_params: Optional[Any] = None  # Traj encoder parameters (for TLDR goal proposer)
    wm_ensemble_params: Optional[Any] = None  # World model ensemble params (for PEG goal proposer)
    obs_encoder_params: Optional[Any] = None  # Encoder params (for latent-space PEG goal proposer)
    obs_decoder_params: Optional[Any] = None  # Decoder params (for latent-space PEG goal proposer)
    rssm_params: Optional[Any] = None   # RSSM world model params (for peg_rssm goal proposer)
    disag_params: Optional[Any] = None  # Disagreement ensemble params (for peg_rssm goal proposer)
    normalizer_params: Optional[Any] = None  # Running obs normalizer (for MPPI actor calls)