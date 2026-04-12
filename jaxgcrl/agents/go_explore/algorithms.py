from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState

from .losses import (
    energy_fn,
    update_actor_and_alpha as crl_update_actor,
    update_actor_sac,
    update_critic as crl_update_critic,
    update_critic_sac,
)
from .networks import Actor as ActorNetwork, Encoder, QNetwork
from .types import Actor, Critic, TrainingState, Transition
from .utils import flatten_batch



def _reshape_and_permute_transitions(transitions: Transition, process_key: jnp.ndarray, batch_size: int):
    """Shared helper to reshape and permute transitions after processing.
    
    Args:
        transitions: Transitions to reshape and permute
        process_key: Random key for permutation
        batch_size: Batch size for final reshaping
        
    Returns:
        Tuple of (reshaped_and_permuted_transitions, new_process_key)
    """
    # Reshape: flatten (num_envs, episode_length-1, ...) -> (total_transitions, ...)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1,) + x.shape[2:], order="F"), transitions
    )
    
    # Permute transitions
    permute_key, new_process_key = jax.random.split(process_key)
    permutation = jax.random.permutation(permute_key, len(transitions.observation))
    transitions = jax.tree_util.tree_map(lambda x: x[permutation], transitions)
    
    # Reshape into batches: (total_transitions, ...) -> (num_batches, batch_size, ...)
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (-1, batch_size) + x.shape[1:]),
        transitions,
    )
    
    return transitions, new_process_key

class CRLActor(Actor):
    """CRL Actor implementation."""
    
    def __init__(self, action_size: int, network_width: int, network_depth: int,
                 skip_connections: int, use_relu: bool, use_ln: bool,
                 discounting: float, state_size: int, goal_indices: tuple):
        self.network = ActorNetwork(
            action_size=action_size,
            network_width=network_width,
            network_depth=network_depth,
            skip_connections=skip_connections,
            use_relu=use_relu,
            use_ln=use_ln,
        )
        self.discounting = discounting
        self.state_size = state_size
        self.goal_indices = goal_indices
    
    def __call__(self, x):
        return self.network(x)
    
    def init(self, key, x):
        return self.network.init(key, x)
    
    def sample_actions(self, params, obs, key, is_deterministic: bool = False):
        means, log_stds = self.apply(params, obs)
        if is_deterministic:
            return nn.tanh(means)
        return nn.tanh(means + jnp.exp(log_stds) * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

    def apply(self, params, obs):
        """Apply actor network to get mean and log_std."""
        return self.network.apply(params, obs)

    def update(self, context: Dict[str, Any], networks: Dict[str, Any],
               transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
        """Update actor and alpha for CRL."""
        return crl_update_actor(context, networks, transitions, training_state, key)
    
    def process_transitions(self, transitions: Transition, process_key: jnp.ndarray, 
                           batch_size: int, discounting: float, state_size: int, goal_indices: tuple,
                           goal_reach_thresh: float, use_her: bool):
        """Process transitions using CRL's flatten_batch, then reshape and permute."""
        # CRL-specific: flatten_batch for future state sampling
        buffer_config = (
            discounting,
            state_size,
            tuple(goal_indices),
        )
        flatten_key, permute_key = jax.random.split(process_key)
        batch_keys = jax.random.split(flatten_key, transitions.observation.shape[0])
        transitions = jax.vmap(flatten_batch, in_axes=(None, 0, 0))(
            buffer_config, transitions, batch_keys
        )

        # Use shared reshape and permute logic
        return _reshape_and_permute_transitions(transitions, permute_key, batch_size)


class CRLCritic(Critic):
    """CRL Critic implementation using encoders with support for multiple critics."""
    
    def __init__(self, repr_dim: int, network_width: int, network_depth: int,
                 skip_connections: int, use_relu: bool, use_ln: bool,
                 energy_fn: str, state_size: int, action_size: int, goal_indices: tuple,
                 n_critics: int = 1):
        # Create n_critics pairs of encoders
        self.sa_encoders = [
            Encoder(
            repr_dim=repr_dim,
            network_width=network_width,
            network_depth=network_depth,
            skip_connections=skip_connections,
            use_relu=use_relu,
            use_ln=use_ln,
        )
            for _ in range(n_critics)
        ]
        self.g_encoders = [
            Encoder(
            repr_dim=repr_dim,
            network_width=network_width,
            network_depth=network_depth,
            skip_connections=skip_connections,
            use_relu=use_relu,
            use_ln=use_ln,
        )
            for _ in range(n_critics)
        ]
        self.energy_fn = energy_fn
        self.state_size = state_size
        self.action_size = action_size
        self.goal_indices = goal_indices
        self.n_critics = n_critics
    
    def compute_representations(self, params, state, action, goal):
        """Compute sa_repr and g_repr for contrastive loss for all critics.
        
        Args:
            params: Critic parameters (dict with sa_encoder_i and g_encoder_i for each critic i)
            state: State with shape (..., state_size)
            action: Action with shape (..., action_size)
            goal: Goal with shape (..., goal_size)
            
        Returns:
            Tuple of (sa_repr_list, g_repr_list) where each is a list of representations for each critic
        """
        sa_input = jnp.concatenate([state, action], axis=-1)
        sa_repr_list = []
        g_repr_list = []
        for i in range(self.n_critics):
            sa_repr = self.sa_encoders[i].apply(params[f"sa_encoder_{i}"], sa_input)
            g_repr = self.g_encoders[i].apply(params[f"g_encoder_{i}"], goal)
            sa_repr_list.append(sa_repr)
            g_repr_list.append(g_repr)
        return sa_repr_list, g_repr_list
    
    def apply(self, params, obs, actions):
        """Apply critic to compute Q-values using the API for all critics.
        
        Args:
            params: Critic parameters (dict with sa_encoder_i and g_encoder_i for each critic i)
            obs: Observations with shape (..., obs_size) where obs = [state, goal]
            actions: Actions with shape (..., action_size)
            
        Returns:
            Q-values with shape (..., n_critics) - concatenated Q-values from all critics
        """
        state = obs[..., :self.state_size]
        goal = obs[..., self.state_size:]
        sa_repr_list, g_repr_list = self.compute_representations(params, state, actions, goal)
        
        # Compute Q-value for each critic
        q_values = []
        for sa_repr, g_repr in zip(sa_repr_list, g_repr_list):
            q_value = energy_fn(self.energy_fn, sa_repr, g_repr)
            q_values.append(q_value[..., None])  # Shape: (..., 1)
        
        return jnp.concatenate(q_values, axis=-1)  # Shape: (..., n_critics)
    
    def __call__(self, obs, actions):
        """Compute Q-value using encoders and energy function for all critics.
        
        Args:
            obs: Observations with shape (..., obs_size) where obs = [state, goal]
            actions: Actions with shape (..., action_size)
            
        Returns:
            Q-values with shape (..., n_critics) - concatenated Q-values from all critics
        """
        # Extract state and goal from obs
        state = obs[..., :self.state_size]
        goal = obs[..., self.state_size:]
        
        # Compute representations using encoders (no params needed for __call__)
        sa_input = jnp.concatenate([state, actions], axis=-1)
        q_values = []
        for sa_encoder, g_encoder in zip(self.sa_encoders, self.g_encoders):
            sa_repr = sa_encoder(sa_input)
            g_repr = g_encoder(goal)
            # Compute energy (Q-value)
            q_value = energy_fn(self.energy_fn, sa_repr, g_repr)
            q_values.append(q_value[..., None])  # Shape: (..., 1)
        
        return jnp.concatenate(q_values, axis=-1)  # Shape: (..., n_critics)
    
    def init(self, key, x):
        # x is dummy obs for shape (batch_size, obs_size)
        # For sa_encoder: input is (state + action)
        dummy_sa = jnp.zeros((x.shape[0], self.state_size + self.action_size))
        # For g_encoder: input is goal (last len(goal_indices) elements of obs)
        goal_size = len(self.goal_indices) if len(self.goal_indices) > 0 else (x.shape[-1] - self.state_size)
        dummy_goal = jnp.zeros((x.shape[0], goal_size))
        
        params = {}
        keys = jax.random.split(key, self.n_critics * 2)
        for i in range(self.n_critics):
            key1, key2 = keys[i * 2], keys[i * 2 + 1]
            sa_params = self.sa_encoders[i].init(key1, dummy_sa)
            g_params = self.g_encoders[i].init(key2, dummy_goal)
            params[f"sa_encoder_{i}"] = sa_params
            params[f"g_encoder_{i}"] = g_params
        return params
    
    def create_critic_states(self, critic_params: dict, learning_rate: float, grad_clip: float = 0.0) -> Tuple[TrainState, ...]:
        """Create separate TrainState for each critic from full critic params (CRL structure)."""
        critic_states = []
        for i in range(self.n_critics):
            critic_i_params = {
                "sa_encoder": critic_params[f"sa_encoder_{i}"],
                "g_encoder": critic_params[f"g_encoder_{i}"],
            }
            if grad_clip > 0:
                tx = optax.chain(
                    optax.clip_by_global_norm(grad_clip),
                    optax.adam(learning_rate=learning_rate, eps=1e-5),
                )
            else:
                tx = optax.adam(learning_rate=learning_rate)
            critic_i_state = TrainState.create(
                apply_fn=None,  # Not needed for individual critic updates
                params=critic_i_params,
                tx=tx,
            )
            critic_states.append(critic_i_state)
        return tuple(critic_states)
    
    def update(self, context: Dict[str, Any], networks: Dict[str, Any],
               transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
        """Update critic for CRL."""
        return crl_update_critic(context, networks, transitions, training_state, key)


class SACActor(Actor):
    """SAC Actor implementation."""
    
    def __init__(self, action_size: int, network_width: int, network_depth: int,
                 use_relu: bool, use_ln: bool):
        self.network = ActorNetwork(
            action_size=action_size,
            network_width=network_width,
            network_depth=network_depth,
            skip_connections=0,
            use_relu=use_relu,
            use_ln=use_ln,
        )
    
    def __call__(self, x):
        return self.network(x)
    
    def init(self, key, x):
        return self.network.init(key, x)
    
    def apply(self, params, x):
        return self.network.apply(params, x)
    
    def sample_actions(self, params, obs, key, is_deterministic: bool = False):
        means, log_stds = self.apply(params, obs)
        if is_deterministic:
            return nn.tanh(means)
        return nn.tanh(means + jnp.exp(log_stds) * jax.random.normal(key, shape=means.shape, dtype=means.dtype))

    def update(self, context: Dict[str, Any], networks: Dict[str, Any],
               transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
        """Update actor for SAC."""
        return update_actor_sac(context, networks, transitions, training_state, key)
    
    def process_transitions(self, transitions: Transition, process_key: jnp.ndarray,
                           batch_size: int, discounting: float, state_size: int, goal_indices: tuple,
                           goal_reach_thresh: float, use_her: bool):
        """Process transitions for SAC: optionally apply HER, then reshape and permute."""
        # SAC: Apply HER if enabled (matching original SAC implementation)
        if use_her:
            def apply_her(transition: Transition) -> Transition:
                """Apply HER to a single transition (matching original SAC flatten_batch)."""
                # Find truncation indexes if present
                seq_len = transition.observation.shape[0]
                arrangement = jnp.arange(seq_len)
                is_future_mask = jnp.array(arrangement[:, None] < arrangement[None], dtype=jnp.float32)
                single_trajectories = jnp.concatenate(
                    [transition.extras["state_extras"]["traj_id"][:, jnp.newaxis].T] * seq_len,
                    axis=0,
                )

                # final_step_mask.shape == (seq_len, seq_len)
                final_step_mask = (
                    is_future_mask * jnp.equal(single_trajectories, single_trajectories.T) + jnp.eye(seq_len) * 1e-5
                )
                final_step_mask = jnp.logical_and(
                    final_step_mask,
                    transition.extras["state_extras"]["truncation"][None, :],
                )
                non_zero_columns = jnp.nonzero(final_step_mask, size=seq_len)[1]

                # If final state is not present use original goal (i.e. don't change anything)
                new_goals_idx = jnp.where(non_zero_columns == 0, arrangement, non_zero_columns)
                binary_mask = jnp.logical_and(non_zero_columns, non_zero_columns)

                # Extract goal from observations (matching original SAC exactly - line 115-118)
                # Original: binary_mask * obs[new_goals_idx][:, goal_indices] + (1-binary_mask) * obs[new_goals_idx][:, state_dim:]
                selected_obs = transition.observation[new_goals_idx]
                new_goals = (
                    binary_mask[:, None] * selected_obs[:, jnp.array(goal_indices)]
                    + jnp.logical_not(binary_mask)[:, None]
                    * selected_obs[:, state_size:]
                )

                # Transform observation: replace goal with new_goals
                state = transition.observation[:, :state_size]
                new_obs = jnp.concatenate([state, new_goals], axis=1)

                # Recalculate reward (matching original SAC line 126)
                dist = jnp.linalg.norm(new_obs[:, state_size:] - new_obs[:, jnp.array(goal_indices)], axis=1)
                new_reward = jnp.array(dist < goal_reach_thresh, dtype=float)  # Matching original SAC line 127

                # Transform next observation
                next_state = transition.next_observation[:, :state_size]
                new_next_obs = jnp.concatenate([next_state, new_goals], axis=1)

                return transition._replace(
                    observation=jnp.squeeze(new_obs),
                    next_observation=jnp.squeeze(new_next_obs),
                    reward=jnp.squeeze(new_reward),
                )
            
            # Apply HER to each transition in the batch
            transitions = jax.vmap(apply_her)(transitions)
        
        # Use shared reshape and permute logic
        return _reshape_and_permute_transitions(transitions, process_key, batch_size)


class SACCritic(Critic):
    """SAC Critic (Q-network) implementation."""
    
    def __init__(self, obs_size: int, action_size: int, network_width: int,
                 network_depth: int, use_relu: bool, use_ln: bool, n_critics: int):
        self.network = QNetwork(
            obs_size=obs_size,
            action_size=action_size,
            network_width=network_width,
            network_depth=network_depth,
            use_relu=use_relu,
            use_ln=use_ln,
            n_critics=n_critics,
        )
        self.n_critics = n_critics
        self.action_size = action_size
    
    def __call__(self, obs, actions):
        return self.network(obs, actions)
    
    def init(self, key, x):
        # x is dummy obs for shape
        dummy_action = jnp.zeros((x.shape[0], self.action_size))
        # network.init() returns {"params": {...}}, but we need to return just the inner dict
        # to match the pattern used by create_critic_states and CRL
        variables = self.network.init(key, x, dummy_action)
        return variables["params"]
    
    def apply(self, params, obs, actions):
        # params is the inner dict (unwrapped), need to wrap it for Flax network.apply()
        return self.network.apply({"params": params}, obs, actions)
    
    def create_critic_states(self, critic_params: dict, learning_rate: float, grad_clip: float = 0.0) -> Tuple[TrainState, ...]:
        """Create separate TrainState for each critic from full critic params (SAC structure)."""
        critic_states = []
        for i in range(self.n_critics):
            # SAC structure: critic_{i}_hidden_{j} and critic_{i}_output
            critic_i_params = {}
            for key, value in critic_params.items():
                if key.startswith(f"critic_{i}_"):
                    # Remove the "critic_{i}_" prefix to get the layer name
                    layer_name = key[len(f"critic_{i}_"):]
                    critic_i_params[layer_name] = value
            if grad_clip > 0:
                tx = optax.chain(
                    optax.clip_by_global_norm(grad_clip),
                    optax.adam(learning_rate=learning_rate, eps=1e-5),
                )
            else:
                tx = optax.adam(learning_rate=learning_rate)
            critic_i_state = TrainState.create(
                apply_fn=None,  # Not needed for individual critic updates
                params=critic_i_params,
                tx=tx,
            )
            critic_states.append(critic_i_state)
        return tuple(critic_states)
    
    def update(self, context: Dict[str, Any], networks: Dict[str, Any],
               transitions: Transition, training_state: TrainingState, key: jnp.ndarray):
        """Update critic for SAC."""
        return update_critic_sac(context, networks, transitions, training_state, key)


def get_algorithm(agent_type: str, **kwargs):
    """Factory function to get algorithm components."""
    if agent_type == "crl":
        action_size = kwargs.get("action_size")
        actor = CRLActor(
            action_size=action_size,
            network_width=kwargs.get("h_dim", 256),
            network_depth=kwargs.get("n_hidden", 4),
            skip_connections=kwargs.get("skip_connections", 0),
            use_relu=kwargs.get("use_relu", False),
            use_ln=False,
            discounting=kwargs.get("discounting", 0.99),
            state_size=kwargs.get("state_size"),
            goal_indices=kwargs.get("goal_indices"),
        )
        critic = CRLCritic(
            repr_dim=kwargs.get("repr_dim", 64),
            network_width=kwargs.get("h_dim", 256),
            network_depth=kwargs.get("n_hidden", 4),
            skip_connections=kwargs.get("skip_connections", 0),
            use_relu=kwargs.get("use_relu", False),
            use_ln=kwargs.get("use_ln", True),
            energy_fn=kwargs.get("energy_fn", "norm"),
            state_size=kwargs.get("state_size"),
            action_size=kwargs.get("action_size"),
            goal_indices=kwargs.get("goal_indices"),
            n_critics=kwargs.get("n_critics", 1),
        )
        return actor, critic
    elif agent_type == "sac":
        action_size = kwargs.get("action_size")
        obs_size = kwargs.get("obs_size")
        actor = SACActor(
            action_size=action_size,
            network_width=kwargs.get("h_dim", 256),
            network_depth=kwargs.get("n_hidden", 4),
            use_relu=kwargs.get("use_relu", False),
            use_ln=kwargs.get("use_ln", True),
        )
        critic = SACCritic(
            obs_size=obs_size,
            action_size=action_size,
            network_width=kwargs.get("h_dim", 256),
            network_depth=kwargs.get("n_hidden", 4),
            use_relu=kwargs.get("use_relu", False),
            use_ln=kwargs.get("use_ln", True),
            n_critics=kwargs.get("n_critics", 2),
        )
        return actor, critic
    else:
        raise ValueError(f"Unknown agent_type: {agent_type}")
