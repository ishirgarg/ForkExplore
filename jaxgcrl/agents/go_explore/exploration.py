"""Exploration metrics for GoExplore fork mode (SMC, etc.).

Factory pattern mirrors ``goal_proposers.create_goal_proposer``: register new
metrics by extending ``create_exploration_metric`` without changing call sites.
"""

from typing import Any, Callable, Optional, Protocol, Tuple

import jax
import jax.numpy as jnp

from jaxgcrl.agents.go_explore.algorithms_utils import reconstruct_full_critic_params
from jaxgcrl.agents.go_explore.types import GoalProposerState


class ExplorationMetric(Protocol):
    """Maps a batch of states to one scalar exploration value per state."""

    def __call__(
        self,
        rng: jax.Array,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        proposer_state: GoalProposerState,
    ) -> jnp.ndarray:
        """Args:
            rng: PRNG key.
            states: ``(num_envs, state_size)`` state vectors.
            goals: ``(num_envs, goal_dim)`` goals paired with each state (full obs prefix/suffix).
            proposer_state: Actor/critic params (and optional buffer sample — unused here).

        Returns:
            ``(num_envs,)`` float32 exploration scores.
        """
        ...


def create_exploration_metric(
    name: str,
    env: Any,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[Tuple[int, ...]] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
    discounting: float = 0.99,
    rnd_module: Optional[Any] = None,
) -> Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]:
    """Return ``metric(rng, states, goals, proposer_state) -> (num_envs,)`` scalars."""
    if name == "q_epistemic":
        return _create_q_epistemic_exploration_metric(
            state_size=state_size,
            goal_indices=goal_indices,
            actor=actor,
            critic=critic,
        )
    if name == "rnd":
        if rnd_module is None:
            raise ValueError("rnd_module is required for rnd exploration metric")
        return _create_rnd_exploration_metric(
            state_size=state_size,
            rnd_module=rnd_module,
        )
    if name == "log_density":
        return _create_log_density_exploration_metric(
            state_size=state_size,
            goal_indices=goal_indices,
        )
    raise ValueError(f"Unknown exploration metric: {name}")


def _create_log_density_exploration_metric(
    state_size: Optional[int],
    goal_indices: Optional[Tuple[int, ...]],
    kde_bandwidth: float = 0.1,
    eps: float = 1e-12,
) -> Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]:
    """Negative-log KDE density in goal-space, MEGA-style.

    Fits a Gaussian KDE to all achieved goals in the current replay-buffer
    sample (same construction as ``create_mega_goal_proposer``) and scores
    each query state by ``-log(density + eps)`` — lower density → higher
    exploration score.
    """
    if state_size is None:
        raise ValueError("state_size is required for log_density exploration metric")
    if goal_indices is None:
        raise ValueError("goal_indices is required for log_density exploration metric")

    goal_idx_array = jnp.array(goal_indices)

    def metric(
        rng: jax.Array,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        proposer_state: GoalProposerState,
    ) -> jnp.ndarray:
        # Avoid a top-level circular import between exploration and goal_proposers.
        from jaxgcrl.agents.go_explore.goal_proposers import _jax_gaussian_kde

        transitions_sample = proposer_state.transitions_sample
        obs_flat = jnp.reshape(
            transitions_sample.observation,
            (-1, transitions_sample.observation.shape[-1]),
        )
        all_goals = obs_flat[:, :state_size][:, goal_idx_array]  # (N_buf, goal_dim)

        query_goals = states[:, :state_size][:, goal_idx_array]  # (num_envs, goal_dim)

        densities = _jax_gaussian_kde(query_goals, all_goals, kde_bandwidth)
        return -jnp.log(densities + eps)

    return metric


def _create_rnd_exploration_metric(
    state_size: Optional[int],
    rnd_module: Any,
) -> Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]:
    """MSE prediction error between RND predictor and frozen random target.

    Both networks see the (optionally normalised) state.  Their outputs are
    compared elementwise and the mean-squared-error per state is returned as
    the exploration score.
    """
    if state_size is None:
        raise ValueError("state_size is required for rnd exploration metric")

    def metric(
        rng: jax.Array,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        proposer_state: GoalProposerState,
    ) -> jnp.ndarray:
        target_params = proposer_state.rnd_target_params
        predictor_params = proposer_state.rnd_predictor_params
        rnd_norm_p = proposer_state.rnd_obs_normalizer_params

        # Whiten state slice with the dedicated RND running stats and clip to
        # [-5, 5].  rnd_obs_normalizer_params is always populated when RND is
        # active (seeded from the prefill rollouts).
        net_states = states[:, :state_size]
        net_states = jnp.clip(
            (net_states - rnd_norm_p.mean) / rnd_norm_p.std, -5.0, 5.0,
        )

        target_out = rnd_module.apply(target_params, net_states)
        pred_out = rnd_module.apply(predictor_params, net_states)
        return jnp.mean((pred_out - target_out) ** 2, axis=-1)

    return metric


def _create_q_epistemic_exploration_metric(
    state_size: Optional[int],
    goal_indices: Optional[Tuple[int, ...]],
    actor: Any,
    critic: Any,
) -> Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]:
    """Std dev across critic ensemble Q-values as exploration signal."""
    if state_size is None:
        raise ValueError("state_size is required for q_epistemic exploration metric")

    def metric(
        rng: jax.Array,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        proposer_state: GoalProposerState,
    ) -> jnp.ndarray:
        from brax.training.acme import running_statistics
        actor_params = proposer_state.actor_params
        critic_params = proposer_state.critic_params
        full_critic_params = reconstruct_full_critic_params(critic_params)
        norm_p = proposer_state.normalizer_params

        def one_state(state: jnp.ndarray, goal: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
            obs = jnp.concatenate([state, goal], axis=-1)
            net_obs = obs[None, :]
            if norm_p is not None:
                net_obs = running_statistics.normalize(net_obs, norm_p)
            key, a_key = jax.random.split(key)
            action = actor.sample_actions(
                actor_params,
                net_obs,
                a_key,
                is_deterministic=True,
            )[0]
            q_vals = critic.apply(
                full_critic_params,
                net_obs,
                action[None, :],
            )[0]
            return jnp.std(q_vals)

        keys = jax.random.split(rng, states.shape[0])
        return jax.vmap(one_state)(states, goals, keys)

    return metric
