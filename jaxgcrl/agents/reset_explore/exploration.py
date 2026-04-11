"""Exploration metrics for ``ResetExplore`` fork mode (SMC, etc.).

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
) -> Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]:
    """Return ``metric(rng, states, goals, proposer_state) -> (num_envs,)`` scalars."""
    del env, num_envs, num_candidates, discounting
    if name == "q_epistemic":
        return _create_q_epistemic_exploration_metric(
            state_size=state_size,
            goal_indices=goal_indices,
            actor=actor,
            critic=critic,
        )
    raise ValueError(f"Unknown exploration metric: {name}")


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
    del goal_indices
    if state_size is None:
        raise ValueError("state_size is required for q_epistemic exploration metric")

    def metric(
        rng: jax.Array,
        states: jnp.ndarray,
        goals: jnp.ndarray,
        proposer_state: GoalProposerState,
    ) -> jnp.ndarray:
        actor_params = proposer_state.actor_params
        critic_params = proposer_state.critic_params
        full_critic_params = reconstruct_full_critic_params(critic_params)

        def one_state(state: jnp.ndarray, goal: jnp.ndarray, key: jax.Array) -> jnp.ndarray:
            obs = jnp.concatenate([state, goal], axis=-1)
            key, a_key = jax.random.split(key)
            action = actor.sample_actions(
                actor_params,
                obs[None, :],
                a_key,
                is_deterministic=True,
            )[0]
            q_vals = critic.apply(
                full_critic_params,
                obs[None, :],
                action[None, :],
            )[0]
            return jnp.std(q_vals)

        keys = jax.random.split(rng, states.shape[0])
        return jax.vmap(one_state)(states, goals, keys)

    return metric
