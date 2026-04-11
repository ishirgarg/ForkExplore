"""Forking algorithms for GoExplore when ``fork_type`` is set."""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from jaxgcrl.agents.go_explore.types import GoalProposerState
from jaxgcrl.agents.go_explore.exploration import create_exploration_metric


def smc_fork(
    rng: jax.Array,
    states: jnp.ndarray,
    goals: jnp.ndarray,
    proposer_state: GoalProposerState,
    metric_fn: Callable[
        [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
        jnp.ndarray,
    ],
    temperature: float,
) -> jnp.ndarray:
    """Resample ``N`` states with replacement using softmax weights over exploration values."""
    n = states.shape[0]
    rng_metric, rng_idx = jax.random.split(rng)
    values = metric_fn(rng_metric, states, goals, proposer_state)
    logits = values / jnp.asarray(temperature, dtype=values.dtype)
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    weights = weights / jnp.sum(weights)
    log_w = jnp.log(weights + jnp.asarray(1e-10, dtype=weights.dtype))
    idx_keys = jax.random.split(rng_idx, n)
    indices = jax.vmap(lambda k: jax.random.categorical(k, log_w))(idx_keys)
    return states[indices]


ForkMetricFn = Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]
ForkResampleFn = Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]


def create_fork_fn(
    fork_type: str,
    env: Any,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[Tuple[int, ...]] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
    exploration_metric_name: str = "q_epistemic",
    fork_sampling_temperature: float = 1.0,
    discounting: float = 0.99,
) -> Tuple[ForkResampleFn, ForkMetricFn]:
    """Return ``(fork, exploration_metric)`` for resampling and visualization."""
    if fork_type == "smc":
        metric_fn = create_exploration_metric(
            exploration_metric_name,
            env,
            num_envs,
            num_candidates,
            state_size=state_size,
            goal_indices=goal_indices,
            actor=actor,
            critic=critic,
            discounting=discounting,
        )
        temp = fork_sampling_temperature

        def fork(
            rng: jax.Array,
            states: jnp.ndarray,
            goals: jnp.ndarray,
            proposer_state: GoalProposerState,
        ) -> jnp.ndarray:
            return smc_fork(
                rng, states, goals, proposer_state, metric_fn, temp,
            )

        return fork, metric_fn

    raise ValueError(f"Unknown fork_type: {fork_type}")
