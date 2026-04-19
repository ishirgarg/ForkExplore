"""Generic forking algorithms for GoExplore.

Each fork function has signature::

    fork(rng, states, scores, env_steps=None, positions=None)
        -> (new_states, forked_mask)

where ``states`` is a pytree with ``(num_envs, ...)`` leaves, ``scores`` is
``(num_envs,)`` (caller sets ``-inf`` for ineligible envs, e.g. go-phase),
``new_states`` is the same pytree with the selected envs' states replaced, and
``forked_mask`` is ``(num_envs,) bool`` — ``True`` for positions that were
replaced (so the caller can bump ``traj_id`` accordingly).

Positions where ``forked_mask`` is ``False`` are guaranteed unchanged
(``new_states[i] == states[i]``), which means no outer ESS gate is required at
the call site.  Ineligible (go-phase) envs are guaranteed to be neither
donors nor recipients of resampled state.
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from jax.experimental import io_callback
import wandb

from jaxgcrl.agents.go_explore.types import GoalProposerState
from jaxgcrl.agents.go_explore.exploration import create_exploration_metric
from jaxgcrl.agents.go_explore.visualization import visualize_fork_grid


ForkMetricFn = Callable[
    [jax.Array, jnp.ndarray, jnp.ndarray, GoalProposerState],
    jnp.ndarray,
]
ForkFn = Callable[..., Tuple[Any, jnp.ndarray]]


def _apply_mask_to_tree(gathered: Any, original: Any, mask: jnp.ndarray) -> Any:
    """Return ``where(mask, gathered, original)`` leaf-wise, tolerant of odd leaves."""
    n = mask.shape[0]

    def _where(g, o):
        if not hasattr(g, "shape") or g.ndim == 0 or g.shape[0] != n:
            return o
        m = jnp.reshape(mask, (n,) + (1,) * (g.ndim - 1))
        return jnp.where(m, g, o)

    return jax.tree.map(_where, gathered, original)


def _gather_tree(states: Any, indices: jnp.ndarray) -> Any:
    n = indices.shape[0]

    def _take(x):
        if not hasattr(x, "shape") or x.ndim == 0 or x.shape[0] != n:
            return x
        return x[indices]

    return jax.tree.map(_take, states)


def smc_fork(
    rng: jax.Array,
    states: Any,
    scores: jnp.ndarray,
    temperature: float,
    ess_fraction: float,
    env_steps: Optional[jnp.ndarray] = None,
    log_prefix: str = "fork",
) -> Tuple[Any, jnp.ndarray]:
    """Softmax-resample ``N`` positions with replacement, ESS-gated.

    Ineligible envs (``-inf`` scores) get weight 0, so they are never picked as
    donors; ``forked_mask`` is additionally AND-ed with ``isfinite(scores)`` so
    they are never marked as recipients either.

    ``log_prefix`` controls the wandb key namespace (e.g. ``"fork"``,
    ``"fork_explore"``, ``"fork_go"``) so callers running SMC twice per step
    can log both resamples without key collisions.
    """
    scores = scores.astype(jnp.float32)
    n = scores.shape[0]

    logits = scores / jnp.asarray(temperature, dtype=scores.dtype)
    logits = logits - jnp.max(logits)
    weights = jnp.exp(logits)
    weights = weights / (jnp.sum(weights) + 1e-10)
    log_w = jnp.log(weights + 1e-10)

    idx_keys = jax.random.split(rng, n)
    indices = jax.vmap(lambda k: jax.random.categorical(k, log_w))(idx_keys)

    eligible = jnp.isfinite(scores)
    n_eligible = jnp.sum(eligible.astype(jnp.float32))
    ess = 1.0 / (jnp.sum(weights ** 2) + 1e-10)
    ess_threshold = jnp.asarray(ess_fraction, dtype=ess.dtype) * n_eligible
    should_fire = ess < ess_threshold
    ess_fraction_empirical = ess / jnp.maximum(n_eligible, 1.0)

    forked_mask = jnp.broadcast_to(should_fire, (n,)) & eligible
    gathered = _gather_tree(states, indices)
    new_states = _apply_mask_to_tree(gathered, states, forked_mask)

    if env_steps is not None:
        prefix = log_prefix

        def _log_stats(ess_val, thr_val, fired_val, n_elig_val, frac_val, steps):
            wandb.log({
                f"{prefix}/ess": float(ess_val),
                f"{prefix}/ess_threshold": float(thr_val),
                f"{prefix}/resample_fired": float(fired_val),
                f"{prefix}/n_eligible": float(n_elig_val),
                f"{prefix}/ess_fraction": float(frac_val),
                f"{prefix}/env_steps": int(steps),
            })
            return jnp.array(0, dtype=jnp.int32)

        io_callback(
            _log_stats, jnp.array(0, dtype=jnp.int32),
            ess, ess_threshold, should_fire.astype(jnp.float32),
            n_eligible, ess_fraction_empirical, env_steps,
        )

    return new_states, forked_mask


def top_k_fork(
    rng: jax.Array,
    states: Any,
    scores: jnp.ndarray,
    k: int,
    env_steps: Optional[jnp.ndarray] = None,
    positions: Optional[jnp.ndarray] = None,
    x_bounds: Optional[jnp.ndarray] = None,
    y_bounds: Optional[jnp.ndarray] = None,
) -> Tuple[Any, jnp.ndarray]:
    """Overwrite the bottom-``k`` explore-phase envs (by ``scores``) with the
    top-``k`` explore-phase envs' states.

    Go-phase envs (``-inf`` scores) are excluded from BOTH top-k and bottom-k
    by score masking, and ``pair_valid`` drops any (donor, recipient) pair that
    would otherwise fall back on a go-phase env when ``n_explore < k``.
    """
    scores = scores.astype(jnp.float32)
    n = scores.shape[0]
    finite = jnp.isfinite(scores)

    # Top-k by score: ineligible envs have -inf so naturally sort last, but
    # when n_explore < k the remainder spills into go-phase; guarded below.
    top_indices = jax.lax.top_k(scores, k)[1]
    # Bottom-k by score: push ineligible to +inf in the working copy so they
    # never win the "smallest-k" race.
    finite_for_bottom = jnp.where(finite, scores, jnp.inf)
    bot_indices = jax.lax.top_k(-finite_for_bottom, k)[1]

    # Drop any (donor, recipient) pair where either side is go-phase OR the
    # pair would degenerate into a self-copy (top and bottom overlap at the
    # same rank — e.g. the median env when 2k > n_explore).
    top_valid = finite[top_indices]
    bot_valid = finite[bot_indices]
    non_self = top_indices != bot_indices
    pair_valid = top_valid & bot_valid & non_self

    # Build the gather index: identity by default; bottom-k positions pull
    # from their paired top-k donor ONLY when the pair is valid.
    gather_indices = jnp.arange(n).at[bot_indices].set(
        jnp.where(pair_valid, top_indices, bot_indices)
    )
    # Recipients actually changed: only bottom-k positions with a valid pair.
    forked_mask = jnp.zeros(n, dtype=jnp.bool_).at[bot_indices].set(pair_valid)
    # Donors actually used (for visualization).
    top_mask = jnp.zeros(n, dtype=jnp.bool_).at[top_indices].set(pair_valid)
    go_mask = ~finite

    gathered = _gather_tree(states, gather_indices)
    new_states = _apply_mask_to_tree(gathered, states, forked_mask)

    if env_steps is not None:
        def _log_stats(k_swapped_val, steps):
            wandb.log({
                "fork/k_swapped": float(k_swapped_val),
                "fork/resample_fired": 1.0,
                "fork/env_steps": int(steps),
            })
            return jnp.array(0, dtype=jnp.int32)

        io_callback(
            _log_stats, jnp.array(0, dtype=jnp.int32),
            jnp.sum(forked_mask.astype(jnp.float32)), env_steps,
        )

    # Grid viz: only possible when positions + bounds were provided.
    if (positions is not None and env_steps is not None
            and x_bounds is not None and y_bounds is not None):
        x_bounds_arr = jnp.asarray(x_bounds)
        y_bounds_arr = jnp.asarray(y_bounds)

        def _log_grid(pos, scr, tk, bk, go, xb, yb, steps):
            visualize_fork_grid(
                positions=pos, scores=scr,
                top_k_mask=tk, bottom_k_mask=bk, go_mask=go,
                x_bounds=xb, y_bounds=yb, env_steps=int(steps),
            )
            return jnp.array(0, dtype=jnp.int32)

        io_callback(
            _log_grid, jnp.array(0, dtype=jnp.int32),
            positions, scores, top_mask, forked_mask, go_mask,
            x_bounds_arr, y_bounds_arr, env_steps,
        )

    return new_states, forked_mask


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
    ess_fraction: float = 0.5,
    fork_all_envs: bool = False,
    fork_go_phase_ess_fraction: float = 0.5,
    fork_top_k: int = 0,
    discounting: float = 0.99,
    rnd_module: Optional[Any] = None,
) -> Tuple[ForkFn, ForkMetricFn]:
    """Return ``(fork, exploration_metric)``.

    ``fork(rng, states, scores, env_steps=None, positions=None, in_explore=None)
        -> (new_states, forked_mask)``

    When ``fork_all_envs=True`` (smc only), the returned closure runs two
    INDEPENDENT SMC resamples per call — one over the go-phase envs and one
    over the explore-phase envs — sharing ``fork_sampling_temperature`` but
    using ``fork_go_phase_ess_fraction`` vs ``ess_fraction`` for their ESS
    gates.  In that mode the caller must pass unmasked ``scores`` plus an
    ``in_explore`` boolean mask; the closure applies the per-phase masking.

    Environment ``x_bounds`` / ``y_bounds`` are closured in so the caller only
    needs to pass ``positions`` for the grid visualization to fire.
    """
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
        rnd_module=rnd_module,
    )

    x_bounds = getattr(env, "x_bounds", None)
    y_bounds = getattr(env, "y_bounds", None)

    if fork_type == "smc":
        temp = fork_sampling_temperature
        explore_ess = ess_fraction
        go_ess = fork_go_phase_ess_fraction

        if not fork_all_envs:
            def fork(rng, states, scores, env_steps=None, positions=None, in_explore=None):
                return smc_fork(rng, states, scores, temp, explore_ess, env_steps)

            return fork, metric_fn

        def fork(rng, states, scores, env_steps=None, positions=None, in_explore=None):
            if in_explore is None:
                raise ValueError(
                    "fork_all_envs=True requires the caller to pass `in_explore`."
                )
            explore_rng, go_rng = jax.random.split(rng)
            explore_scores = jnp.where(in_explore, scores, -jnp.inf)
            go_scores = jnp.where(in_explore, -jnp.inf, scores)

            states_after_explore, explore_mask = smc_fork(
                explore_rng, states, explore_scores, temp, explore_ess,
                env_steps, log_prefix="fork_explore",
            )
            new_states, go_mask = smc_fork(
                go_rng, states_after_explore, go_scores, temp, go_ess,
                env_steps, log_prefix="fork_go",
            )
            return new_states, explore_mask | go_mask

        return fork, metric_fn

    if fork_type == "top_k":
        if fork_top_k <= 0:
            raise ValueError(
                f"fork_top_k must be > 0 for top_k fork_type, got {fork_top_k}"
            )
        if fork_top_k > num_envs:
            raise ValueError(
                f"fork_top_k ({fork_top_k}) must be <= num_envs ({num_envs})"
            )
        k = int(fork_top_k)

        def fork(rng, states, scores, env_steps=None, positions=None, in_explore=None):
            return top_k_fork(
                rng, states, scores, k, env_steps,
                positions=positions, x_bounds=x_bounds, y_bounds=y_bounds,
            )

        return fork, metric_fn

    raise ValueError(f"Unknown fork_type: {fork_type}")
