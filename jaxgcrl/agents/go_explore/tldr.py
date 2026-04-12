"""TLDR (Temporal Distance-aware Representations) components for GoExplore.

Provides:
  - ``TrajEncoder``: Flax MLP mapping obs → latent representation φ.
  - PBE (Particle-Based Entropy) intrinsic reward via K-NN novelty.
  - Temporal distance training with dual Lagrange formulation.

Reference: ``tldr/iod/tldr.py`` (lines 390-428, 504-518, 632-644)
           ``tldr/iod/apt_utils.py`` (PBE class)
"""

from typing import Any, Dict, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState


# ── TrajEncoder network ─────────────────────────────────────────────────────

class TrajEncoder(nn.Module):
    """Gaussian MLP mapping observations to a latent representation for temporal distance.

    Matches the original ``GaussianMLPIndependentStdModuleEx`` from TLDR:
    two independent sub-networks (mean head and std head) each with
    ``hidden_layers`` hidden layers of width ``hidden_dim``.  Only the mean is
    returned from ``__call__`` because the original always accesses
    ``traj_encoder(obs).mean`` — the std head exists for architectural parity
    (same parameter count) but receives no gradient in TLDR's loss.

    ``use_layer_norm`` defaults to ``False`` to match the original default
    (``--traj_encoder_layer_normalization`` defaults to ``None`` / off).
    """

    hidden_dim: int = 1024
    hidden_layers: int = 2
    output_dim: int = 2
    use_layer_norm: bool = False

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        xavier_uniform = nn.initializers.glorot_uniform()
        bias_init = nn.initializers.zeros

        # ── Mean head ────────────────────────────────────────────────────────
        x = obs
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=xavier_uniform, bias_init=bias_init)(x)
            if self.use_layer_norm:
                x = nn.LayerNorm()(x)
            x = nn.relu(x)
        mean = nn.Dense(self.output_dim, kernel_init=xavier_uniform, bias_init=bias_init)(x)

        # ── Std head (independent hidden layers, not used in any loss) ───────
        # Matches GaussianMLPIndependentStdModuleEx where std_hidden_sizes ==
        # hidden_sizes.  Since only .mean is used downstream, these parameters
        # exist but receive zero gradient — identical to the original behaviour.
        x_std = obs
        for _ in range(self.hidden_layers):
            x_std = nn.Dense(self.hidden_dim, kernel_init=xavier_uniform, bias_init=bias_init)(x_std)
            if self.use_layer_norm:
                x_std = nn.LayerNorm()(x_std)
            x_std = nn.relu(x_std)
        nn.Dense(self.output_dim, kernel_init=xavier_uniform, bias_init=bias_init)(x_std)  # log_std (unused)

        return mean


# ── PBE (Particle-Based Entropy) ────────────────────────────────────────────

def pbe_rms_update(
    rms_state: Dict[str, jnp.ndarray],
    x: jnp.ndarray,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray]:
    """Welford's online mean/variance update.

    Args:
        rms_state: ``{"M": mean, "S": variance, "n": count}``
        x: ``(N, 1)`` new samples.

    Returns:
        ``(new_rms_state, mean)``
    """
    M, S, n = rms_state["M"], rms_state["S"], rms_state["n"]
    bs = x.shape[0]
    delta = jnp.mean(x, axis=0) - M
    new_M = M + delta * bs / (n + bs)
    new_S = (
        S * n + jnp.var(x, axis=0) * bs + jnp.square(delta) * n * bs / (n + bs)
    ) / (n + bs)
    new_n = n + bs
    return {"M": new_M, "S": new_S, "n": new_n}, new_M


def pbe_get_reward(
    source: jnp.ndarray,
    target: jnp.ndarray,
    knn_k: int,
    knn_clip: float,
    rms_state=None,
) -> Tuple[jnp.ndarray, Any]:
    """PBE K-NN novelty score.

    Args:
        source: ``(B1, D)`` query embeddings.
        target: ``(B2, D)`` reference embeddings.
        knn_k:  number of nearest neighbours.
        knn_clip: distance clipping threshold.
        rms_state: optional RMS state for normalisation.

    Returns:
        ``(rewards (B1,), new_rms_state)``
    """
    b1 = source.shape[0]
    # L2 distance matrix  (B1, B2)
    dist_matrix = jnp.sqrt(
        jnp.sum((source[:, None, :] - target[None, :, :]) ** 2, axis=-1) + 1e-8
    )
    # K smallest distances (negate → top_k finds largest → negate back)
    neg_topk, _ = jax.lax.top_k(-dist_matrix, knn_k)  # (B1, K)
    reward = -neg_topk  # (B1, K)  smallest distances

    # RMS normalisation (before clipping, matching TLDR)
    if rms_state is not None:
        reward_flat = reward.reshape(-1, 1)  # (B1*K, 1)
        rms_state, rms_mean = pbe_rms_update(rms_state, reward_flat)
        reward_flat = reward_flat / jnp.maximum(rms_mean, 1e-8)
        reward = reward_flat.reshape(b1, knn_k)

    # Clip, average, log
    reward = jnp.maximum(reward - knn_clip, 0.0)
    reward = jnp.mean(reward, axis=1)  # (B1,)
    reward = jnp.log(reward + 1.0)
    return reward, rms_state


def compute_pbe_intrinsic_reward(
    te_params,
    traj_encoder: TrajEncoder,
    obs: jnp.ndarray,
    next_obs: jnp.ndarray,
    knn_k: int,
    knn_clip: float,
    rms_state: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Full PBE intrinsic reward: ``PBE(next) - PBE(current)``.

    Returns:
        ``(rewards (B,), new_rms_state)``
    """
    rep = traj_encoder.apply(te_params, obs)
    next_rep = traj_encoder.apply(te_params, next_obs)
    target = rep  # self-similarity within the batch
    r1, rms_state = pbe_get_reward(rep, target, knn_k, knn_clip, rms_state)
    r2, rms_state = pbe_get_reward(next_rep, target, knn_k, knn_clip, rms_state)
    return r2 - r1, rms_state


# ── Temporal distance encoder training ──────────────────────────────────────

def update_traj_encoder(
    te_state: TrainState,
    dual_lam_state: TrainState,
    traj_encoder: TrajEncoder,
    obs: jnp.ndarray,
    next_obs: jnp.ndarray,
    dual_slack: float,
) -> Tuple[TrainState, TrainState, Dict[str, jnp.ndarray]]:
    """Train traj encoder (temporal distance loss) and dual lambda.

    Reference: ``tldr/iod/tldr.py`` lines 390-428 (TE loss), 632-644 (dual loss).

    Returns:
        ``(new_te_state, new_dual_lam_state, metrics_dict)``
    """
    goals = jnp.roll(next_obs, 1, axis=0)  # shifted batch as goals

    def te_loss_fn(te_params):
        phi_all = traj_encoder.apply(te_params, jnp.concatenate([obs, next_obs, goals], axis=0))
        phi_x, phi_y, phi_g = jnp.split(phi_all, 3, axis=0)

        # Temporal distance in latent space
        squared_dist = jnp.sum((phi_x - phi_g) ** 2, axis=-1)
        dist = jnp.sqrt(jnp.maximum(squared_dist, 1e-6))

        # Continuity constraint: ||φ_y - φ_x||² should be ≤ 1
        cst_penalty = 1.0 - jnp.mean((phi_y - phi_x) ** 2, axis=1)
        cst_penalty = jnp.minimum(cst_penalty, dual_slack)

        dual_lam = jnp.exp(dual_lam_state.params["log_lam"])

        # softplus(x, beta=0.01) = softplus(0.01*x) / 0.01
        softplus_term = jax.nn.softplus(0.01 * (500.0 - dist)) / 0.01
        te_obj = -jnp.mean(softplus_term) + jnp.mean(jax.lax.stop_gradient(dual_lam) * cst_penalty)
        loss_te = -te_obj
        return loss_te, cst_penalty

    (loss_te, cst_penalty), te_grads = jax.value_and_grad(te_loss_fn, has_aux=True)(te_state.params)
    new_te_state = te_state.apply_gradients(grads=te_grads)

    # Dual lambda update
    def dual_loss_fn(dual_params):
        log_lam = dual_params["log_lam"]
        return log_lam * jnp.mean(jax.lax.stop_gradient(cst_penalty))

    dual_loss, dual_grads = jax.value_and_grad(dual_loss_fn)(dual_lam_state.params)
    new_dual_lam_state = dual_lam_state.apply_gradients(grads=dual_grads)

    metrics = {
        "te_loss": loss_te,
        "dual_lam": jnp.exp(new_dual_lam_state.params["log_lam"]),
        "cst_penalty": jnp.mean(cst_penalty),
        "dual_loss": dual_loss,
    }
    return new_te_state, new_dual_lam_state, metrics


# ── PBE scoring for goal proposer ───────────────────────────────────────────

def pbe_score_candidates(
    te_params,
    traj_encoder: TrajEncoder,
    candidate_states: jnp.ndarray,
    knn_k: int,
    knn_clip: float,
) -> jnp.ndarray:
    """Score candidate states by PBE self-similarity (no RMS, used for goal selection).

    Returns:
        ``(num_candidates,)`` novelty scores (higher = more novel).
    """
    z = traj_encoder.apply(te_params, candidate_states)
    scores, _ = pbe_get_reward(z, z, knn_k, knn_clip, rms_state=None)
    return scores
