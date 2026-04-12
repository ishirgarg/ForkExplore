"""Intrinsic rewards for the explore policy in GoExplore.

Each reward type is a factory-created callable with the unified signature::

    compute_reward(state, obs, next_obs, action, key)
        -> (reward, new_state, metrics)

where:
  - ``state``     — :class:`ExploreRewardState` carrying mutable params/stats
  - ``reward``    — ``(batch_size,)`` float32, ``stop_gradient`` applied inside
  - ``new_state`` — updated :class:`ExploreRewardState`
  - ``metrics``   — dict of scalar logs (may be empty)

Use :func:`create_explore_reward_fn` to obtain the right callable by name.
The actual SAC losses are reused from ``losses.py`` via the algorithm's
``.update()`` methods — see ``update_networks`` in ``go_explore.py``.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp

from jaxgcrl.agents.go_explore.algorithms_utils import reconstruct_full_critic_params
from jaxgcrl.agents.go_explore.types import ExploreRewardState


# ── Q-uncertainty (default) ──────────────────────────────────────────────────

def compute_explore_reward(
    explore_actor,
    explore_critic,
    actor_params,
    critic_states,
    obs_state: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Intrinsic reward: ``std(Q_i(s, pi(s)))`` across the critic ensemble.

    Returns:
        ``(batch_size,)`` float32 rewards.
    """
    actions = explore_actor.sample_actions(actor_params, obs_state, key, is_deterministic=True)
    critic_params = {i: cs.params for i, cs in enumerate(critic_states)}
    full_cp = reconstruct_full_critic_params(critic_params)
    q_vals = explore_critic.apply(full_cp, obs_state, actions)  # (batch, n_critics)
    return jnp.std(q_vals, axis=-1)  # (batch,)


def _make_q_uncertainty_fn(explore_actor, explore_critic) -> Callable:
    def compute_reward(
        state: ExploreRewardState,
        obs: jnp.ndarray,
        next_obs: jnp.ndarray,
        action: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, ExploreRewardState, Dict]:
        reward = jax.lax.stop_gradient(
            compute_explore_reward(
                explore_actor, explore_critic,
                state.explore_actor_params, state.explore_critic_states,
                obs, key,
            )
        )
        return reward, state, {}

    return compute_reward


# ── TLDR / PBE ───────────────────────────────────────────────────────────────

def _make_tldr_fn(traj_encoder_module, knn_k: int, knn_clip: float, dual_slack: float) -> Callable:
    def compute_reward(
        state: ExploreRewardState,
        obs: jnp.ndarray,
        next_obs: jnp.ndarray,
        action: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, ExploreRewardState, Dict]:
        from .tldr import compute_pbe_intrinsic_reward, update_traj_encoder

        new_te_state, new_dual_lam_state, metrics = update_traj_encoder(
            state.te_state, state.dual_lam_state,
            traj_encoder_module, obs, next_obs, dual_slack,
        )
        reward, new_rms = compute_pbe_intrinsic_reward(
            new_te_state.params, traj_encoder_module,
            obs, next_obs, knn_k, knn_clip, state.pbe_rms_state,
        )
        reward = jax.lax.stop_gradient(reward)
        new_state = state.replace(
            te_state=new_te_state,
            dual_lam_state=new_dual_lam_state,
            pbe_rms_state=new_rms,
        )
        return reward, new_state, metrics

    return compute_reward


# ── PEG (ensemble disagreement) ──────────────────────────────────────────────

def _make_peg_fn(wm_modules, obs_encoder_module, use_peg_latent_space: bool) -> Callable:
    def compute_reward(
        state: ExploreRewardState,
        obs: jnp.ndarray,
        next_obs: jnp.ndarray,
        action: jnp.ndarray,
        key: jax.Array,
    ) -> Tuple[jnp.ndarray, ExploreRewardState, Dict]:
        from .peg import compute_peg_explore_reward

        enc_mod = obs_encoder_module if use_peg_latent_space else None
        enc_params = state.obs_encoder_params if use_peg_latent_space else None
        raw_reward, new_rms = compute_peg_explore_reward(
            state.wm_ensemble_states, wm_modules,
            obs, action, state.peg_rms_state,
            obs_encoder=enc_mod,
            obs_encoder_params=enc_params,
        )
        reward = jax.lax.stop_gradient(raw_reward)
        new_state = state.replace(peg_rms_state=new_rms)
        return reward, new_state, {}

    return compute_reward


# ── Public factory ───────────────────────────────────────────────────────────

def create_explore_reward_fn(
    reward_type: str,
    # q_uncertainty
    explore_actor: Optional[Any] = None,
    explore_critic: Optional[Any] = None,
    # tldr
    traj_encoder_module: Optional[Any] = None,
    knn_k: int = 5,
    knn_clip: float = 1e-3,
    dual_slack: float = 0.1,
    # peg
    wm_modules: Optional[Any] = None,
    obs_encoder_module: Optional[Any] = None,
    use_peg_latent_space: bool = False,
) -> Callable:
    """Factory: return a unified explore-reward callable closed over static modules.

    Args:
        reward_type: One of ``"q_uncertainty"``, ``"tldr"``, ``"peg"``.
        explore_actor: Actor module (q_uncertainty only).
        explore_critic: Critic module (q_uncertainty only).
        traj_encoder_module: TrajEncoder module (tldr only).
        knn_k: K for PBE k-NN (tldr only).
        knn_clip: Distance clip for PBE (tldr only).
        dual_slack: Slack for dual Lagrange (tldr only).
        wm_modules: World-model ensemble modules (peg only).
        obs_encoder_module: Observation encoder module (peg latent-space only).
        use_peg_latent_space: Whether to encode obs before disagreement (peg only).

    Returns:
        ``compute_reward(state, obs, next_obs, action, key)``
            ``-> (reward, new_state, metrics)``
    """
    if reward_type == "tldr":
        return _make_tldr_fn(traj_encoder_module, knn_k, knn_clip, dual_slack)
    elif reward_type == "peg":
        return _make_peg_fn(wm_modules, obs_encoder_module, use_peg_latent_space)
    else:  # q_uncertainty (default)
        return _make_q_uncertainty_fn(explore_actor, explore_critic)
