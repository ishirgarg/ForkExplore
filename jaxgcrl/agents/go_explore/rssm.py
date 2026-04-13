"""DreamerV2-style Recurrent State Space Model (RSSM) for PEG.

Matches the original Plan2Explore implementation exactly:
  - GRUCell: Dense(3*size, no bias) + LayerNorm + sigmoid/tanh gates, update_bias=-1.0
  - RSSM: ensemble=1, stoch=50, deter=200, hidden=200, std_act=sigmoid2, min_std=0.1
  - Encoder/Decoder: 3-layer MLP with ELU (peg_walker: mlp_layers=[400,400,400])
  - DisagreementEnsemble: 10 MLP heads (4×400 ELU) → stoch predictions
  - Reward: std across heads, mean across stoch dims
  - KL: balanced (balance=0.8, free=1.0, free_avg=True, forward=False)
  - model_opt/expl_opt: {lr=3e-4, eps=1e-5, clip=100, wd=1e-6}

Reference:
  peg/dreamerv2/common/nets.py  — EnsembleRSSM, GRUCell, MLP
  peg/dreamerv2/expl.py         — Plan2Explore, _intr_reward, _train_ensemble
  peg/dreamerv2/configs.yaml    — peg_walker config
"""

from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState


# ── GRU Cell ─────────────────────────────────────────────────────────────────

class GRUCell(nn.Module):
    """DreamerV2 GRU cell with multiplicative reset gating and LayerNorm.

    Original (nets.py GRUCell, norm=True):
      - Dense(3*size, use_bias=False) + LayerNorm → split(reset, cand, update)
      - cand  = tanh(reset * cand)
      - update = sigmoid(update + update_bias)   [update_bias = -1.0]
      - out  = update * cand + (1 - update) * state
    """

    size: int
    update_bias: float = -1.0

    def setup(self):
        self.gates = nn.Dense(3 * self.size, use_bias=False)
        self.layer_norm = nn.LayerNorm()

    def __call__(self, x: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        inputs = jnp.concatenate([x, state], axis=-1)
        parts = self.gates(inputs)
        # LayerNorm in float32 (matching DreamerV2 mixed-precision handling)
        parts = self.layer_norm(parts.astype(jnp.float32)).astype(parts.dtype)
        reset, cand, update = jnp.split(parts, 3, axis=-1)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)          # GRU uses tanh, not elu
        update = jax.nn.sigmoid(update + self.update_bias)
        return update * cand + (1 - update) * state


# ── RSSM ─────────────────────────────────────────────────────────────────────

class RSSM(nn.Module):
    """Recurrent State Space Model (EnsembleRSSM with ensemble=1).

    States dict keys: stoch, deter, mean, std, prior_mean, prior_std.
    Feature = concat(stoch, deter), dim = stoch + deter.

    Prior (img_step):
      concat(stoch, action) → Dense(hidden) → elu → GRU(deter)
      → Dense(hidden) → elu → Dense(2*stoch) → sigmoid2+min_std

    Posterior (obs_step, on top of img_step):
      concat(deter, embed) → Dense(hidden) → elu → Dense(2*stoch) → sigmoid2+min_std
    """

    stoch: int = 50
    deter: int = 200
    hidden: int = 200
    min_std: float = 0.1

    def setup(self):
        # img_step layers
        self.img_in = nn.Dense(self.hidden)         # concat(stoch,action) → hidden
        self.gru = GRUCell(self.deter)               # hidden → deter
        self.img_out = nn.Dense(self.hidden)         # deter → hidden (prior stats)
        self.img_dist = nn.Dense(2 * self.stoch)     # hidden → 2*stoch (prior mean+std)
        # obs_step layer (posterior stats from deter+embed)
        self.obs_out = nn.Dense(self.hidden)         # concat(deter,embed) → hidden
        self.obs_dist = nn.Dense(2 * self.stoch)     # hidden → 2*stoch (post mean+std)

    def _sigmoid2_std(self, raw: jnp.ndarray) -> jnp.ndarray:
        """std = 2*sigmoid(raw/2) + min_std  (sigmoid2 activation)."""
        return 2.0 * jax.nn.sigmoid(raw / 2.0) + self.min_std

    def _prior_stats(self, deter: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Prior: deter → Dense(hidden) → elu → Dense(2*stoch) → (mean, std)."""
        h = nn.elu(self.img_out(deter))
        raw = self.img_dist(h)
        mean, std_raw = jnp.split(raw, 2, axis=-1)
        return mean, self._sigmoid2_std(std_raw)

    def _post_stats(self, deter: jnp.ndarray, embed: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Posterior: concat(deter,embed) → Dense(hidden) → elu → Dense(2*stoch) → (mean, std)."""
        x = jnp.concatenate([deter, embed], axis=-1)
        h = nn.elu(self.obs_out(x))
        raw = self.obs_dist(h)
        mean, std_raw = jnp.split(raw, 2, axis=-1)
        return mean, self._sigmoid2_std(std_raw)

    def img_step(self, prev_state: Dict, prev_action: jnp.ndarray) -> Tuple:
        """Prior imagination step.

        Returns:
            (new_deter, prior_mean, prior_std)
        """
        x = jnp.concatenate([prev_state["stoch"], prev_action], axis=-1)
        x = nn.elu(self.img_in(x))
        new_deter = self.gru(x, prev_state["deter"])
        prior_mean, prior_std = self._prior_stats(new_deter)
        return new_deter, prior_mean, prior_std

    def get_feat(self, state: Dict) -> jnp.ndarray:
        """Feature = concat(stoch, deter)."""
        return jnp.concatenate([state["stoch"], state["deter"]], axis=-1)

    def observe(self, embed: jnp.ndarray, action: jnp.ndarray, key: jax.Array) -> Dict:
        """Posterior sequence over T steps (Python for-loop, traced by JAX).

        Convention:  obs_step(s_{t-1}, a_{t-1}, embed_t) → s_t
        At t=0: zero initial state and zero action.

        Args:
            embed:  (B, T, embed_dim)  encoded observations
            action: (B, T, action_dim) actions taken at each step

        Returns:
            Dict with keys stoch, deter, mean, std, prior_mean, prior_std
            — each (B, T, ...).

        Note: uses a Python for-loop so JAX traces T copies of the operations.
        This is necessary because jax.lax.scan has tracer-leak issues when
        Flax sub-modules (Dense, LayerNorm) are called inside it.  The for-loop
        is fully equivalent and safe.
        """
        B, T, _ = embed.shape
        stoch = jnp.zeros((B, self.stoch))
        deter = jnp.zeros((B, self.deter))

        # Shift action right: at step t we use action[t-1] (a_{t-1})
        zero_a = jnp.zeros_like(action[:, :1, :])
        shifted_action = jnp.concatenate([zero_a, action[:, :-1, :]], axis=1)

        keys = jax.random.split(key, T)
        outputs = []

        for t in range(T):
            emb = embed[:, t, :]                      # (B, E)
            act = shifted_action[:, t, :]             # (B, A)
            k   = keys[t]                             # scalar key

            # img_step: prior
            x = jnp.concatenate([stoch, act], axis=-1)
            x = nn.elu(self.img_in(x))
            deter = self.gru(x, deter)
            prior_mean, prior_std = self._prior_stats(deter)

            # obs_step: posterior
            post_mean, post_std = self._post_stats(deter, emb)
            stoch = post_mean + post_std * jax.random.normal(k, post_mean.shape)

            outputs.append({
                "stoch": stoch,
                "deter": deter,
                "mean": post_mean,
                "std": post_std,
                "prior_mean": prior_mean,
                "prior_std": prior_std,
            })

        # Stack T outputs along axis 1: each value becomes (B, T, dim)
        return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=1), *outputs)

    def kl_loss(
        self,
        states: Dict,
        free: float = 1.0,
        balance: float = 0.8,
        free_avg: bool = True,
    ) -> jnp.ndarray:
        """Balanced KL with free bits (peg_walker: free=1.0, balance=0.8, free_avg=True).

        With forward=False:
          mix = 1 - balance = 0.2
          loss = mix * max(mean(KL(post || sg(prior))), free)
               + (1-mix) * max(mean(KL(sg(post) || prior)), free)
        """

        def kl_normal(m1, s1, m2, s2):
            """KL(N(m1,s1) || N(m2,s2)), summed over last dim."""
            return jnp.sum(
                jnp.log(s2) - jnp.log(s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2.0 * s2 ** 2) - 0.5,
                axis=-1,
            )

        sg = jax.lax.stop_gradient
        pm, ps = states["mean"], states["std"]
        qm, qs = states["prior_mean"], states["prior_std"]

        # lhs: KL(post || sg(prior))  — drives post toward prior
        lhs = kl_normal(pm, ps, sg(qm), sg(qs))
        # rhs: KL(sg(post) || prior)  — drives prior toward post
        rhs = kl_normal(sg(pm), sg(ps), qm, qs)

        mix = 1.0 - balance  # 0.2
        if free_avg:
            loss = mix * jnp.maximum(lhs.mean(), free) + (1.0 - mix) * jnp.maximum(rhs.mean(), free)
        else:
            loss = mix * jnp.maximum(lhs, free).mean() + (1.0 - mix) * jnp.maximum(rhs, free).mean()

        return loss


# ── World model (encoder + RSSM + decoder) ───────────────────────────────────

class RSSMWorldModel(nn.Module):
    """Combined RSSM + MLP encoder + MLP decoder.

    peg_walker:
      encoder: mlp_layers=[400,400,400]  → embed_dim = 400
      decoder: mlp_layers=[400,400,400]  → obs_dim
      rssm:    {stoch=50, deter=200, hidden=200, min_std=0.1}
    """

    obs_dim: int
    stoch: int = 50
    deter: int = 200
    hidden: int = 200
    min_std: float = 0.1
    n_enc_layers: int = 3
    enc_units: int = 400      # both encoder and decoder use this width

    def setup(self):
        self.enc_layers = [nn.Dense(self.enc_units) for _ in range(self.n_enc_layers)]
        self.dec_layers = [nn.Dense(self.enc_units) for _ in range(self.n_enc_layers)]
        self.dec_out = nn.Dense(self.obs_dim)
        self.rssm = RSSM(stoch=self.stoch, deter=self.deter, hidden=self.hidden, min_std=self.min_std)

    def encode(self, obs: jnp.ndarray) -> jnp.ndarray:
        """Encode obs: (..., obs_dim) → (..., enc_units)."""
        x = obs
        for layer in self.enc_layers:
            x = nn.elu(layer(x))
        return x

    def decode(self, feat: jnp.ndarray) -> jnp.ndarray:
        """Decode feat: (..., feat_dim) → (..., obs_dim)."""
        x = feat
        for layer in self.dec_layers:
            x = nn.elu(layer(x))
        return self.dec_out(x)

    def get_feat(self, states: Dict) -> jnp.ndarray:
        return self.rssm.get_feat(states)

    def observe(self, obs: jnp.ndarray, action: jnp.ndarray, key: jax.Array) -> Dict:
        """Encode obs then run RSSM observe."""
        embed = self.encode(obs)
        return self.rssm.observe(embed, action, key)

    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray, key: jax.Array) -> Tuple:
        """Full training call: returns (states, recon_obs).

        Exercises encoder, RSSM, and decoder so all sub-module parameters
        are created during ``init``.
        """
        states = self.observe(obs, action, key)
        feat = self.get_feat(states)
        recon = self.decode(feat)
        return states, recon


# ── Disagreement ensemble ─────────────────────────────────────────────────────

class _DisagHead(nn.Module):
    """Single disagreement head: n_layers×Dense(n_units)+ELU → Dense(stoch_size)."""

    stoch_size: int
    n_layers: int
    n_units: int

    def setup(self):
        self.hidden = [nn.Dense(self.n_units) for _ in range(self.n_layers)]
        self.out = nn.Dense(self.stoch_size)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden:
            x = nn.elu(layer(x))
        return self.out(x)


class DisagreementEnsemble(nn.Module):
    """n_heads independent MLP heads predicting stoch[t+1] from (feat[t], action[t]).

    peg_walker: expl_head={layers:4, units:400}, disag_models=10, disag_target=stoch (size 50).

    All heads are trained jointly via a single TrainState.  Flax names them
    heads_0 … heads_{n_heads-1}.

    Reward = preds.std(axis=heads).mean(axis=stoch_dim)  (matches original _intr_reward).
    """

    n_heads: int = 10
    stoch_size: int = 50
    n_layers: int = 4
    n_units: int = 400

    def setup(self):
        self.heads = [
            _DisagHead(stoch_size=self.stoch_size, n_layers=self.n_layers, n_units=self.n_units)
            for _ in range(self.n_heads)
        ]

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Forward all heads.

        Args:
            x: (..., input_dim)

        Returns:
            preds: (n_heads, ..., stoch_size)
        """
        return jnp.stack([head(x) for head in self.heads], axis=0)


# ── Optimizer factory ─────────────────────────────────────────────────────────

def _make_rssm_optimizer(lr: float = 3e-4, eps: float = 1e-5, clip: float = 100.0, wd: float = 1e-6) -> optax.GradientTransformation:
    """model_opt / expl_opt from original PEG: {lr, eps, clip, wd}."""
    return optax.chain(
        optax.clip_by_global_norm(clip),
        optax.adam(learning_rate=lr, eps=eps),
        optax.add_decayed_weights(wd),
    )


def init_rssm_states(
    rssm_module: RSSMWorldModel,
    disag_module: DisagreementEnsemble,
    obs_dim: int,
    action_dim: int,
    key: jax.Array,
    lr: float = 3e-4,
    eps: float = 1e-5,
    clip: float = 100.0,
    wd: float = 1e-6,
) -> Tuple[TrainState, TrainState]:
    """Initialise TrainStates for the RSSM world model and disagreement ensemble.

    Returns:
        (rssm_state, disag_state)
    """
    key, rssm_key, disag_key = jax.random.split(key, 3)

    feat_dim = rssm_module.stoch + rssm_module.deter
    dummy_obs = jnp.ones([1, 2, obs_dim])
    dummy_action = jnp.ones([1, 2, action_dim])
    rssm_params = rssm_module.init(rssm_key, dummy_obs, dummy_action, rssm_key)

    dummy_disag_input = jnp.ones([1, feat_dim + action_dim])
    disag_params = disag_module.init(disag_key, dummy_disag_input)

    tx = _make_rssm_optimizer(lr, eps, clip, wd)
    rssm_state = TrainState.create(apply_fn=rssm_module.apply, params=rssm_params, tx=tx)
    disag_state = TrainState.create(apply_fn=disag_module.apply, params=disag_params, tx=tx)

    return rssm_state, disag_state


# ── Training step ─────────────────────────────────────────────────────────────

def train_rssm_step(
    rssm_state: TrainState,
    disag_state: TrainState,
    rssm_module: RSSMWorldModel,
    disag_module: DisagreementEnsemble,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    key: jax.Array,
    kl_scale: float = 1.0,
    kl_free: float = 1.0,
    kl_balance: float = 0.8,
) -> Tuple[TrainState, TrainState, jnp.ndarray, Dict]:
    """Train RSSMWorldModel + DisagreementEnsemble on a batch of trajectories.

    Args:
        rssm_state: current RSSM TrainState.
        disag_state: current DisagreementEnsemble TrainState.
        rssm_module: RSSMWorldModel module.
        disag_module: DisagreementEnsemble module.
        obs:    (B, T, obs_dim) raw state observations.
        action: (B, T, action_dim) actions.
        key: PRNG key.
        kl_scale: weight for KL loss (default 1.0).
        kl_free:  free-bits threshold (peg_walker: 1.0).
        kl_balance: KL balance (peg_walker: 0.8).

    Returns:
        (new_rssm_state, new_disag_state, disagree_rewards, metrics)
        disagree_rewards: (B, T) — last step padded with 0.
    """
    B, T, _ = obs.shape
    key, rssm_key, disag_key = jax.random.split(key, 3)

    # ── RSSM world model update ────────────────────────────────────────────────
    def rssm_loss_fn(params):
        # __call__ returns (states, recon) — exercises encode, rssm, decode
        states, recon = rssm_module.apply(params, obs, action, rssm_key)
        feat = jnp.concatenate([states["stoch"], states["deter"]], axis=-1)

        # Reconstruction: MSE between decoded feat and original obs
        recon_loss = jnp.mean((recon - jax.lax.stop_gradient(obs)) ** 2)

        # Balanced KL with free bits
        pm, ps = states["mean"], states["std"]
        qm, qs = states["prior_mean"], states["prior_std"]
        sg = jax.lax.stop_gradient

        def _kl(m1, s1, m2, s2):
            return jnp.sum(
                jnp.log(s2) - jnp.log(s1) + (s1 ** 2 + (m1 - m2) ** 2) / (2.0 * s2 ** 2) - 0.5,
                axis=-1,
            )

        lhs = _kl(pm, ps, sg(qm), sg(qs))
        rhs = _kl(sg(pm), sg(ps), qm, qs)
        mix = 1.0 - kl_balance  # 0.2
        kl_loss = (
            mix * jnp.maximum(lhs.mean(), kl_free)
            + (1.0 - mix) * jnp.maximum(rhs.mean(), kl_free)
        )

        total = recon_loss + kl_scale * kl_loss
        return total, (states, feat, recon_loss, kl_loss)

    (_, (states, feat, recon_loss, kl_loss)), rssm_grads = jax.value_and_grad(
        rssm_loss_fn, has_aux=True
    )(rssm_state.params)
    new_rssm_state = rssm_state.apply_gradients(grads=rssm_grads)

    # ── DisagreementEnsemble update ───────────────────────────────────────────
    # Input:  concat(feat[0..T-2], action[0..T-2])
    # Target: stoch[1..T]  (stop_gradient: don't backprop into RSSM)
    feat_sg = jax.lax.stop_gradient(feat)
    stoch_sg = jax.lax.stop_gradient(states["stoch"])

    disag_input = jnp.concatenate([feat_sg[:, :-1, :], action[:, :-1, :]], axis=-1)  # (B, T-1, F+A)
    disag_target = stoch_sg[:, 1:, :]                                                 # (B, T-1, stoch)

    disag_input_flat = disag_input.reshape(B * (T - 1), -1)
    disag_target_flat = disag_target.reshape(B * (T - 1), -1)

    def disag_loss_fn(params):
        preds = disag_module.apply(params, disag_input_flat)  # (n_heads, B*(T-1), stoch)
        target = jax.lax.stop_gradient(disag_target_flat[None, :, :])  # (1, B*(T-1), stoch)
        return jnp.mean((preds - target) ** 2)

    disag_loss, disag_grads = jax.value_and_grad(disag_loss_fn)(disag_state.params)
    new_disag_state = disag_state.apply_gradients(grads=disag_grads)

    # ── Compute disagreement reward ────────────────────────────────────────────
    # Use the NEW disag params (same as original which trains then evaluates)
    preds = disag_module.apply(new_disag_state.params, disag_input_flat)  # (n_heads, B*(T-1), stoch)
    disag_flat = jnp.std(preds, axis=0).mean(axis=-1)  # (B*(T-1),)
    disag = disag_flat.reshape(B, T - 1)               # (B, T-1)

    # Pad last step with 0 to get (B, T)
    rewards = jnp.concatenate([disag, jnp.zeros((B, 1))], axis=1)

    metrics = {
        "rssm_recon_loss": recon_loss,
        "rssm_kl_loss": kl_loss,
        "rssm_disag_loss": disag_loss,
        "rssm_disag_reward_mean": disag.mean(),
    }
    return new_rssm_state, new_disag_state, rewards, metrics


# ── Single-step reward computation (for flat batches) ─────────────────────────

def compute_rssm_disagree_reward(
    rssm_state: TrainState,
    disag_state: TrainState,
    rssm_module: RSSMWorldModel,
    disag_module: DisagreementEnsemble,
    obs: jnp.ndarray,
    action: jnp.ndarray,
    key: jax.Array,
) -> jnp.ndarray:
    """Disagreement reward for a flat batch using single-step RSSM approximation.

    Encodes obs[t], runs one-step posterior (deter=0 initial), then passes
    (feat, action) to the disagreement ensemble.

    Args:
        obs:    (B, obs_dim) current observations.
        action: (B, action_dim) actions.
        key: PRNG key.

    Returns:
        rewards: (B,)
    """
    # Single-step: treat each obs as a length-1 sequence with zero init state
    obs_1 = obs[:, None, :]       # (B, 1, obs_dim)
    action_1 = action[:, None, :] # (B, 1, action_dim)

    states, _ = rssm_module.apply(rssm_state.params, obs_1, action_1, key)
    feat = jnp.concatenate([states["stoch"][:, 0, :], states["deter"][:, 0, :]], axis=-1)  # (B, F)

    disag_input = jnp.concatenate([feat, action], axis=-1)  # (B, F+A)
    preds = disag_module.apply(disag_state.params, disag_input)   # (n_heads, B, stoch)
    reward = jnp.std(preds, axis=0).mean(axis=-1)                 # (B,)
    return reward


# ── MPPI planner using RSSM imagination ──────────────────────────────────────

def mppi_plan_goal_rssm(
    gcp_actor,
    actor_params,
    rssm_params: Any,
    disag_params: Any,
    rssm_module: RSSMWorldModel,
    disag_module: DisagreementEnsemble,
    current_obs: jnp.ndarray,
    action_dim: int,
    goal_dim: int,
    goal_min: jnp.ndarray,
    goal_max: jnp.ndarray,
    num_samples: int,
    horizon: int,
    num_iterations: int,
    gamma: float,
    key: jax.Array,
    normalizer_params: Optional[Any] = None,
) -> jnp.ndarray:
    """MPPI planning through RSSM imagination to maximise disagreement.

    Encodes current_obs once, then imagines rollouts by advancing the RSSM
    deterministic state with img_step (no observations) and scoring with
    the disagreement ensemble.

    Args:
        current_obs: (obs_dim,) current environment observation.
        action_dim: action space dimensionality (for zero-action init).
        goal_dim: dimensionality of the goal space.
        goal_min, goal_max: (goal_dim,) bounds for MPPI sampling.
        num_samples: candidate goals per iteration.
        horizon: imagination rollout length.
        num_iterations: MPPI refinement steps.
        gamma: MPPI temperature.
        key: PRNG key.

    Returns:
        (goal_dim,) planned goal.
    """

    # Pre-compute state-prefix and goal-suffix slices of the obs normaliser so
    # the RSSM (trained on normalised state slices) and the GCP actor (trained
    # on normalised [state, goal]) both see consistent inputs.
    state_size_local = current_obs.shape[-1]
    if normalizer_params is not None:
        state_mean = normalizer_params.mean[:state_size_local]
        state_std  = normalizer_params.std[:state_size_local]
        goal_mean  = normalizer_params.mean[state_size_local:]
        goal_std   = normalizer_params.std[state_size_local:]
        current_obs = (current_obs - state_mean) / state_std
    else:
        goal_mean = goal_std = None

    # Encode current obs → initial RSSM state (posterior, zero prior)
    obs_1 = current_obs[None, None, :]                # (1, 1, obs_dim)
    action_zero = jnp.zeros((1, 1, action_dim))       # (1, 1, action_dim)

    # Build initial RSSM state via single obs_step with zero action/state
    key, init_key = jax.random.split(key)
    init_states, _ = rssm_module.apply(rssm_params, obs_1, action_zero, init_key)
    init_stoch = init_states["stoch"][0, 0, :]   # (stoch,)
    init_deter = init_states["deter"][0, 0, :]   # (deter,)

    goal_means = (goal_min + goal_max) / 2.0
    goal_stds = (goal_max - goal_min) / 2.0

    feat_size = rssm_module.stoch + rssm_module.deter

    def eval_fitness(goals: jnp.ndarray, rng: jax.Array) -> jnp.ndarray:
        """Imagination rollout: (num_samples, goal_dim) → (num_samples,) cumulative reward."""
        # Tile initial RSSM state for all candidates
        stochs = jnp.tile(init_stoch[None, :], (num_samples, 1))
        deters = jnp.tile(init_deter[None, :], (num_samples, 1))
        # Goals are MPPI-sampled in raw physical space (between goal_min/goal_max);
        # normalise them once for the actor input which expects normalised obs.
        if goal_mean is not None:
            goals_for_actor = (goals - goal_mean) / goal_std
        else:
            goals_for_actor = goals

        def step_fn(carry, _):
            stoch, deter, total, rng = carry
            rng, action_rng, sample_rng = jax.random.split(rng, 3)

            # GCP actor: concat(decoded_obs_approx, goal)
            feat = jnp.concatenate([stoch, deter], axis=-1)  # (S, F)
            # Decode feat → obs for actor input. RSSM was trained on normalised
            # obs (when normalize_obs is on) so obs_hat is already in that space.
            obs_hat = rssm_module.apply(rssm_params, feat, method=RSSMWorldModel.decode)  # (S, obs_dim)
            actor_input = jnp.concatenate([obs_hat, goals_for_actor], axis=-1)
            actions = gcp_actor.sample_actions(
                actor_params, actor_input, action_rng, is_deterministic=False
            )  # (S, action_dim)

            # Disagree reward in feature space
            disag_in = jnp.concatenate([feat, actions], axis=-1)       # (S, F+A)
            preds = disag_module.apply(disag_params, disag_in)          # (n_heads, S, stoch)
            disag = jnp.std(preds, axis=0).mean(axis=-1)               # (S,)
            total = total + disag

            # Advance RSSM via img_step (no new observation)
            new_deter, pm, ps = rssm_module.apply(
                rssm_params, {"stoch": stoch, "deter": deter}, actions,
                method=lambda m, prev, act: m.rssm.img_step(prev, act),
            )
            new_stoch = pm + ps * jax.random.normal(sample_rng, pm.shape)
            return (new_stoch, new_deter, total, rng), None

        rng, rollout_rng = jax.random.split(rng)
        (_, _, total, _), _ = jax.lax.scan(
            step_fn, (stochs, deters, jnp.zeros(num_samples), rollout_rng), None, length=horizon
        )
        return total

    def mppi_iter(carry, _):
        means, stds, rng = carry
        rng, sample_rng, eval_rng = jax.random.split(rng, 3)
        goals = means[None, :] + stds[None, :] * jax.random.normal(sample_rng, (num_samples, goal_dim))
        goals = jnp.clip(goals, goal_min, goal_max)
        fitness = eval_fitness(goals, eval_rng)
        # Shift by max for numerical stability (log-sum-exp trick)
        fitness_shifted = gamma * (fitness - jnp.max(fitness))
        weights = jax.nn.softmax(fitness_shifted)[:, None]
        new_means = jnp.sum(weights * goals, axis=0)
        new_stds = jnp.sqrt(jnp.sum(weights * (goals - new_means) ** 2, axis=0) + 1e-6)
        return (new_means, new_stds, rng), None

    key, mppi_key = jax.random.split(key)
    (goal_means, _, _), _ = jax.lax.scan(
        mppi_iter, (goal_means, goal_stds, mppi_key), None, length=num_iterations
    )
    return goal_means
