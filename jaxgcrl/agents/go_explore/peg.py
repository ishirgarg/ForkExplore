"""PEG (Planning Goals for Exploration) components for GoExplore.

Provides:
  - ``WorldModelMLP``: Flax MLP for deterministic dynamics prediction.
  - ``ObsEncoder`` / ``ObsDecoder``: Optional encoder-decoder for latent-space PEG.
  - Ensemble training: MSE loss on real transitions (obs-space or latent-space).
  - Disagreement reward: std across ensemble predictions.
  - MPPI goal planner: plans through the world model to maximise
    cumulative disagreement.

Reference: ``peg/dreamerv2/expl.py`` (Plan2Explore disagreement)
           ``peg/dreamerv2/goal_picker.py`` (MPPI planner)
"""

from typing import Any, Dict, List, Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling

from jaxgcrl.agents.go_explore.algorithms_utils import welford_update
from flax.training.train_state import TrainState


# ── Encoder / Decoder (latent-space PEG) ───────────────────────────────────

class ObsEncoder(nn.Module):
    """Maps raw observations to a latent representation.

    LayerNorm on the output prevents representation collapse when the
    downstream world model is trained jointly.
    """

    latent_dim: int = 64
    hidden_dim: int = 256
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = obs
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = nn.elu(x)
        x = nn.Dense(self.latent_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        x = nn.LayerNorm()(x)  # prevent collapse
        return x


class ObsDecoder(nn.Module):
    """Maps latent representation back to observation space."""

    obs_dim: int
    hidden_dim: int = 256
    hidden_layers: int = 2

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = z
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = nn.elu(x)
        x = nn.Dense(self.obs_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


# ── World model MLP ─────────────────────────────────────────────────────────

class WorldModelMLP(nn.Module):
    """Deterministic dynamics model: ``(state, action) -> next_state``.

    When used with latent-space PEG, ``state_size`` is the latent dim and
    inputs/outputs are latent vectors rather than raw observations.
    """

    state_size: int
    hidden_dim: int = 400
    hidden_layers: int = 4

    @nn.compact
    def __call__(self, sa_input: jnp.ndarray) -> jnp.ndarray:
        lecun_uniform = variance_scaling(1 / 3, "fan_in", "uniform")
        bias_init = nn.initializers.zeros
        x = sa_input
        for _ in range(self.hidden_layers):
            x = nn.Dense(self.hidden_dim, kernel_init=lecun_uniform, bias_init=bias_init)(x)
            x = nn.elu(x)
        x = nn.Dense(self.state_size, kernel_init=lecun_uniform, bias_init=bias_init)(x)
        return x


# ── Ensemble training (obs-space) ──────────────────────────────────────────

def train_world_model_ensemble(
    wm_ensemble_states: Tuple[TrainState, ...],
    wm_modules: List[WorldModelMLP],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
) -> Tuple[Tuple[TrainState, ...], Dict[str, jnp.ndarray]]:
    """Train each ensemble member on MSE prediction loss (obs-space).

    Args:
        wm_ensemble_states: tuple of TrainStates, one per member.
        wm_modules: list of WorldModelMLP modules.
        obs: ``(batch, state_size)`` current states.
        action: ``(batch, action_size)`` actions taken.
        next_obs: ``(batch, state_size)`` ground-truth next states.

    Returns:
        ``(new_wm_ensemble_states, metrics)``
    """
    sa_input = jnp.concatenate([obs, action], axis=-1)
    target = jax.lax.stop_gradient(next_obs)

    new_states = []
    losses = []
    for wm_state, wm_module in zip(wm_ensemble_states, wm_modules):

        def loss_fn(params):
            pred = wm_module.apply(params, sa_input)
            return jnp.mean((pred - target) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(wm_state.params)
        new_states.append(wm_state.apply_gradients(grads=grads))
        losses.append(loss)

    metrics = {"wm_loss": jnp.mean(jnp.array(losses))}
    return tuple(new_states), metrics


# ── Encoder-decoder + ensemble joint training (latent-space) ───────────────

def train_encoder_decoder_and_ensemble(
    enc_state: TrainState,
    dec_state: TrainState,
    wm_ensemble_states: Tuple[TrainState, ...],
    obs_encoder: ObsEncoder,
    obs_decoder: ObsDecoder,
    wm_modules: List[WorldModelMLP],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
    recon_coef: float = 1.0,
) -> Tuple[TrainState, TrainState, Tuple[TrainState, ...], Dict[str, jnp.ndarray]]:
    """Joint training of encoder, decoder, and world-model ensemble.

    Three separate gradient passes:
      1. **Encoder**: prediction loss (all ensemble members, gradients through
         ``z = enc(obs)``) + reconstruction loss.
      2. **Decoder**: reconstruction loss with stop-gradient encoder.
      3. **Each WM member**: prediction MSE with stop-gradient encoder.

    Args:
        enc_state: encoder TrainState.
        dec_state: decoder TrainState.
        wm_ensemble_states: tuple of WM TrainStates.
        obs_encoder: ObsEncoder module.
        obs_decoder: ObsDecoder module.
        wm_modules: list of WorldModelMLP modules.
        obs: ``(batch, obs_dim)`` observations.
        action: ``(batch, action_dim)`` actions.
        next_obs: ``(batch, obs_dim)`` next observations.
        recon_coef: weight for reconstruction loss relative to prediction loss.

    Returns:
        ``(new_enc_state, new_dec_state, new_wm_ensemble_states, metrics)``
    """
    # Pre-compute stop-gradient targets for decoder and WM updates
    z_next_fixed = jax.lax.stop_gradient(obs_encoder.apply(enc_state.params, next_obs))

    # ── Pass 1: update encoder (prediction + reconstruction) ─────────────────
    def enc_loss_fn(enc_params):
        z_obs = obs_encoder.apply(enc_params, obs)
        z_next_tgt = jax.lax.stop_gradient(obs_encoder.apply(enc_params, next_obs))

        # Prediction loss: average over ensemble members (z_obs has gradient)
        sa_z = jnp.concatenate([z_obs, action], axis=-1)
        pred_loss = jnp.mean(jnp.array([
            jnp.mean((wm.apply(wm_state.params, sa_z) - z_next_tgt) ** 2)
            for wm, wm_state in zip(wm_modules, wm_ensemble_states)
        ]))

        # Reconstruction loss
        recon = obs_decoder.apply(jax.lax.stop_gradient(dec_state.params), z_obs)
        recon_loss = jnp.mean((recon - obs) ** 2)

        return pred_loss + recon_coef * recon_loss, (pred_loss, recon_loss)

    (_, (pred_loss, recon_loss)), enc_grads = jax.value_and_grad(
        enc_loss_fn, has_aux=True
    )(enc_state.params)
    new_enc_state = enc_state.apply_gradients(grads=enc_grads)

    # ── Pass 2: update decoder (reconstruction with stop-gradient encoder) ───
    z_obs_fixed = jax.lax.stop_gradient(obs_encoder.apply(new_enc_state.params, obs))

    def dec_loss_fn(dec_params):
        recon = obs_decoder.apply(dec_params, z_obs_fixed)
        return jnp.mean((recon - obs) ** 2)

    dec_loss, dec_grads = jax.value_and_grad(dec_loss_fn)(dec_state.params)
    new_dec_state = dec_state.apply_gradients(grads=dec_grads)

    # ── Pass 3: update each WM member (prediction with stop-gradient encoder) ─
    sa_z_fixed = jnp.concatenate([z_obs_fixed, action], axis=-1)
    target_z = z_next_fixed

    new_wm_states = []
    wm_losses = []
    for wm_state, wm_module in zip(wm_ensemble_states, wm_modules):

        def wm_loss_fn(params):
            pred = wm_module.apply(params, sa_z_fixed)
            return jnp.mean((pred - target_z) ** 2)

        wm_loss, wm_grads = jax.value_and_grad(wm_loss_fn)(wm_state.params)
        new_wm_states.append(wm_state.apply_gradients(grads=wm_grads))
        wm_losses.append(wm_loss)

    metrics = {
        "wm_loss": jnp.mean(jnp.array(wm_losses)),
        "enc_pred_loss": pred_loss,
        "enc_recon_loss": recon_loss,
        "dec_recon_loss": dec_loss,
    }
    return new_enc_state, new_dec_state, tuple(new_wm_states), metrics


# ── Disagreement reward ─────────────────────────────────────────────────────

def compute_peg_explore_reward(
    wm_ensemble_states: Tuple[TrainState, ...],
    wm_modules: List[WorldModelMLP],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    rms_state: Any = None,
    obs_encoder: Optional[ObsEncoder] = None,
    obs_encoder_params: Optional[Any] = None,
) -> Tuple[jnp.ndarray, Any]:
    """Ensemble disagreement as intrinsic reward, normalised by running mean.

    Reference: ``peg/dreamerv2/expl.py`` lines 84-88.

    When ``obs_encoder`` is provided the disagreement is computed in the
    encoder's latent space (latent-space PEG).

    Args:
        rms_state: optional ``{"M", "S", "n"}`` running-stats dict.  When
            provided the raw disagreement is divided by its running mean,
            matching the ``intr_rewnorm`` normalisation in the original PEG.
        obs_encoder: optional encoder module; if given, obs are encoded before
            the ensemble forward pass.
        obs_encoder_params: encoder params (required when obs_encoder is given).

    Returns:
        ``((batch,) rewards, updated_rms_state)``
    """
    if obs_encoder is not None:
        obs_z = obs_encoder.apply(obs_encoder_params, obs)
        sa_input = jnp.concatenate([obs_z, action], axis=-1)
    else:
        sa_input = jnp.concatenate([obs, action], axis=-1)

    preds = jnp.stack([
        wm_module.apply(wm_state.params, sa_input)
        for wm_state, wm_module in zip(wm_ensemble_states, wm_modules)
    ])  # (N, batch, state_size)
    disag = jnp.std(preds, axis=0).mean(axis=-1)  # (batch,)

    if rms_state is not None:
        rms_state, rms_mean = welford_update(rms_state, disag[:, None])
        disag = disag / jnp.maximum(rms_mean[0], 1e-8)

    return disag, rms_state


# ── MPPI goal planner ───────────────────────────────────────────────────────

def _ensemble_predict(wm_modules, wm_params_list, sa_input):
    """Forward all ensemble members, return stacked predictions."""
    return jnp.stack([
        wm.apply(p, sa_input) for wm, p in zip(wm_modules, wm_params_list)
    ])  # (N, batch, state_size)


def mppi_plan_goal(
    gcp_actor,
    actor_params,
    wm_modules: List[WorldModelMLP],
    wm_params_list: Tuple[Any, ...],
    current_state: jnp.ndarray,
    goal_dim: int,
    goal_min: jnp.ndarray,
    goal_max: jnp.ndarray,
    num_samples: int,
    horizon: int,
    num_iterations: int,
    gamma: float,
    rng: jax.Array,
    init_means: jnp.ndarray = None,
    obs_encoder: Optional[ObsEncoder] = None,
    obs_encoder_params: Optional[Any] = None,
    obs_decoder: Optional[ObsDecoder] = None,
    obs_decoder_params: Optional[Any] = None,
) -> jnp.ndarray:
    """MPPI planning to find goal maximising cumulative ensemble disagreement.

    Simulates rollouts through the world model using the GCP actor, scores
    each candidate goal by cumulative disagreement, and refines via MPPI
    softmax reweighting.

    When ``obs_encoder`` / ``obs_decoder`` are provided the rollout operates
    entirely in latent space: the current state is encoded once, the world
    model advances the latent state, and the decoder reconstructs an
    observation at each step for the GCP actor input.

    Reference: ``peg/dreamerv2/goal_picker.py`` lines 188-307.

    Args:
        gcp_actor: goal-conditioned actor module.
        actor_params: current actor parameters.
        wm_modules: list of WorldModelMLP modules.
        wm_params_list: tuple of params, one per module.
        current_state: ``(state_size,)`` current env state.
        goal_dim: dimensionality of goal space.
        goal_min: ``(goal_dim,)`` lower bounds for goals.
        goal_max: ``(goal_dim,)`` upper bounds for goals.
        num_samples: candidates per MPPI iteration.
        horizon: rollout length.
        num_iterations: MPPI refinement iterations.
        gamma: MPPI temperature.
        rng: PRNG key.
        init_means: ``(goal_dim,)`` initial MPPI mean; defaults to the
            midpoint of [goal_min, goal_max] when None.  Pass the current
            agent's goal-space position to seed the planner near the agent,
            matching the original PEG ``get_distribution_from_obs`` init.
        obs_encoder: optional encoder module (latent-space rollout).
        obs_encoder_params: encoder params (required when obs_encoder given).
        obs_decoder: optional decoder module (latent-space rollout).
        obs_decoder_params: decoder params (required when obs_decoder given).

    Returns:
        ``(goal_dim,)`` planned goal.
    """
    goal_means = (goal_min + goal_max) / 2.0 if init_means is None else init_means
    goal_stds = (goal_max - goal_min) / 2.0

    use_latent = obs_encoder is not None

    if use_latent:
        # Encode current state once; latent rollout thereafter
        current_z = obs_encoder.apply(obs_encoder_params, current_state[None, :])[0]

        def eval_fitness(goals, rng):
            """Simulate latent-space rollouts, return cumulative disagreement."""
            zs = jnp.tile(current_z[None, :], (num_samples, 1))

            def step_fn(carry, _):
                zs, total_reward, rng = carry
                rng, action_rng = jax.random.split(rng)

                # Decode latent state for actor input
                obs_hat = obs_decoder.apply(obs_decoder_params, zs)  # (num_samples, obs_dim)

                # GCP actor: concat decoded obs with candidate goal
                actor_input = jnp.concatenate([obs_hat, goals], axis=-1)
                actions = gcp_actor.sample_actions(
                    actor_params, actor_input, action_rng, is_deterministic=False
                )

                # Ensemble disagreement in latent space
                sa_z = jnp.concatenate([zs, actions], axis=-1)
                preds = _ensemble_predict(wm_modules, wm_params_list, sa_z)  # (N, S, latent)
                disag = jnp.std(preds, axis=0).mean(axis=-1)  # (S,)
                total_reward = total_reward + disag

                # Advance latent state using ensemble mean
                next_zs = jnp.mean(preds, axis=0)
                return (next_zs, total_reward, rng), None

            init_carry = (zs, jnp.zeros(num_samples), rng)
            (_, total_reward, _), _ = jax.lax.scan(step_fn, init_carry, None, length=horizon)
            return total_reward

    else:
        def eval_fitness(goals, rng):
            """Simulate obs-space rollouts, return cumulative disagreement."""
            states = jnp.tile(current_state[None, :], (num_samples, 1))

            def step_fn(carry, _):
                states, total_reward, rng = carry
                rng, action_rng = jax.random.split(rng)

                # GCP actor produces actions toward candidate goals
                obs = jnp.concatenate([states, goals], axis=-1)
                actions = gcp_actor.sample_actions(
                    actor_params, obs, action_rng, is_deterministic=False
                )

                # World model ensemble predictions
                sa_input = jnp.concatenate([states, actions], axis=-1)
                preds = _ensemble_predict(wm_modules, wm_params_list, sa_input)

                # Disagreement reward
                disag = jnp.std(preds, axis=0).mean(axis=-1)  # (num_samples,)
                total_reward = total_reward + disag

                # Step using ensemble mean; clip to prevent divergence with untrained WM
                next_states = jnp.clip(jnp.mean(preds, axis=0), -1e3, 1e3)
                return (next_states, total_reward, rng), None

            init_carry = (states, jnp.zeros(num_samples), rng)
            (_, total_reward, _), _ = jax.lax.scan(step_fn, init_carry, None, length=horizon)
            return total_reward

    def mppi_iteration(carry, _):
        goal_means, goal_stds, rng = carry
        rng, sample_rng, eval_rng = jax.random.split(rng, 3)

        # Sample candidate goals
        goals = goal_means[None, :] + goal_stds[None, :] * jax.random.normal(
            sample_rng, (num_samples, goal_dim)
        )
        goals = jnp.clip(goals, goal_min[None, :], goal_max[None, :])

        # Evaluate fitness
        fitness = eval_fitness(goals, eval_rng)

        # MPPI reweighting — shift by max for numerical stability (log-sum-exp trick)
        # prevents exp() overflow when fitness values are large (e.g. early untrained WM)
        fitness_shifted = gamma * (fitness - jnp.max(fitness))
        weights = jax.nn.softmax(fitness_shifted)[:, None]  # (num_samples, 1)
        new_means = jnp.sum(weights * goals, axis=0)
        new_stds = jnp.sqrt(
            jnp.sum(weights * (goals - new_means[None, :]) ** 2, axis=0) + 1e-6
        )
        return (new_means, new_stds, rng), None

    init_carry = (goal_means, goal_stds, rng)
    (goal_means, _, _), _ = jax.lax.scan(mppi_iteration, init_carry, None, length=num_iterations)
    return goal_means


# ── PBE scoring for goal proposer ───────────────────────────────────────────
# (retained for any non-PEG callers that may import from this module)
