"""PEG (Planning Goals for Exploration) components for GoExplore.

Provides:
  - ``WorldModelMLP``: Flax MLP for deterministic dynamics prediction.
  - Ensemble training: MSE loss on real transitions.
  - Disagreement reward: std across ensemble predictions.
  - MPPI goal planner: plans through the world model to maximise
    cumulative disagreement.

Reference: ``peg/dreamerv2/expl.py`` (Plan2Explore disagreement)
           ``peg/dreamerv2/goal_picker.py`` (MPPI planner)
"""

from typing import Any, Dict, List, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import variance_scaling
from flax.training.train_state import TrainState


# ── World model MLP ─────────────────────────────────────────────────────────

class WorldModelMLP(nn.Module):
    """Deterministic dynamics model: ``(state, action) -> next_state``."""

    state_size: int
    hidden_dim: int = 400
    hidden_layers: int = 3

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


# ── Ensemble training ───────────────────────────────────────────────────────

def train_world_model_ensemble(
    wm_ensemble_states: Tuple[TrainState, ...],
    wm_modules: List[WorldModelMLP],
    obs: jnp.ndarray,
    action: jnp.ndarray,
    next_obs: jnp.ndarray,
) -> Tuple[Tuple[TrainState, ...], Dict[str, jnp.ndarray]]:
    """Train each ensemble member on MSE prediction loss.

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


# ── Disagreement reward ─────────────────────────────────────────────────────

def compute_peg_explore_reward(
    wm_ensemble_states: Tuple[TrainState, ...],
    wm_modules: List[WorldModelMLP],
    obs: jnp.ndarray,
    action: jnp.ndarray,
) -> jnp.ndarray:
    """Ensemble disagreement as intrinsic reward.

    Reference: ``peg/dreamerv2/expl.py`` lines 84-85.

    Returns:
        ``(batch,)`` disagreement rewards.
    """
    sa_input = jnp.concatenate([obs, action], axis=-1)
    preds = jnp.stack([
        wm_module.apply(wm_state.params, sa_input)
        for wm_state, wm_module in zip(wm_ensemble_states, wm_modules)
    ])  # (N, batch, state_size)
    return jnp.std(preds, axis=0).mean(axis=-1)  # (batch,)


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
) -> jnp.ndarray:
    """MPPI planning to find goal maximising cumulative ensemble disagreement.

    Simulates rollouts through the world model using the GCP actor, scores
    each candidate goal by cumulative disagreement, and refines via MPPI
    softmax reweighting.

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

    Returns:
        ``(goal_dim,)`` planned goal.
    """
    goal_means = (goal_min + goal_max) / 2.0
    goal_stds = (goal_max - goal_min) / 2.0

    def eval_fitness(goals, rng):
        """Simulate rollouts, return cumulative disagreement per goal."""
        states = jnp.tile(current_state[None, :], (num_samples, 1))

        def step_fn(carry, _):
            states, total_reward, rng = carry
            rng, action_rng = jax.random.split(rng)

            # GCP actor produces actions toward candidate goals
            obs = jnp.concatenate([states, goals], axis=-1)
            actions = gcp_actor.sample_actions(actor_params, obs, action_rng, is_deterministic=False)

            # World model ensemble predictions
            sa_input = jnp.concatenate([states, actions], axis=-1)
            preds = _ensemble_predict(wm_modules, wm_params_list, sa_input)

            # Disagreement reward
            disag = jnp.std(preds, axis=0).mean(axis=-1)  # (num_samples,)
            total_reward = total_reward + disag

            # Step using ensemble mean
            next_states = jnp.mean(preds, axis=0)
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

        # MPPI reweighting
        weights = jax.nn.softmax(gamma * fitness)[:, None]  # (num_samples, 1)
        new_means = jnp.sum(weights * goals, axis=0)
        new_stds = jnp.sqrt(
            jnp.sum(weights * (goals - new_means[None, :]) ** 2, axis=0) + 1e-6
        )
        return (new_means, new_stds, rng), None

    init_carry = (goal_means, goal_stds, rng)
    (goal_means, _, _), _ = jax.lax.scan(mppi_iteration, init_carry, None, length=num_iterations)
    return goal_means
