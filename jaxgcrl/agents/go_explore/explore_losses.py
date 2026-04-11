"""Intrinsic reward for the explore policy in GoExplore.

The explore policy is non-goal-conditioned (obs = raw state) and trained only
on explore-phase transitions via sample_weights (phase mask).  The intrinsic
reward is ``std(Q_explore(s, pi_explore(s)))`` across the explore critic ensemble.

The actual SAC losses are reused from ``losses.py`` via the algorithm's
``.update()`` methods — see ``update_networks`` in ``go_explore.py``.
"""

import jax
import jax.numpy as jnp

from jaxgcrl.agents.go_explore.algorithms_utils import reconstruct_full_critic_params


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
