from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics
from jaxgcrl.agents.go_explore.algorithms_utils import reconstruct_full_critic_params
from jaxgcrl.agents.go_explore.types import GoalProposerState
from jaxgcrl.agents.go_explore.utils import geometric_sample_one_triple


def _maybe_normalize(obs: jnp.ndarray, normalizer_params: Any) -> jnp.ndarray:
    """Normalise obs with running stats if normalizer_params is set, else identity.

    Networks queried inside goal proposers were trained on normalised obs when
    GoExplore.normalize_obs is enabled, so any obs constructed from raw replay
    buffer / env data must be normalised before the actor / critic forward pass.
    """
    if normalizer_params is None:
        return obs
    return running_statistics.normalize(obs, normalizer_params)


def _maybe_normalize_state(state: jnp.ndarray, normalizer_params: Any) -> jnp.ndarray:
    """Apply the state-prefix of the (obs_size,) normaliser to a state-only tensor.

    Used by proposers that pass only the state slice (not full obs) to a network
    trained on the normalised state slice (e.g. TLDR traj encoder).
    """
    if normalizer_params is None:
        return state
    state_dim = state.shape[-1]
    mean = normalizer_params.mean[..., :state_dim]
    std  = normalizer_params.std[..., :state_dim]
    return (state - mean) / std


def create_goal_proposer(
    goal_proposer_name: str,
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
    discounting: float = 0.99,
    traj_encoder: Optional[Any] = None,
    knn_k: int = 12,
    knn_clip: float = 0.0001,
    wm_modules: Optional[Any] = None,
    mppi_horizon: int = 50,
    mppi_samples: int = 500,
    mppi_iterations: int = 5,
    mppi_gamma: float = 10.0,
    obs_encoder: Optional[Any] = None,
    obs_decoder: Optional[Any] = None,
    rssm_module: Optional[Any] = None,
    disag_module: Optional[Any] = None,
    action_size: Optional[int] = None,
) -> Callable:
    """
    Factory function to create a goal proposer function.

    Args:
        goal_proposer_name: Name of the goal proposer to create
        env: The environment instance
        num_envs: Number of parallel environments
        state_size: Size of state dimension (required for rb)
        goal_indices: Indices in state that represent the goal (required for rb)
        num_candidates: Number of candidate goals to evaluate before final selection.
        actor: Optional actor network object (for goal proposers that need to sample actions)
        critic: Optional critic network object (for goal proposers that need to compute values)
        discounting: Discount factor for geometric future-state sampling (ucgr only)
        traj_encoder: Optional TrajEncoder module (for TLDR goal proposer)
        knn_k: K nearest neighbors for PBE (for TLDR goal proposer)
        knn_clip: Distance clipping for PBE (for TLDR goal proposer)

    Returns:
        A goal proposer function that takes (rng, start_obs, goal_proposer_state) and returns (goal, updated_state).
        The goal proposer state can be read from and written to.
    """
    if goal_proposer_name == "random_env_goals":
        proposer_fn = create_random_env_goals_proposer(env, num_envs)
        # Wrap to take (rng, start_obs, goal_proposer_state) - start_obs and state ignored
        def wrapped_proposer(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
            goal = proposer_fn(rng)
            # Return empty log_data dict (no visualization for random goals)
            log_data = {}
            return goal, goal_proposer_state, log_data
        return wrapped_proposer
    elif goal_proposer_name == "rb":
        return create_rb_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic)
    elif goal_proposer_name == "ucgr":
        return create_ucgr_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic, discounting)
    elif goal_proposer_name == "q_epistemic":
        return create_q_epistemic_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic)
    elif goal_proposer_name == "max_critic_to_env":
        return create_max_critic_to_env_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, actor, critic)
    elif goal_proposer_name == "mega":
        return create_mega_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices)
    elif goal_proposer_name == "omega":
        return create_omega_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices)
    elif goal_proposer_name == "tldr":
        return create_tldr_goal_proposer(env, num_envs, num_candidates, state_size, goal_indices, traj_encoder, knn_k, knn_clip)
    elif goal_proposer_name == "peg":
        return create_peg_goal_proposer(env, num_envs, state_size, goal_indices, actor, wm_modules,
                                        mppi_horizon, mppi_samples, mppi_iterations, mppi_gamma,
                                        obs_encoder=obs_encoder, obs_decoder=obs_decoder)
    elif goal_proposer_name == "peg_rssm":
        return create_peg_rssm_goal_proposer(env, state_size, goal_indices, actor, rssm_module,
                                             disag_module, action_size,
                                             mppi_horizon, mppi_samples, mppi_iterations, mppi_gamma)
    else:
        raise ValueError(f"Unknown goal proposer: {goal_proposer_name}")


def create_random_env_goals_proposer(
    env,
    num_envs: int,
) -> Callable[[jax.Array], jnp.ndarray]:
    possible_goals = env.possible_goals  # Shape: (num_goals, goal_dim)
    num_goals = possible_goals.shape[0]  # Use .shape[0] for JIT compatibility
    
    def propose_goal(rng: jax.Array) -> jnp.ndarray:
        idx = jax.random.randint(rng, (), 0, num_goals)
        goal = possible_goals[idx]  # Shape: (goal_dim,)
        return goal
    
    return propose_goal


def create_rb_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    def propose_goal(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
        # Extract transitions_sample from goal proposer state
        transitions_sample = goal_proposer_state.transitions_sample
        
        # transitions_sample.observation shape: (num_envs, episode_length, obs_size)
        # Flatten to (num_envs * episode_length, obs_size)
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, jnp.array(goal_indices)]  # (N, goal_dim)
        
        # First select num_candidates random states, then randomly select from those
        num_states = positions.shape[0]
        rng1, rng2 = jax.random.split(rng, 2)
        candidate_indices = jax.random.randint(rng1, (num_candidates,), 0, num_states)
        candidate_positions = positions[candidate_indices]  # (num_candidates, goal_dim)
        
        # Randomly select one from candidates
        idx = jax.random.randint(rng2, (), 0, num_candidates)
        goal = candidate_positions[idx]
        
        # Return empty log_data dict (no visualization for rb proposer)
        log_data = {}
        return goal, goal_proposer_state, log_data
    
    return propose_goal


def create_ucgr_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
    discounting: float = 0.99,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """
    Create a goal proposer implementing Unsupervised Contrastive Goal-Reaching (UCGR).
 
    Strategy (MinLSE): score each candidate goal by how reachable it is, using the
    CRL critic as an implicit dynamics-aware reachability model:
 
        S(g_i) = f(s_i, a_i, g_i)
 
    where (s_i, a_i, g_i) are matched triples — g_i is geometrically sampled from
    the future state occupancy of (s_i, a_i). argmin_i S(g_i) returns the goal
    that is hardest to reach from its anchor, i.e. at the frontier of the agent's
    current capability.
 
    Memory-efficient: exactly num_candidates (env_idx, t) pairs are sampled first,
    then geometric future-state sampling is applied only to those num_candidates
    anchors via vmap — never materializing the full buffer.
 
    Args:
        env: The environment instance.
        num_envs: Number of parallel environments.
        num_candidates: Number of (s, a, g) triples to sample and score.
        state_size: Number of elements in the state portion of an observation.
        goal_indices: Indices within the state that encode the goal position.
        actor: CRL actor (unused, kept for API consistency).
        critic: CRLCritic instance — used via critic.apply(params, obs, actions).
        discounting: Discount factor γ for geometric future-state sampling.
    """
    goal_idx_array = jnp.array(goal_indices)
 
    def propose_goal(
        rng: jax.Array,
        start_obs: jnp.ndarray,
        goal_proposer_state: GoalProposerState,
    ):
        transitions_sample = goal_proposer_state.transitions_sample
        critic_params = goal_proposer_state.critic_params
        full_params = reconstruct_full_critic_params(critic_params)
 
        # transitions_sample shapes:
        #   observation: (num_envs_buf, episode_length, obs_size)
        #   action:      (num_envs_buf, episode_length, action_size)
        #   traj_id:     (num_envs_buf, episode_length)
        num_envs_buf = transitions_sample.observation.shape[0]
        episode_length = transitions_sample.observation.shape[1]
        all_traj_ids = transitions_sample.extras["state_extras"]["traj_id"]
 
        # ── 1. Sample num_candidates (env_idx, t) anchor pairs ───────────────
        # Only these num_candidates locations will ever be touched — no full
        # buffer materialisation.
        rng, env_rng, t_rng, fb_rng = jax.random.split(rng, 4)
        env_indices = jax.random.randint(env_rng, (num_candidates,), 0, num_envs_buf)
        t_indices   = jax.random.randint(t_rng,   (num_candidates,), 0, episode_length - 1)
        triple_keys = jax.random.split(fb_rng, num_candidates)
 
        # ── 2. Geometric future-state sampling — only num_candidates calls ────
        # vmap over the num_candidates anchor pairs; each call accesses one row
        # of the buffer (one env trajectory) and samples one future timestep.
        anchor_obs, anchor_acts = jax.vmap(
            lambda env_idx, t, key: geometric_sample_one_triple(
                discounting, state_size, goal_idx_array,
                transitions_sample.observation,
                transitions_sample.action,
                all_traj_ids,
                env_idx, t, key,
            )
        )(env_indices, t_indices, triple_keys)
        # anchor_obs:  (K, obs_size)   where obs = [state_t, geom-sampled goal]
        # anchor_acts: (K, action_size)
 
        candidate_goals = anchor_obs[:, state_size:]   # (K, goal_dim)
 
        # ── 3. MinLSE score for each candidate goal ───────────────────────────
        # Each triple (s_i, a_i, g_i) is already matched by geometric sampling:
        # g_i ~ p^π(sf | s_i, a_i). So f(s_i, a_i, g_i) is the critic's
        # reachability estimate for that specific pair.
        #
        # Score = critic value on the matched triple. argmin finds the goal g_i
        # that is hardest to reach from its anchor (s_i, a_i) — the frontier.
        #
        # This is a single forward pass of size K (vs the O(K²) cost of scoring
        # every g against every anchor).
        anchor_obs_in = _maybe_normalize(anchor_obs, goal_proposer_state.normalizer_params)
        q_vals = critic.apply(full_params, anchor_obs_in, anchor_acts)  # (K, n_critics)
        scores = jnp.mean(q_vals, axis=-1)                           # (K,)
 
        # ── 4. Select the hardest (lowest MinLSE score) goal ─────────────────
        best_idx = jnp.argmin(scores)
        selected_goal = candidate_goals[best_idx]  # (goal_dim,)
 
        # ── 5. Build log_data for visualization ──────────────────────────────
        first_obs_position = start_obs[:state_size][goal_idx_array]
        log_data = {
            "candidate_goals":    candidate_goals,    # (K, goal_dim)
            "first_obs_position": first_obs_position, # (goal_dim,)
            "minlse_scores":      scores,             # (K,)
            "selected_goal":      selected_goal,      # (goal_dim,)
        }
 
        return selected_goal, goal_proposer_state, log_data
 
    return propose_goal


def create_q_epistemic_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """
    Create a goal proposer that selects goals with highest Q-value variance across the ensemble.
    """
    def propose_goal(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
        # Extract required data from goal proposer state
        transitions_sample = goal_proposer_state.transitions_sample
        actor_params = goal_proposer_state.actor_params
        critic_params = goal_proposer_state.critic_params
        
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        positions = obs_flat[:, :state_size][:, jnp.array(goal_indices)]  # (N, goal_dim)
        
        # Randomly sample num_candidates goals from all states
        num_states = positions.shape[0]
        rng, sample_rng = jax.random.split(rng)
        candidate_indices = jax.random.randint(sample_rng, (num_candidates,), 0, num_states)
        candidate_goals = positions[candidate_indices]  # (num_candidates, goal_dim)
    
        s0 = start_obs[:state_size]  # Shape: (state_size,)
        goal_dim = len(goal_indices)
        
        # Reconstruct full critic params using utility function
        full_critic_params = reconstruct_full_critic_params(critic_params)
        
        norm_p = goal_proposer_state.normalizer_params

        # For each candidate goal, compute Q-value mean and std
        def compute_q_stats_for_goal(candidate_goal, rng_key):
            """Compute Q-value mean and std for a single candidate goal."""
            # Construct observation: obs = [s0, g] where s0 is from start_obs and g is candidate_goal
            # Observation structure is [state, goal], so concatenate state with candidate goal
            obs = jnp.concatenate([s0, candidate_goal], axis=-1)  # Shape: (obs_size,)
            net_obs = _maybe_normalize(obs[None, :], norm_p)  # (1, obs_size)

            # Sample action deterministically from policy
            rng_key, action_key = jax.random.split(rng_key)
            action = actor.sample_actions(
                actor_params,
                net_obs,
                action_key,
                is_deterministic=True
            )  # Shape: (1, action_size)
            action = action[0]  # Remove batch dimension: (action_size,)

            # Compute Q-values using critic
            q_values = critic.apply(
                full_critic_params,
                net_obs,
                action[None, :]  # Add batch dimension: (1, action_size)
            )  # Shape: (1, n_critics)
            q_values = q_values[0]  # Remove batch dimension: (n_critics,)
            
            # Compute mean and std across the ensemble
            q_mean = jnp.mean(q_values)
            q_std = jnp.std(q_values)
            
            return q_mean, q_std
        
        # Compute mean and std for all candidate goals
        rng, var_rng = jax.random.split(rng)
        var_keys = jax.random.split(var_rng, num_candidates)
        q_means, q_stds = jax.vmap(compute_q_stats_for_goal)(candidate_goals, var_keys)  # Both shape: (num_candidates,)
        
        # Compute variance from std for selection (variance = std^2)
        variances = q_stds ** 2
        
        # Select goal with highest variance
        best_idx = jnp.argmax(variances)
        selected_goal = candidate_goals[best_idx]  # Shape: (goal_dim,)
        
        # Prepare log_data dict with visualization data
        first_obs_position = s0[jnp.array(goal_indices)]  # Shape: (goal_dim,)
        log_data = {
            'candidate_goals': candidate_goals,        # (num_candidates, goal_dim)
            'first_obs_position': first_obs_position,  # (goal_dim,)
            'q_means': q_means,                        # (num_candidates,)
            'q_stds': q_stds,                          # (num_candidates,)
            'selected_goal': selected_goal,            # (goal_dim,)
        }
        
        return selected_goal, goal_proposer_state, log_data
    
    return propose_goal


def create_max_critic_to_env_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    actor: Optional[Any] = None,
    critic: Optional[Any] = None,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """
    Create a proposer that:
    1. Chooses a random environment goal g
    2. Samples num_candidates states w from the replay buffer
    3. For each candidate state w, computes mean Q(w, g) across the critic ensemble
    4. Selects the state w that maximizes mean Q(w, g)
    5. Returns the selected state's goal coordinates (w_goal)

    Note:
    - This is useful as a reset proposer (resetting to high-value buffer states).
    - `env_goal` is still logged in `log_data` for visualization/debugging.
    """
    possible_goals = env.possible_goals  # Shape: (num_goals, goal_dim)
    num_goals = possible_goals.shape[0]  # Use .shape[0] for JIT compatibility
    
    def propose_goal(rng: jax.Array, start_obs: jnp.ndarray, goal_proposer_state: GoalProposerState):
        # Extract required data from goal proposer state
        transitions_sample = goal_proposer_state.transitions_sample
        actor_params = goal_proposer_state.actor_params
        critic_params = goal_proposer_state.critic_params
        
        # Sample a random environment goal g
        rng, goal_rng = jax.random.split(rng)
        goal_idx = jax.random.randint(goal_rng, (), 0, num_goals)
        env_goal = possible_goals[goal_idx]  # Shape: (goal_dim,)
        
        # Sample num_candidates states w from the replay buffer
        obs_flat = jnp.reshape(transitions_sample.observation, (-1, transitions_sample.observation.shape[-1]))
        states = obs_flat[:, :state_size]  # (N, state_size)
        
        num_states = states.shape[0]
        rng, sample_rng = jax.random.split(rng)
        candidate_indices = jax.random.randint(sample_rng, (num_candidates,), 0, num_states)
        candidate_states = states[candidate_indices]  # (num_candidates, state_size)
        
        # Reconstruct full critic params using utility function
        full_critic_params = reconstruct_full_critic_params(critic_params)
        norm_p = goal_proposer_state.normalizer_params

        # For each candidate state w, compute mean Q(w, g)
        def compute_mean_q_for_state(candidate_state, rng_key):
            """Compute mean Q-value for a single candidate state w with goal g."""
            # Construct observation: obs = [w, g] where w is candidate_state and g is env_goal
            obs = jnp.concatenate([candidate_state, env_goal], axis=-1)  # Shape: (obs_size,)
            net_obs = _maybe_normalize(obs[None, :], norm_p)

            # Sample action deterministically from policy
            rng_key, action_key = jax.random.split(rng_key)
            action = actor.sample_actions(
                actor_params,
                net_obs,
                action_key,
                is_deterministic=True
            )  # Shape: (1, action_size)
            action = action[0]  # Remove batch dimension: (action_size,)

            # Compute Q-values using critic
            q_values = critic.apply(
                full_critic_params,
                net_obs,
                action[None, :]  # Add batch dimension: (1, action_size)
            )  # Shape: (1, n_critics)
            q_values = q_values[0]  # Remove batch dimension: (n_critics,)
            
            # Compute mean across the ensemble
            q_mean = jnp.mean(q_values)
            
            return q_mean
        
        # Compute mean Q for all candidate states
        rng, var_rng = jax.random.split(rng)
        var_keys = jax.random.split(var_rng, num_candidates)
        q_means = jax.vmap(compute_mean_q_for_state)(candidate_states, var_keys)  # Shape: (num_candidates,)
        
        # Select state w that maximizes mean Q(w, g)
        best_idx = jnp.argmax(q_means)
        selected_state = candidate_states[best_idx]  # Shape: (state_size,)
        selected_state_goal = selected_state[jnp.array(goal_indices)]  # Shape: (goal_dim,)
        
        # Prepare log_data dict with visualization data
        first_obs_state = start_obs[:state_size]  # Shape: (state_size,)
        first_obs_position = first_obs_state[jnp.array(goal_indices)]  # Shape: (goal_dim,)
        candidate_goals = candidate_states[:, jnp.array(goal_indices)]  # (num_candidates, goal_dim)
        
        log_data = {
            'candidate_goals': candidate_goals,        # (num_candidates, goal_dim)
            'first_obs_position': first_obs_position,  # (goal_dim,)
            'q_means': q_means,                        # (num_candidates,)
            'selected_goal': env_goal,                 # (goal_dim,) - the random environment goal
            'selected_state_goal': selected_state_goal, # (goal_dim,) - goal coordinates of maximizing state
        }
        
        return selected_state_goal, goal_proposer_state, log_data
    
    return propose_goal



# ─────────────────────────────────────────────────────────────────────────────
# JAX Gaussian KDE helper (shared by MEGA and OMEGA)
# ─────────────────────────────────────────────────────────────────────────────

def _jax_gaussian_kde(
    query_points: jnp.ndarray,
    data_points: jnp.ndarray,
    bandwidth: float = 0.1,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Vectorised Gaussian KDE density estimate, fully JAX-compatible.

    Normalises ``data_points`` to zero mean / unit std per dimension before
    computing the kernel, replicating the sklearn default when applied to
    pre-normalised inputs (bandwidth=0.1).

    Args:
        query_points: (M, D) - points at which to evaluate the density.
        data_points:  (N, D) - reference samples that define the distribution.
        bandwidth:    Gaussian kernel bandwidth (applied after normalisation).
        eps:          Small constant for numerical stability in std normalisation.

    Returns:
        (M,) unnormalised density values (proportional to true KDE density).
    """
    # Per-dimension normalisation using buffer statistics
    data_mean = jnp.mean(data_points, axis=0, keepdims=True)       # (1, D)
    data_std  = jnp.std(data_points,  axis=0, keepdims=True) + eps  # (1, D)

    query_norm = (query_points - data_mean) / data_std  # (M, D)
    data_norm  = (data_points  - data_mean) / data_std  # (N, D)

    # Pairwise squared Euclidean distances in normalised space: (M, N)
    diffs    = query_norm[:, None, :] - data_norm[None, :, :]  # (M, N, D)
    sq_dists = jnp.sum(diffs ** 2, axis=-1)                    # (M, N)

    # Average Gaussian kernel value
    density = jnp.mean(jnp.exp(-0.5 * sq_dists / (bandwidth ** 2)), axis=-1)
    return density  # (M,)


# ─────────────────────────────────────────────────────────────────────────────
# MEGA: MaxEnt Goal Achievement
# ─────────────────────────────────────────────────────────────────────────────

def create_mega_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    kde_bandwidth: float = 0.1,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """Create a MEGA (Maximum Entropy Goal Achievement) goal proposer.

    Implements the minimum-density heuristic from:
      Pitis et al., "Maximum Entropy Gain Exploration for Long Horizon
      Multi-goal Reinforcement Learning", ICML 2020.

    At each call the proposer:
      1. Samples ``num_candidates`` past achieved goals from the replay buffer.
      2. Fits a Gaussian KDE to ALL achieved goals in the current buffer sample.
      3. Returns the candidate whose KDE density is lowest — i.e. the most
         sparsely explored region of the achieved-goal space (the frontier).

    The density estimates are returned in ``log_data`` so they can be colour-
    coded in the visualisation.

    Args:
        env:            Environment instance (unused beyond type consistency).
        num_envs:       Number of parallel environments.
        num_candidates: Number of candidate goals to evaluate (paper uses 100).
        state_size:     Number of elements in the state portion of an observation.
        goal_indices:   Indices within the state that encode the goal.
        kde_bandwidth:  Gaussian kernel bandwidth (applied after normalisation).
    """
    goal_idx_array = jnp.array(goal_indices)

    def propose_goal(
        rng: jax.Array,
        start_obs: jnp.ndarray,
        goal_proposer_state: GoalProposerState,
    ):
        transitions_sample = goal_proposer_state.transitions_sample

        # ── 1. Collect all achieved goals from the buffer sample ──────────────
        # shape: (num_envs_buf * episode_length, obs_size)
        obs_flat = jnp.reshape(
            transitions_sample.observation,
            (-1, transitions_sample.observation.shape[-1]),
        )
        all_goals = obs_flat[:, :state_size][:, goal_idx_array]  # (N_buf, goal_dim)

        # ── 2. Sample num_candidates candidates ───────────────────────────────
        n_buf = all_goals.shape[0]
        rng, sample_rng = jax.random.split(rng)
        cand_indices = jax.random.randint(sample_rng, (num_candidates,), 0, n_buf)
        candidate_goals = all_goals[cand_indices]  # (num_candidates, goal_dim)

        # ── 3. KDE density for each candidate ─────────────────────────────────
        densities = _jax_gaussian_kde(candidate_goals, all_goals, kde_bandwidth)
        # (num_candidates,) — lower density ↔ more sparsely explored

        # ── 4. Select the minimum-density candidate (frontier goal) ──────────
        best_idx     = jnp.argmin(densities)
        selected_goal = candidate_goals[best_idx]  # (goal_dim,)

        # ── 5. Build log_data for visualisation ──────────────────────────────
        first_obs_position = start_obs[:state_size][goal_idx_array]
        log_data = {
            "candidate_goals":    candidate_goals,    # (num_candidates, goal_dim)
            "densities":          densities,          # (num_candidates,)
            "first_obs_position": first_obs_position, # (goal_dim,)
            "selected_goal":      selected_goal,      # (goal_dim,)
        }

        return selected_goal, goal_proposer_state, log_data

    return propose_goal


# ─────────────────────────────────────────────────────────────────────────────
# OMEGA: annealed MEGA → desired-goal objective
# ─────────────────────────────────────────────────────────────────────────────

def create_omega_goal_proposer(
    env,
    num_envs: int,
    num_candidates: int,
    state_size: Optional[int] = None,
    goal_indices: Optional[tuple] = None,
    kde_bandwidth: float = 0.1,
    omega_bias: float = -3.0,
    n_desired_eval: int = 100,
) -> Callable[[jax.Array, jnp.ndarray, GoalProposerState], tuple]:
    """Create an OMEGA goal proposer.

    OMEGA (Optimised MEGA) anneals the MEGA intrinsic objective into the
    original supervised objective once the agent's achieved-goal distribution
    starts covering the desired-goal distribution.

    Concretely, at each call:
      * α  = 1 / max(b + DKL(p_dg ‖ p_ag), 1)   where b = ``omega_bias``
      * With probability α  → return a random desired (environment) goal.
      * With probability 1-α → return the MEGA minimum-density goal.

    DKL(p_dg ‖ p_ag) is estimated by evaluating the KDE density p_ag at a
    random sample of desired goals (uniform over ``env.possible_goals``):
        DKL ≈ mean_{g ~ p_dg}[ -log p_ag(g) ]
    (the log(N_goals) constant term cancels in the α formula up to the bias).

    The scalar α and the estimated KL divergence are logged in ``log_data``
    alongside the standard MEGA candidate visualisation data.

    Args:
        env:             Environment (must expose ``env.possible_goals``).
        num_envs:        Number of parallel environments.
        num_candidates:  Number of candidate goals to evaluate.
        state_size:      State portion size in observations.
        goal_indices:    Indices encoding the goal within the state.
        kde_bandwidth:   Gaussian KDE bandwidth (post-normalisation).
        omega_bias:      Bias b in the α formula; paper uses b = -3.
        n_desired_eval:  How many desired goals to sample when estimating DKL.
    """
    goal_idx_array  = jnp.array(goal_indices)
    possible_goals  = jnp.array(env.possible_goals)  # (N_goals, goal_dim)
    n_possible      = possible_goals.shape[0]

    def propose_goal(
        rng: jax.Array,
        start_obs: jnp.ndarray,
        goal_proposer_state: GoalProposerState,
    ):
        transitions_sample = goal_proposer_state.transitions_sample

        # ── 1. Collect all achieved goals from the buffer sample ──────────────
        obs_flat  = jnp.reshape(
            transitions_sample.observation,
            (-1, transitions_sample.observation.shape[-1]),
        )
        all_goals = obs_flat[:, :state_size][:, goal_idx_array]  # (N_buf, goal_dim)

        # ── 2. Sample candidates ───────────────────────────────────────────────
        n_buf     = all_goals.shape[0]
        rng, sample_rng, desired_rng, env_rng, alpha_rng = jax.random.split(rng, 5)

        cand_indices   = jax.random.randint(sample_rng, (num_candidates,), 0, n_buf)
        candidate_goals = all_goals[cand_indices]  # (num_candidates, goal_dim)

        # ── 3. KDE density for each candidate (for MEGA selection) ────────────
        densities = _jax_gaussian_kde(candidate_goals, all_goals, kde_bandwidth)
        # (num_candidates,) — lower = frontier

        # ── 4. MEGA goal: minimum-density candidate ───────────────────────────
        best_idx   = jnp.argmin(densities)
        mega_goal  = candidate_goals[best_idx]  # (goal_dim,)

        # ── 5. Compute α via DKL(p_dg ‖ p_ag) ────────────────────────────────
        # Sample n_desired_eval goals from the desired distribution
        desired_idx    = jax.random.randint(desired_rng, (n_desired_eval,), 0, n_possible)
        desired_sample = possible_goals[desired_idx]   # (n_desired_eval, goal_dim)

        # Evaluate p_ag density at desired goals
        pag_at_desired = _jax_gaussian_kde(desired_sample, all_goals, kde_bandwidth)
        pag_at_desired = jnp.clip(pag_at_desired, 1e-10, None)  # avoid log(0)

        # KL divergence: DKL(p_dg ‖ p_ag) ≈ mean_g[-log p_ag(g)] for uniform p_dg
        # (the -log(N_goals) term from the uniform prior is absorbed into omega_bias)
        kl_div = jnp.mean(-jnp.log(pag_at_desired))  # scalar

        # α = 1 / max(b + KL, 1)
        alpha = 1.0 / jnp.maximum(omega_bias + kl_div, 1.0)

        # ── 6. Sample an environment (desired) goal ────────────────────────────
        env_goal_idx = jax.random.randint(env_rng, (), 0, n_possible)
        env_goal     = possible_goals[env_goal_idx]  # (goal_dim,)

        # ── 7. Mix: with probability α use env goal, else MEGA goal ──────────
        use_env_goal  = jax.random.uniform(alpha_rng) < alpha
        selected_goal = jax.lax.cond(
            use_env_goal,
            lambda: env_goal,
            lambda: mega_goal,
        )

        # ── 8. Build log_data ─────────────────────────────────────────────────
        first_obs_position = start_obs[:state_size][goal_idx_array]
        log_data = {
            "candidate_goals":    candidate_goals,    # (num_candidates, goal_dim)
            "densities":          densities,          # (num_candidates,)
            "first_obs_position": first_obs_position, # (goal_dim,)
            "selected_goal":      selected_goal,      # (goal_dim,)
            "mega_goal":          mega_goal,          # (goal_dim,)  — always the MEGA choice
            "env_goal":           env_goal,           # (goal_dim,)  — always the env choice
            "alpha":              alpha,              # scalar
            "kl_div":             kl_div,             # scalar
        }

        return selected_goal, goal_proposer_state, log_data

    return propose_goal


# ── TLDR goal proposer ──────────────────────────────────────────────────────

def create_tldr_goal_proposer(
    env, num_envs, num_candidates, state_size, goal_indices, traj_encoder, knn_k, knn_clip,
):
    """Select goals by PBE novelty in the traj encoder's latent space.

    Samples candidate states from the replay buffer, encodes them via the
    trajectory encoder, scores each by K-NN self-similarity (PBE), and
    selects the most novel candidate as the goal.

    Reference: ``tldr/iod/tldr.py`` ``get_random_goals()`` lines 265-300.
    """
    from .tldr import pbe_score_candidates

    goal_idx_array = jnp.array(goal_indices)

    def propose_goal(rng, start_obs, goal_proposer_state):
        transitions_sample = goal_proposer_state.transitions_sample
        te_params = goal_proposer_state.te_params

        # Flatten buffer next_observations → (N, obs_size).
        # Original TLDR get_random_goals encodes next_obs, not obs.
        next_obs_flat = jnp.reshape(
            transitions_sample.next_observation,
            (-1, transitions_sample.next_observation.shape[-1]),
        )
        n_buf = next_obs_flat.shape[0]

        # Sample num_candidates random indices from buffer
        rng, sample_rng = jax.random.split(rng)
        cand_indices = jax.random.randint(sample_rng, (num_candidates,), 0, n_buf)
        candidate_obs = next_obs_flat[cand_indices]  # (num_candidates, obs_size)
        candidate_states = candidate_obs[:, :state_size]  # raw state only
        # The traj encoder is trained on normalised state slices in update_networks,
        # so feed it normalised states here too.
        candidate_states_in = _maybe_normalize_state(
            candidate_states, goal_proposer_state.normalizer_params,
        )

        # Score candidates by PBE novelty in latent space
        scores = pbe_score_candidates(
            te_params, traj_encoder, candidate_states_in, knn_k, knn_clip,
        )

        # Select most novel candidate
        best_idx = jnp.argmax(scores)
        candidate_goals = candidate_states[:, goal_idx_array]  # (num_candidates, goal_dim)
        selected_goal = candidate_goals[best_idx]

        first_obs_position = start_obs[:state_size][goal_idx_array]
        log_data = {
            "candidate_goals": candidate_goals,
            "pbe_scores": scores,
            "first_obs_position": first_obs_position,
            "selected_goal": selected_goal,
        }
        return selected_goal, goal_proposer_state, log_data

    return propose_goal


# ── PEG goal proposer (MPPI planning through world model) ───────────────────

def create_peg_goal_proposer(
    env, num_envs, state_size, goal_indices, gcp_actor, wm_modules,
    mppi_horizon, mppi_samples, mppi_iterations, mppi_gamma,
    obs_encoder=None, obs_decoder=None,
):
    """Plan goals via MPPI through the world model to maximise disagreement.

    When ``obs_encoder`` and ``obs_decoder`` are provided, the MPPI rollout
    operates in latent space (latent-space PEG).

    Reference: ``peg/dreamerv2/goal_picker.py`` SubgoalPlanner.
    """
    from .peg import mppi_plan_goal

    goal_idx_array = jnp.array(goal_indices)
    goal_dim = len(goal_indices)
    goal_min = jnp.concatenate([env.x_bounds[:1], env.y_bounds[:1]])
    goal_max = jnp.concatenate([env.x_bounds[1:], env.y_bounds[1:]])

    def propose_goal(rng, start_obs, goal_proposer_state):
        actor_params = goal_proposer_state.actor_params
        wm_params_list = goal_proposer_state.wm_ensemble_params
        enc_params = goal_proposer_state.obs_encoder_params
        dec_params = goal_proposer_state.obs_decoder_params
        current_state = start_obs[:state_size]

        # Seed MPPI from the agent's current goal-space position, matching the
        # original PEG get_distribution_from_obs initialisation.
        current_goal_pos = current_state[goal_idx_array]  # (goal_dim,)

        planned_goal = mppi_plan_goal(
            gcp_actor, actor_params,
            wm_modules, wm_params_list,
            current_state, goal_dim, goal_min, goal_max,
            mppi_samples, mppi_horizon, mppi_iterations, mppi_gamma,
            rng,
            init_means=current_goal_pos,
            obs_encoder=obs_encoder,
            obs_encoder_params=enc_params,
            obs_decoder=obs_decoder,
            obs_decoder_params=dec_params,
            normalizer_params=goal_proposer_state.normalizer_params,
        )

        log_data = {
            "first_obs_position": current_state[goal_idx_array],
            "selected_goal": planned_goal,
        }
        return planned_goal, goal_proposer_state, log_data

    return propose_goal


# ── PEG-RSSM goal proposer (MPPI planning through DreamerV2 RSSM) ────────────

def create_peg_rssm_goal_proposer(
    env,
    state_size: int,
    goal_indices,
    gcp_actor,
    rssm_module,
    disag_module,
    action_size: int,
    mppi_horizon: int = 50,
    mppi_samples: int = 500,
    mppi_iterations: int = 5,
    mppi_gamma: float = 10.0,
):
    """Plan goals via MPPI through the DreamerV2 RSSM imagination.

    Uses the RSSM's learned world model to imagine future trajectories and
    scores them via the disagreement ensemble (Plan2Explore-style reward).

    Requires ``goal_proposer_state.rssm_params`` and
    ``goal_proposer_state.disag_params`` to be set (non-None).

    Reference: ``peg/dreamerv2/goal_picker.py`` SubgoalPlanner + expl.py Plan2Explore.
    """
    from .rssm import mppi_plan_goal_rssm

    goal_idx_array = jnp.array(goal_indices)
    goal_dim = len(goal_indices)
    goal_min = jnp.concatenate([env.x_bounds[:1], env.y_bounds[:1]])
    goal_max = jnp.concatenate([env.x_bounds[1:], env.y_bounds[1:]])

    def propose_goal(rng, start_obs, goal_proposer_state):
        actor_params = goal_proposer_state.actor_params
        rssm_params = goal_proposer_state.rssm_params
        disag_params = goal_proposer_state.disag_params
        current_state = start_obs[:state_size]

        planned_goal = mppi_plan_goal_rssm(
            gcp_actor, actor_params,
            rssm_params, disag_params,
            rssm_module, disag_module,
            current_state, action_size, goal_dim, goal_min, goal_max,
            mppi_samples, mppi_horizon, mppi_iterations, mppi_gamma,
            rng,
            normalizer_params=goal_proposer_state.normalizer_params,
        )

        log_data = {
            "first_obs_position": current_state[goal_idx_array],
            "selected_goal": planned_goal,
        }
        return planned_goal, goal_proposer_state, log_data

    return propose_goal