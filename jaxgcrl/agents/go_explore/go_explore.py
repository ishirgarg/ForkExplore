"""Go Explore agent.

Two-phase training loop:
  - Go phase   (phase == 0): GCP navigates to a proposed frontier goal (stochastic).
  - Explore phase (phase == 1): optional separate non-goal-conditioned explore policy,
    or reuse GCP with higher eps-random.  Intrinsic reward: ``std(V(s))``.

Optional features: reset proposer, SMC fork redistribution (explore-phase only).
Phase management is handled by ``GoExploreWrapper`` (see ``jaxgcrl/envs/wrappers.py``).
"""

import logging
import random
import time
from typing import Callable, Literal, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax import base, envs
from brax.training import types
from brax.v1 import envs as envs_v1
from flax.struct import dataclass
from flax.training.train_state import TrainState

from jaxgcrl.envs.wrappers import (
    EvalAutoResetWrapper,
    GoExploreWrapper,
    TrajectoryIdWrapper,
    EpisodeWrapper,
    VmapWrapper,
)
from jaxgcrl.utils.evaluator import ActorEvaluator
from jaxgcrl.utils.replay_buffer import TrajectoryUniformSamplingQueue
from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization

from .types import TrainingState, Transition, GoalProposerState, ExploreRewardState
from .algorithms import get_algorithm
from .utils import (
    save_params,
    create_single_dummy_transition,
    create_dummy_transition_for_buffer,
    create_dummy_transition_for_goal_proposer,
)
from .losses import update_alpha_sac
from .visualization import all_visualizations, visualize_go_explore_phases
from .goal_proposers import (
    create_goal_proposer,
    create_random_env_goals_proposer,
)
from .fork_algo import create_fork_fn
from .explore_losses import create_explore_reward_fn

Metrics = types.Metrics
Env = Union[envs.Env, envs_v1.Env, envs_v1.Wrapper]
State = Union[envs.State, envs_v1.State]


@dataclass
class GoExplore:
    """Go Explore agent with two-phase training.

    Go phase (phase 0): GCP navigates to a proposed frontier goal (stochastic).
    Explore phase (phase 1): either a separate non-goal-conditioned policy or
    the same GCP with higher eps-random probability.

    Phase management is handled by ``GoExploreWrapper`` (see ``jaxgcrl/envs/wrappers.py``).
    """

    # Algorithm type for the goal-conditioned policy
    agent_type: Literal["sac", "crl"] = "crl"

    policy_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    batch_size: int = 128

    discounting: float = 0.99
    logsumexp_penalty_coeff: float = 0.1

    train_step_multiplier: int = 1
    disable_entropy_actor: bool = False

    max_replay_size: int = 30000
    min_replay_size: int = 1000
    unroll_length: int = 50
    h_dim: int = 256
    n_hidden: int = 4
    skip_connections: int = 4
    use_relu: bool = False

    repr_dim: int = 64
    use_ln: bool = True

    contrastive_loss_fn: Literal["fwd_infonce", "sym_infonce", "bwd_infonce", "binary_nce"] = "fwd_infonce"
    energy_fn: Literal["norm", "l2", "dot", "cosine"] = "norm"

    tau: float = 0.005
    n_critics: int = 1
    use_her: bool = True

    goal_proposer_name: Literal[
        "random_env_goals", "rb", "q_epistemic", "ucgr",
        "max_critic_to_env", "mega", "omega", "tldr", "peg", "peg_rssm",
    ] = "random_env_goals"
    num_candidates: int = 256

    # ── Go Explore specific parameters ──────────────────────────────────────
    num_gcp_steps: int = -1      # max steps in go phase before forcing explore
    num_ep_steps: int = -1       # steps in explore phase before reset

    # ── Explore policy (None = reuse Go policy) ─────────────────────────────
    explore_policy_type: Optional[Literal["sac"]] = None  # "sac" supported; extend as needed
    explore_n_critics: int = 2
    explore_policy_lr: float = 3e-4
    explore_critic_lr: float = 3e-4
    explore_alpha_lr: float = 3e-4
    explore_grad_clip: float = 100.0   # gradient norm clip for explore policy (original PEG expl_opt.clip: 100)
    explore_eps_random_action: float = 0.2  # eps_random when explore_policy_type is None

    # ── Explore reward type ────────────────────────────────────────────────
    explore_reward_type: Literal["q_uncertainty", "tldr", "peg"] = "q_uncertainty"

    # ── TLDR hyperparameters (used when explore_reward_type="tldr") ────────
    te_hidden_dim: int = 1024       # traj encoder hidden layer width
    te_hidden_layers: int = 2       # traj encoder hidden layer count
    te_output_dim: int = 2          # traj encoder latent dimension
    te_layer_norm: bool = False     # traj encoder layer normalisation (default off, matching TLDR)
    te_lr: float = 1e-4             # traj encoder learning rate
    dual_lam_init: float = 30.0     # initial dual Lagrange multiplier
    dual_slack: float = 1e-3        # constraint slack
    dual_lr: float = 1e-4           # dual lambda learning rate
    knn_k: int = 12                 # K nearest neighbors for PBE
    knn_clip: float = 0.0001        # distance clipping for PBE

    # ── PEG hyperparameters (used when explore_reward_type="peg" or goal_proposer_name="peg")
    wm_ensemble_size: int = 10      # world model ensemble members (original PEG: 10)
    wm_hidden_dim: int = 400        # world model MLP hidden width
    wm_hidden_layers: int = 4       # world model MLP hidden depth
    wm_lr: float = 3e-4             # world model learning rate (original PEG: 3e-4)
    wm_eps: float = 1e-5            # Adam epsilon for world model (original PEG: 1e-5)
    wm_grad_clip: float = 100.0     # gradient norm clip for world model (original PEG: 100)
    wm_wd: float = 1e-6             # weight decay for world model (original PEG: 1e-6)
    mppi_horizon: int = 50          # MPPI planning horizon (original PEG: 50)
    mppi_samples: int = 500         # MPPI samples per iteration (original PEG: 500)
    mppi_iterations: int = 5        # MPPI optimization steps
    mppi_gamma: float = 10.0        # MPPI temperature
    # ── PEG latent-space (encoder-decoder) ──────────────────────────────────
    use_peg_latent_space: bool = False  # enable encoder-decoder latent-space PEG
    wm_latent_dim: int = 64             # encoder output / WM state dimension
    enc_hidden_dim: int = 256           # encoder/decoder MLP hidden width
    enc_hidden_layers: int = 2          # encoder/decoder MLP hidden depth
    enc_lr: float = 3e-4                # encoder learning rate
    dec_lr: float = 3e-4                # decoder learning rate
    enc_recon_coef: float = 1.0         # reconstruction loss weight

    # ── RSSM (DreamerV2-style, replaces WorldModelMLP when use_rssm=True) ───
    use_rssm: bool = False              # use DreamerV2 RSSM instead of WorldModelMLP
    rssm_stoch: int = 50               # stochastic state dimension (peg_walker: 50)
    rssm_deter: int = 200              # GRU deterministic state dimension (peg_walker: 200)
    rssm_hidden: int = 200             # RSSM MLP hidden width (peg_walker: 200)
    rssm_min_std: float = 0.1          # minimum std for stochastic state
    rssm_enc_layers: int = 3           # encoder/decoder MLP depth (peg_walker: 3)
    rssm_enc_units: int = 400          # encoder/decoder MLP width (peg_walker: 400)
    rssm_kl_scale: float = 1.0         # KL loss weight
    rssm_kl_free: float = 1.0          # free-bits threshold (peg_walker: 1.0)
    rssm_kl_balance: float = 0.8       # KL balance (peg_walker: 0.8)
    disag_heads: int = 10              # disagreement ensemble size (peg_walker: 10)
    disag_layers: int = 4              # disagreement head MLP depth (peg_walker: 4)
    disag_units: int = 400             # disagreement head MLP width (peg_walker: 400)
    rssm_lr: float = 3e-4             # model + expl optimizer LR (peg_walker: 3e-4)
    rssm_eps: float = 1e-5            # Adam eps (peg_walker: 1e-5)
    rssm_clip: float = 100.0          # gradient clip (peg_walker: 100)
    rssm_wd: float = 1e-6             # weight decay (peg_walker: 1e-6)

    # ── Reset proposer (None = initial state distribution) ──────────────────
    reset_proposer_name: Optional[Literal[
        "random_env_goals", "rb", "q_epistemic", "ucgr",
        "max_critic_to_env", "mega", "omega", "tldr", "peg",
    ]] = None

    # ── Fork redistribution (None = disabled) ───────────────────────────────
    fork_type: Optional[str] = None  # "smc"
    fork_sampling_temperature: float = 1.0
    exploration_metric_name: str = "q_epistemic"

    def check_config(self, config):
        assert config.episode_length - 1 == self.num_gcp_steps + self.num_ep_steps, (
            "episode_length - 1 must be equal to num_gcp_steps + num_ep_steps"
        )
        assert config.num_envs * (config.episode_length - 1) % self.batch_size == 0, (
            "num_envs * (episode_length - 1) must be divisible by batch_size"
        )
        if self.fork_type is not None:
            assert config.episode_length % self.unroll_length == 0, (
                "episode_length must be divisible by unroll_length when fork is enabled"
            )

    def train_fn(
        self,
        config: "RunConfig",
        train_env: Union[envs_v1.Env, envs.Env],
        eval_env: Optional[Union[envs_v1.Env, envs.Env]] = None,
        randomization_fn: Optional[
            Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
        ] = None,
        progress_fn: Callable[[int, Metrics], None] = lambda *args: None,
    ):
        self.check_config(config)

        unwrapped_env = train_env

        action_size = train_env.action_size
        state_size  = train_env.state_dim
        goal_size   = len(train_env.goal_indices)
        obs_size    = state_size + goal_size

        # ── Eval env ──────────────────────────────────────────────────────────
        eval_env = TrajectoryIdWrapper(eval_env)
        eval_env = VmapWrapper(eval_env)
        eval_env = EpisodeWrapper(eval_env, config.episode_length, config.action_repeat)
        eval_env = EvalAutoResetWrapper(eval_env)

        # ── Train env: GoExploreWrapper manages phase transitions ─────────────
        train_env = VmapWrapper(train_env)
        train_env = EpisodeWrapper(train_env, config.episode_length, config.action_repeat)
        train_env = GoExploreWrapper(
            train_env,
            num_gcp_steps=self.num_gcp_steps,
            num_ep_steps=self.num_ep_steps,
            state_size=state_size,
            goal_size=goal_size,
            goal_indices=unwrapped_env.goal_indices,
        )

        # ── Step count bookkeeping ────────────────────────────────────────────
        env_steps_per_actor_step = config.num_envs * self.unroll_length
        num_prefill_env_steps    = self.min_replay_size * config.num_envs
        num_prefill_actor_steps  = int(np.ceil(self.min_replay_size / self.unroll_length))

        available_env_steps          = config.total_env_steps - num_prefill_env_steps
        env_steps_per_epoch          = available_env_steps // config.num_evals
        num_training_steps_per_epoch = env_steps_per_epoch // env_steps_per_actor_step

        assert num_training_steps_per_epoch > 0

        logging.info("num_prefill_env_steps:          %d", num_prefill_env_steps)
        logging.info("num_prefill_actor_steps:         %d", num_prefill_actor_steps)
        logging.info("env_steps_per_epoch:             %d", env_steps_per_epoch)
        logging.info("num_training_steps_per_epoch:    %d", num_training_steps_per_epoch)

        random.seed(config.seed)
        np.random.seed(config.seed)
        key = jax.random.PRNGKey(config.seed)
        key, buffer_key, eval_env_key, env_key, actor_key, critic_key = jax.random.split(key, 6)

        # ── GCP (goal-conditioned policy) — the only policy ───────────────────
        gcp_actor, gcp_critic = get_algorithm(
            agent_type=self.agent_type,
            action_size=action_size,
            obs_size=obs_size,
            state_size=state_size,
            goal_indices=train_env.goal_indices,
            h_dim=self.h_dim,
            n_hidden=self.n_hidden,
            skip_connections=self.skip_connections,
            use_relu=self.use_relu,
            use_ln=self.use_ln,
            repr_dim=self.repr_dim,
            discounting=self.discounting,
            energy_fn=self.energy_fn,
            n_critics=self.n_critics,
        )

        gcp_actor_params  = gcp_actor.init(actor_key,  np.ones([1, obs_size]))
        gcp_critic_params = gcp_critic.init(critic_key, np.ones([1, obs_size]))

        gcp_actor_state = TrainState.create(
            apply_fn=gcp_actor.apply,
            params=gcp_actor_params,
            tx=optax.adam(learning_rate=self.policy_lr),
        )
        gcp_critic_states = gcp_critic.create_critic_states(gcp_critic_params, self.critic_lr)

        target_entropy = -0.5 * action_size
        log_alpha      = jnp.asarray(0.0, dtype=jnp.float32)
        alpha_state    = TrainState.create(
            apply_fn=None,
            params={"log_alpha": log_alpha},
            tx=optax.adam(learning_rate=self.alpha_lr),
        )

        target_critic_params = None
        if self.agent_type == "sac":
            target_critic_params = gcp_critic_params

        # ── Explore policy (optional separate non-goal-conditioned policy) ────
        has_explore_policy = self.explore_policy_type is not None
        explore_actor = None
        explore_critic = None
        explore_actor_state = None
        explore_critic_states = None
        explore_alpha_state = None
        explore_target_critic_params = None

        if has_explore_policy:
            key, explore_actor_key, explore_critic_key = jax.random.split(key, 3)
            explore_actor, explore_critic = get_algorithm(
                self.explore_policy_type,
                action_size=action_size,
                obs_size=state_size,  # non-goal-conditioned: raw state only
                h_dim=self.h_dim,
                n_hidden=self.n_hidden,
                use_relu=self.use_relu,
                use_ln=self.use_ln,
                n_critics=self.explore_n_critics,
            )
            explore_actor_params = explore_actor.init(explore_actor_key, np.ones([1, state_size]))
            explore_critic_params = explore_critic.init(explore_critic_key, np.ones([1, state_size]))
            explore_actor_state = TrainState.create(
                apply_fn=explore_actor.apply,
                params=explore_actor_params,
                tx=optax.chain(
                    optax.clip_by_global_norm(self.explore_grad_clip),
                    optax.adam(learning_rate=self.explore_policy_lr, eps=1e-5),
                ),
            )
            explore_critic_states = explore_critic.create_critic_states(
                explore_critic_params, self.explore_critic_lr,
                grad_clip=self.explore_grad_clip,
            )
            explore_log_alpha = jnp.asarray(0.0, dtype=jnp.float32)
            explore_alpha_state = TrainState.create(
                apply_fn=None,
                params={"log_alpha": explore_log_alpha},
                tx=optax.chain(
                    optax.clip_by_global_norm(self.explore_grad_clip),
                    optax.adam(learning_rate=self.explore_alpha_lr, eps=1e-5),
                ),
            )
            explore_target_critic_params = explore_critic_params

        # ── TLDR traj encoder (when explore_reward_type == "tldr") ────────────
        te_state = None
        dual_lam_state = None
        pbe_rms_state = None
        traj_encoder_module = None

        if self.explore_reward_type == "tldr":
            from .tldr import TrajEncoder
            traj_encoder_module = TrajEncoder(
                hidden_dim=self.te_hidden_dim,
                hidden_layers=self.te_hidden_layers,
                output_dim=self.te_output_dim,
                use_layer_norm=self.te_layer_norm,
            )
            key, te_key = jax.random.split(key)
            te_params = traj_encoder_module.init(te_key, jnp.ones([1, state_size]))
            te_state = TrainState.create(
                apply_fn=traj_encoder_module.apply,
                params=te_params,
                tx=optax.adam(learning_rate=self.te_lr),
            )
            log_lam = jnp.log(jnp.asarray(self.dual_lam_init, dtype=jnp.float32))
            dual_lam_state = TrainState.create(
                apply_fn=None,
                params={"log_lam": log_lam},
                tx=optax.adam(learning_rate=self.dual_lr),
            )
            pbe_rms_state = {
                "M": jnp.zeros((1,)),
                "S": jnp.ones((1,)),
                "n": jnp.asarray(1e-4),
            }

        # ── PEG world model ensemble (+ optional encoder/decoder) ────────────
        wm_ensemble_states = None
        wm_modules = None
        obs_encoder_module = None
        obs_decoder_module = None
        use_peg = self.explore_reward_type == "peg" or self.goal_proposer_name == "peg"

        if use_peg:
            from .peg import ObsDecoder, ObsEncoder, WorldModelMLP

            # When using latent space, WM operates on latent vectors.
            wm_state_size = self.wm_latent_dim if self.use_peg_latent_space else state_size
            wm_input_size = wm_state_size + action_size

            wm_modules = [
                WorldModelMLP(
                    state_size=wm_state_size,
                    hidden_dim=self.wm_hidden_dim,
                    hidden_layers=self.wm_hidden_layers,
                )
                for _ in range(self.wm_ensemble_size)
            ]
            wm_ensemble_states_list = []
            for i in range(self.wm_ensemble_size):
                key, wm_key = jax.random.split(key)
                wm_params = wm_modules[i].init(wm_key, jnp.ones([1, wm_input_size]))
                wm_adam = optax.adam(learning_rate=self.wm_lr, eps=self.wm_eps)
                wm_tx = optax.chain(
                    optax.clip_by_global_norm(self.wm_grad_clip),
                    wm_adam,
                    optax.add_decayed_weights(self.wm_wd),
                ) if self.wm_grad_clip > 0 else optax.chain(
                    wm_adam, optax.add_decayed_weights(self.wm_wd)
                )
                wm_ensemble_states_list.append(TrainState.create(
                    apply_fn=wm_modules[i].apply,
                    params=wm_params,
                    tx=wm_tx,
                ))
            wm_ensemble_states = tuple(wm_ensemble_states_list)

            if self.use_peg_latent_space:
                obs_encoder_module = ObsEncoder(
                    latent_dim=self.wm_latent_dim,
                    hidden_dim=self.enc_hidden_dim,
                    hidden_layers=self.enc_hidden_layers,
                )
                obs_decoder_module = ObsDecoder(
                    obs_dim=state_size,
                    hidden_dim=self.enc_hidden_dim,
                    hidden_layers=self.enc_hidden_layers,
                )
                key, enc_key, dec_key = jax.random.split(key, 3)
                enc_params = obs_encoder_module.init(enc_key, jnp.ones([1, state_size]))
                dec_params = obs_decoder_module.init(dec_key, jnp.ones([1, self.wm_latent_dim]))
                obs_encoder_state = TrainState.create(
                    apply_fn=obs_encoder_module.apply,
                    params=enc_params,
                    tx=optax.adam(learning_rate=self.enc_lr),
                )
                obs_decoder_state = TrainState.create(
                    apply_fn=obs_decoder_module.apply,
                    params=dec_params,
                    tx=optax.adam(learning_rate=self.dec_lr),
                )
            else:
                obs_encoder_state = None
                obs_decoder_state = None

        else:
            obs_encoder_state = None
            obs_decoder_state = None

        # Reward normalisation: the original PEG uses StreamNorm(momentum=1.0) which
        # is explicitly disabled ("momentum of 1 disables normalisation" per DreamerV2).
        # We match that by keeping peg_rms_state=None so compute_peg_explore_reward
        # skips the running-mean division entirely.
        peg_rms_state = None

        # ── RSSM world model (replaces WorldModelMLP when use_rssm=True, or for peg_rssm) ─────
        rssm_state = None
        disag_state = None
        rssm_module = None
        disag_module = None
        _use_rssm = self.use_rssm or (self.goal_proposer_name == "peg_rssm")

        if _use_rssm:
            from .rssm import RSSMWorldModel, DisagreementEnsemble, init_rssm_states
            rssm_module = RSSMWorldModel(
                obs_dim=state_size,
                stoch=self.rssm_stoch,
                deter=self.rssm_deter,
                hidden=self.rssm_hidden,
                min_std=self.rssm_min_std,
                n_enc_layers=self.rssm_enc_layers,
                enc_units=self.rssm_enc_units,
            )
            disag_module = DisagreementEnsemble(
                n_heads=self.disag_heads,
                stoch_size=self.rssm_stoch,
                n_layers=self.disag_layers,
                n_units=self.disag_units,
            )
            key, rssm_init_key = jax.random.split(key)
            rssm_state, disag_state = init_rssm_states(
                rssm_module, disag_module,
                obs_dim=state_size, action_dim=action_size,
                key=rssm_init_key,
                lr=self.rssm_lr, eps=self.rssm_eps, clip=self.rssm_clip, wd=self.rssm_wd,
            )

        # ── TrainingState ─────────────────────────────────────────────────────
        training_state = TrainingState(
            env_steps=jnp.zeros(()),
            gradient_steps=jnp.zeros(()),
            experience_count=jnp.array(0, dtype=jnp.int32),
            actor_state=gcp_actor_state,
            critic_states=gcp_critic_states,
            alpha_state=alpha_state,
            target_critic_params=target_critic_params,
            explore_actor_state=explore_actor_state,
            explore_critic_states=explore_critic_states,
            explore_alpha_state=explore_alpha_state,
            explore_target_critic_params=explore_target_critic_params,
            te_state=te_state,
            dual_lam_state=dual_lam_state,
            pbe_rms_state=pbe_rms_state,
            wm_ensemble_states=wm_ensemble_states,
            peg_rms_state=peg_rms_state,
            obs_encoder_state=obs_encoder_state,
            obs_decoder_state=obs_decoder_state,
            rssm_state=rssm_state,
            disag_state=disag_state,
        )

        # ── Goal proposer ────────────────────────────────────────────────────
        goal_proposer = create_goal_proposer(
            self.goal_proposer_name,
            unwrapped_env,
            config.num_envs,
            self.num_candidates,
            state_size=unwrapped_env.state_dim,
            goal_indices=unwrapped_env.goal_indices,
            actor=gcp_actor,
            critic=gcp_critic,
            discounting=self.discounting,
            traj_encoder=traj_encoder_module,
            knn_k=self.knn_k,
            knn_clip=self.knn_clip,
            wm_modules=wm_modules,
            mppi_horizon=self.mppi_horizon,
            mppi_samples=self.mppi_samples,
            mppi_iterations=self.mppi_iterations,
            mppi_gamma=self.mppi_gamma,
            obs_encoder=obs_encoder_module,
            obs_decoder=obs_decoder_module,
            rssm_module=rssm_module,
            disag_module=disag_module,
            action_size=action_size,
        )

        # ── Reset proposer (optional) ────────────────────────────────────────
        reset_proposer = None
        if self.reset_proposer_name is not None:
            reset_proposer = create_goal_proposer(
                self.reset_proposer_name,
                unwrapped_env,
                config.num_envs,
                self.num_candidates,
                state_size=unwrapped_env.state_dim,
                goal_indices=unwrapped_env.goal_indices,
                actor=gcp_actor,
                critic=gcp_critic,
                discounting=self.discounting,
                traj_encoder=traj_encoder_module,
                knn_k=self.knn_k,
                knn_clip=self.knn_clip,
                wm_modules=wm_modules,
                mppi_horizon=self.mppi_horizon,
                mppi_samples=self.mppi_samples,
                mppi_iterations=self.mppi_iterations,
                mppi_gamma=self.mppi_gamma,
            )

        # ── Fork redistribution (optional) ───────────────────────────────────
        use_fork = self.fork_type is not None
        fork_metric_fn = None
        goal_idx_arr = jnp.array(unwrapped_env.goal_indices)
        if use_fork:
            # We only use fork_metric_fn; the resampling is done inline
            # with phase masking (go-phase states get -inf scores).
            _, fork_metric_fn = create_fork_fn(
                fork_type=self.fork_type,
                env=unwrapped_env,
                num_envs=config.num_envs,
                num_candidates=self.num_candidates,
                state_size=state_size,
                goal_indices=tuple(unwrapped_env.goal_indices),
                actor=gcp_actor,
                critic=gcp_critic,
                exploration_metric_name=self.exploration_metric_name,
                fork_sampling_temperature=self.fork_sampling_temperature,
                discounting=self.discounting,
            )

        # ── Explore reward function ──────────────────────────────────────────
        explore_reward_fn = None
        if has_explore_policy:
            explore_reward_fn = create_explore_reward_fn(
                reward_type=self.explore_reward_type,
                explore_actor=explore_actor,
                explore_critic=explore_critic,
                traj_encoder_module=traj_encoder_module,
                knn_k=self.knn_k,
                knn_clip=self.knn_clip,
                dual_slack=self.dual_slack,
                wm_modules=wm_modules,
                obs_encoder_module=obs_encoder_module,
                use_peg_latent_space=self.use_peg_latent_space,
            )

        # ── Env reset ────────────────────────────────────────────────────────
        random_goals_proposer = create_random_env_goals_proposer(unwrapped_env, config.num_envs)
        env_keys      = jax.random.split(env_key, config.num_envs)
        initial_goals = jax.vmap(random_goals_proposer)(env_keys)

        env_state = train_env.reset(env_keys, goal=initial_goals)
        info = dict(env_state.info)
        info['proposed_goals'] = initial_goals
        env_state = env_state.replace(info=info)

        train_env.step = jax.jit(train_env.step)
        assert obs_size == train_env.observation_size, (
            f"obs_size: {obs_size}, observation_size: {train_env.observation_size}"
        )

        # ── Replay buffer ────────────────────────────────────────────────────
        # Force SAC-style next_observation storage if explore policy needs it
        needs_next_obs = (self.agent_type == "sac") or has_explore_policy
        buffer_agent_type = "sac" if needs_next_obs else self.agent_type

        dummy_transition = create_single_dummy_transition(
            obs_size=obs_size,
            action_size=action_size,
            agent_type=buffer_agent_type,
            include_phase=True,
        )

        def jit_wrap(buffer):
            buffer.insert_internal = jax.jit(buffer.insert_internal)
            buffer.sample_internal = jax.jit(buffer.sample_internal)
            return buffer

        replay_buffer = jit_wrap(
            TrajectoryUniformSamplingQueue(
                max_replay_size=self.max_replay_size,
                dummy_data_sample=dummy_transition,
                sample_batch_size=self.batch_size,
                num_envs=config.num_envs,
                episode_length=config.episode_length,
            )
        )
        buffer_state = jax.jit(replay_buffer.init)(buffer_key)

        dummy_batch_transition = create_dummy_transition_for_buffer(
            unroll_length=self.unroll_length,
            num_envs=config.num_envs,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=buffer_agent_type,
            include_phase=True,
        )
        buffer_state = replay_buffer.insert(buffer_state, dummy_batch_transition)

        dummy_goal_proposer_transition = create_dummy_transition_for_goal_proposer(
            num_envs=config.num_envs,
            episode_length=config.episode_length,
            obs_size=obs_size,
            action_size=action_size,
            agent_type=buffer_agent_type,
            include_phase=True,
        )
        goal_proposer_state = GoalProposerState(
            transitions_sample=dummy_goal_proposer_transition,
            actor_params=gcp_actor_state.params,
            critic_params={i: cs.params for i, cs in enumerate(gcp_critic_states)},
            te_params=te_state.params if te_state is not None else None,
            wm_ensemble_params=tuple(s.params for s in wm_ensemble_states) if wm_ensemble_states is not None else None,
            obs_encoder_params=obs_encoder_state.params if obs_encoder_state is not None else None,
            obs_decoder_params=obs_decoder_state.params if obs_decoder_state is not None else None,
            rssm_params=rssm_state.params if rssm_state is not None else None,
            disag_params=disag_state.params if disag_state is not None else None,
        )

        # ── actor_step ────────────────────────────────────────────────────────
        explore_eps = self.explore_eps_random_action

        def actor_step(training_state, env, env_state, key, extra_fields):
            """One env step dispatching between Go policy and Explore policy."""
            key, gcp_key, explore_key, random_key, eps_key, env_rng = jax.random.split(key, 6)

            phase     = env_state.info['phase']           # (num_envs,)
            go_goal   = env_state.info['go_goal']         # (num_envs, goal_size)
            raw_state = env_state.obs[:, :state_size]     # (num_envs, state_size)
            in_explore = (phase == 1)                     # (num_envs,)

            # GCP obs: [state, go_goal] — used in both phases when no explore policy
            gcp_obs = jnp.concatenate([raw_state, go_goal], axis=-1)

            # Go phase: always stochastic, no eps_random
            gcp_actions = gcp_actor.sample_actions(
                training_state.actor_state.params, gcp_obs, gcp_key, is_deterministic=False
            )

            if has_explore_policy:
                # Explore phase: separate non-goal-conditioned SAC policy, no eps_random
                explore_actions = explore_actor.sample_actions(
                    training_state.explore_actor_state.params,
                    raw_state,  # state only, no goal
                    explore_key,
                    is_deterministic=False,
                )
                # Select per-env: go policy for go phase, explore policy for explore phase
                actions = jnp.where(in_explore[:, None], explore_actions, gcp_actions)
            else:
                # Default: GCP in both phases, explore phase gets eps_random=0.2
                random_actions = jax.random.uniform(
                    random_key, shape=gcp_actions.shape, minval=-1.0, maxval=1.0
                )
                use_random = jax.random.uniform(eps_key, shape=(gcp_actions.shape[0],)) < explore_eps
                use_random = jnp.logical_and(in_explore, use_random)
                actions = jnp.where(use_random[:, None], random_actions, gcp_actions)

            nstate = env.step(env_state, actions, env_rng)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            state_extras['phase'] = phase

            # Always store [state, goal] as observation and next_observation
            next_obs = jnp.concatenate(
                [nstate.obs[:, :state_size], go_goal], axis=-1
            ) if needs_next_obs else None

            return nstate, Transition(
                observation=gcp_obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=next_obs,
                extras={"state_extras": state_extras},
            )

        # ── get_experience (with macro_step, reset proposer, fork) ────────────
        num_envs_     = config.num_envs
        episode_length_ = config.episode_length

        def get_experience(training_state, env_state, buffer_state, key,
                           goal_proposer_state, macro_step):
            buffer_state, transitions_sample = replay_buffer.sample(buffer_state)

            goal_proposer_state = goal_proposer_state.replace(
                transitions_sample=transitions_sample,
                actor_params=training_state.actor_state.params,
                critic_params={i: cs.params for i, cs in enumerate(training_state.critic_states)},
                te_params=training_state.te_state.params if training_state.te_state is not None else None,
                wm_ensemble_params=tuple(s.params for s in training_state.wm_ensemble_states) if training_state.wm_ensemble_states is not None else None,
                obs_encoder_params=training_state.obs_encoder_state.params if training_state.obs_encoder_state is not None else None,
                obs_decoder_params=training_state.obs_decoder_state.params if training_state.obs_decoder_state is not None else None,
                rssm_params=training_state.rssm_state.params if training_state.rssm_state is not None else None,
                disag_params=training_state.disag_state.params if training_state.disag_state is not None else None,
            )

            key, k_propose, k_roll, k_fork = jax.random.split(key, 4)

            # ── Episode start: propose goals (and optionally starts) ─────────
            def propose_goals_and_starts(es, prop_key, gps):
                info = dict(es.info)
                first_obs = info['first_obs']
                viz_key, goal_key, reset_key = jax.random.split(prop_key, 3)
                viz_env_idx = jax.random.randint(viz_key, (), 0, num_envs_)
                goal_keys = jax.random.split(goal_key, num_envs_)

                def propose_single(rng_key, obs, state):
                    goal, _, log_data = goal_proposer(rng_key, obs, state)
                    return goal, log_data

                new_goals, log_data_tree = jax.vmap(
                    propose_single, in_axes=(0, 0, None)
                )(goal_keys, first_obs, gps)
                info['proposed_goals'] = new_goals

                # Reset proposer: propose start states if configured
                if reset_proposer is not None:
                    reset_keys = jax.random.split(reset_key, num_envs_)
                    def propose_start(rng_key, obs, state):
                        start, _, _ = reset_proposer(rng_key, obs, state)
                        return start
                    new_starts = jax.vmap(
                        propose_start, in_axes=(0, 0, None)
                    )(reset_keys, first_obs, gps)
                    info['proposed_starts'] = new_starts

                env_steps = training_state.env_steps

                def log_viz(log_data_tree_np, viz_idx, steps):
                    selected = {k: v[viz_idx] for k, v in log_data_tree_np.items()}
                    handle_goal_proposer_visualization(
                        selected, self.goal_proposer_name,
                        unwrapped_env.x_bounds, unwrapped_env.y_bounds, steps
                    )
                    return jnp.array(0, dtype=jnp.int32)

                jax.experimental.io_callback(
                    log_viz, jnp.array(0, dtype=jnp.int32),
                    log_data_tree, viz_env_idx, env_steps,
                )
                return es.replace(info=info)

            env_state = jax.lax.cond(
                macro_step == 0,
                lambda: propose_goals_and_starts(env_state, k_propose, goal_proposer_state),
                lambda: env_state,
            )

            # ── Rollout for unroll_length steps ──────────────────────────────
            def rollout_fn(carry, _):
                env_state, k = carry
                k, next_k = jax.random.split(k)
                env_state, transition = actor_step(
                    training_state,
                    train_env,
                    env_state,
                    k,
                    extra_fields=("truncation", "traj_id", "phase"),
                )
                return (env_state, next_k), transition

            (env_state, _), data = jax.lax.scan(
                rollout_fn, (env_state, k_roll), (), length=self.unroll_length
            )
            buffer_state = replay_buffer.insert(buffer_state, data)

            # ── Macro step update ────────────────────────────────────────────
            macro_new = macro_step + self.unroll_length

            # ── Fork redistribution (between unroll chunks, not at episode end)
            if use_fork:
                def fork_redistribute(es, fkey, gps):
                    fk, rk = jax.random.split(fkey)
                    info = dict(es.info)
                    phase = info['phase']
                    in_explore = (phase == 1)

                    states_full = es.obs[:, :state_size]
                    goals = info['go_goal']

                    # Score all states, set go-phase scores to -inf so they get
                    # zero weight in the softmax resampling
                    all_scores = fork_metric_fn(fk, states_full, goals, gps)
                    masked_scores = jnp.where(in_explore, all_scores, -jnp.inf)

                    # Run SMC on all states but with masked scores
                    n = states_full.shape[0]
                    logits = masked_scores / jnp.asarray(self.fork_sampling_temperature, dtype=masked_scores.dtype)
                    logits = logits - jnp.max(logits)
                    weights = jnp.exp(logits)
                    weights = weights / (jnp.sum(weights) + 1e-10)
                    log_w = jnp.log(weights + 1e-10)
                    rk_split = jax.random.split(rk, n)
                    indices = jax.vmap(lambda k: jax.random.categorical(k, log_w))(rk_split)
                    forked_states = states_full[indices]

                    # Only apply fork to explore-phase envs
                    selected = jnp.where(in_explore[:, None], forked_states, states_full)
                    new_starts = selected[:, goal_idx_arr]

                    # Reset physics for all (JAX traces both), mask to explore only
                    rk2 = jax.random.split(jax.random.fold_in(rk, 1), num_envs_)
                    reset_state = train_env.env.reset(rk2, goal=goals, start=new_starts)

                    def _where_explore(x_reset, x_current):
                        if not hasattr(x_reset, 'shape'):
                            return x_current
                        if x_reset.ndim == 0:
                            return x_current
                        if x_reset.shape[0] != in_explore.shape[0]:
                            return x_current
                        mask = jnp.reshape(in_explore, [in_explore.shape[0]] + [1] * (x_reset.ndim - 1))
                        return jnp.where(mask, x_reset, x_current)

                    new_pipeline = jax.tree.map(
                        _where_explore, reset_state.pipeline_state, es.pipeline_state
                    )
                    new_obs = jnp.where(in_explore[:, None], reset_state.obs, es.obs)

                    # Increment traj_id for forked explore envs only
                    info['traj_id'] = info['traj_id'] + jnp.where(in_explore, 1.0, 0.0)

                    return es.replace(pipeline_state=new_pipeline, obs=new_obs, info=info)

                env_state = jax.lax.cond(
                    macro_new < episode_length_,
                    lambda: fork_redistribute(env_state, k_fork, goal_proposer_state),
                    lambda: env_state,
                )

            # Reset macro_step at episode boundary
            macro_out = jax.lax.cond(
                macro_new >= episode_length_,
                lambda: jnp.int32(0),
                lambda: macro_new,
            )

            return env_state, buffer_state, goal_proposer_state, macro_out

        # ── prefill_replay_buffer ─────────────────────────────────────────────
        def prefill_replay_buffer(training_state, env_state, buffer_state, key,
                                  goal_proposer_state, macro_step):
            def f(carry, _):
                ts, es, bs, k, gps, ms = carry
                k, new_k = jax.random.split(k)
                es, bs, gps, ms = get_experience(ts, es, bs, k, gps, ms)
                ts = ts.replace(
                    env_steps=ts.env_steps + config.num_envs * self.unroll_length,
                )
                return (ts, es, bs, new_k, gps, ms), ()

            return jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state, macro_step),
                (),
                length=num_prefill_actor_steps,
            )[0]

        # ── update_networks ───────────────────────────────────────────────────
        explore_target_entropy = -0.5 * action_size

        @jax.jit
        def update_networks(carry, transitions):
            training_state, key = carry
            if self.agent_type == "sac":
                key, alpha_key, critic_key, actor_key = jax.random.split(key, 4)
            else:
                key, critic_key, actor_key = jax.random.split(key, 3)

            context = dict(
                **vars(self),
                **vars(config),
                state_size=state_size,
                action_size=action_size,
                goal_size=goal_size,
                obs_size=obs_size,
                goal_indices=train_env.goal_indices,
                target_entropy=target_entropy,
                explore_target_entropy=explore_target_entropy,
            )
            networks = dict(actor=gcp_actor, critic=gcp_critic)

            # ── GCP update on ALL transitions ────────────────────────────────
            metrics = {}
            if self.agent_type == "crl":
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
            else:  # sac
                training_state, alpha_metrics  = update_alpha_sac(context, networks, transitions, training_state, alpha_key)
                training_state, critic_metrics = gcp_critic.update(context, networks, transitions, training_state, critic_key)
                training_state, actor_metrics  = gcp_actor.update(context, networks, transitions, training_state, actor_key)
                metrics.update(alpha_metrics)
            metrics.update(critic_metrics)
            metrics.update(actor_metrics)

            # Update GCP SAC target network
            if self.agent_type == "sac" and training_state.target_critic_params is not None:
                full_cp = {}
                for i, cs in enumerate(training_state.critic_states):
                    for lname, lparams in cs.params.items():
                        full_cp[f"critic_{i}_{lname}"] = lparams
                new_target = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    training_state.target_critic_params, full_cp,
                )
                training_state = training_state.replace(target_critic_params=new_target)

            # ── PEG world model training (on ALL transitions) ─────────────────
            if use_peg:
                wm_obs = transitions.observation[:, :state_size]
                wm_next_obs = transitions.next_observation[:, :state_size]
                wm_actions = transitions.action
                if self.use_peg_latent_space:
                    from .peg import train_encoder_decoder_and_ensemble as train_enc_wm
                    new_enc_state, new_dec_state, new_wm_states, wm_metrics = train_enc_wm(
                        training_state.obs_encoder_state,
                        training_state.obs_decoder_state,
                        training_state.wm_ensemble_states,
                        obs_encoder_module, obs_decoder_module, wm_modules,
                        wm_obs, wm_actions, wm_next_obs,
                        recon_coef=self.enc_recon_coef,
                    )
                    training_state = training_state.replace(
                        wm_ensemble_states=new_wm_states,
                        obs_encoder_state=new_enc_state,
                        obs_decoder_state=new_dec_state,
                    )
                else:
                    from .peg import train_world_model_ensemble as train_wm
                    new_wm_states, wm_metrics = train_wm(
                        training_state.wm_ensemble_states, wm_modules,
                        wm_obs, wm_actions, wm_next_obs,
                    )
                    training_state = training_state.replace(wm_ensemble_states=new_wm_states)
                for k, v in wm_metrics.items():
                    metrics[k] = v

            # ── Explore policy update (reuses standard SAC losses) ────────────
            if has_explore_policy:
                key, explore_alpha_key, explore_critic_key, explore_actor_key, reward_key = jax.random.split(key, 5)

                # Phase mask → sample_weights so standard losses weight by phase
                phase = transitions.extras["state_extras"]["phase"]
                phase_mask = (phase == 1).astype(jnp.float32)

                # Compute intrinsic reward via the factory-created reward fn
                explore_obs = transitions.observation[:, :state_size]
                explore_next_obs = transitions.next_observation[:, :state_size]

                if _use_rssm and "rssm_reward" in transitions.extras.get("state_extras", {}):
                    # RSSM rewards were pre-computed on the raw trajectories and attached
                    # to state_extras before process_transitions; retrieve them here.
                    explore_reward = transitions.extras["state_extras"]["rssm_reward"]
                    reward_metrics = {}
                else:
                    reward_state = ExploreRewardState(
                        explore_actor_params=training_state.explore_actor_state.params,
                        explore_critic_states=training_state.explore_critic_states,
                        te_state=training_state.te_state,
                        dual_lam_state=training_state.dual_lam_state,
                        pbe_rms_state=training_state.pbe_rms_state,
                        wm_ensemble_states=training_state.wm_ensemble_states,
                        peg_rms_state=training_state.peg_rms_state,
                        obs_encoder_params=(
                            training_state.obs_encoder_state.params
                            if training_state.obs_encoder_state is not None else None
                        ),
                    )
                    explore_reward, reward_state, reward_metrics = explore_reward_fn(
                        reward_state, explore_obs, explore_next_obs, transitions.action, reward_key,
                    )
                    training_state = training_state.replace(
                        te_state=reward_state.te_state,
                        dual_lam_state=reward_state.dual_lam_state,
                        pbe_rms_state=reward_state.pbe_rms_state,
                        peg_rms_state=reward_state.peg_rms_state,
                    )
                for k, v in reward_metrics.items():
                    metrics[f"explore_{k}"] = v
                explore_transitions = transitions._replace(
                    observation=explore_obs,
                    next_observation=explore_next_obs,
                    reward=explore_reward,
                )

                # Swap explore fields into standard positions so standard losses work
                explore_ts = training_state.replace(
                    actor_state=training_state.explore_actor_state,
                    critic_states=training_state.explore_critic_states,
                    alpha_state=training_state.explore_alpha_state,
                    target_critic_params=training_state.explore_target_critic_params,
                )
                explore_context = {**context, "sample_weights": phase_mask, "target_entropy": explore_target_entropy}
                explore_nets = dict(actor=explore_actor, critic=explore_critic)

                # Standard SAC update (alpha → critic → actor)
                explore_ts, explore_alpha_m = update_alpha_sac(
                    explore_context, explore_nets, explore_transitions, explore_ts, explore_alpha_key,
                )
                explore_ts, explore_critic_m = explore_critic.update(
                    explore_context, explore_nets, explore_transitions, explore_ts, explore_critic_key,
                )
                explore_ts, explore_actor_m = explore_actor.update(
                    explore_context, explore_nets, explore_transitions, explore_ts, explore_actor_key,
                )

                # Target network EMA for explore critic
                full_cp = {}
                for i, cs in enumerate(explore_ts.critic_states):
                    for lname, lparams in cs.params.items():
                        full_cp[f"critic_{i}_{lname}"] = lparams
                new_explore_target = jax.tree_util.tree_map(
                    lambda x, y: x * (1 - self.tau) + y * self.tau,
                    explore_ts.target_critic_params, full_cp,
                )

                # Swap results back into explore fields
                training_state = training_state.replace(
                    explore_actor_state=explore_ts.actor_state,
                    explore_critic_states=explore_ts.critic_states,
                    explore_alpha_state=explore_ts.alpha_state,
                    explore_target_critic_params=new_explore_target,
                )

                # Prefix metrics with "explore_"
                for k, v in {**explore_alpha_m, **explore_critic_m, **explore_actor_m}.items():
                    metrics[f"explore_{k}"] = v

            training_state = training_state.replace(gradient_steps=training_state.gradient_steps + 1)
            return (training_state, key), metrics

        # ── training_step ─────────────────────────────────────────────────────
        @jax.jit
        def training_step(training_state, env_state, buffer_state, key,
                          goal_proposer_state, macro_step):
            exp_key, process_key, train_key, rssm_key = jax.random.split(key, 4)

            env_state, buffer_state, updated_gps, macro_next = get_experience(
                training_state, env_state, buffer_state, exp_key,
                goal_proposer_state, macro_step,
            )
            training_state = training_state.replace(
                env_steps=training_state.env_steps + env_steps_per_actor_step,
            )

            # Sample raw trajectories (num_envs, episode_length, ...) for RSSM training
            # and GCP update.  RSSM training happens BEFORE process_transitions flattening.
            buffer_state, raw_transitions = replay_buffer.sample(buffer_state)

            # ── RSSM training on raw trajectories ────────────────────────────
            if _use_rssm:
                from .rssm import train_rssm_step as _train_rssm
                rssm_obs = raw_transitions.observation[:, :, :state_size]
                rssm_act = raw_transitions.action
                new_rssm_st, new_disag_st, rssm_rewards, rssm_metrics = _train_rssm(
                    training_state.rssm_state,
                    training_state.disag_state,
                    rssm_module,
                    disag_module,
                    rssm_obs,
                    rssm_act,
                    rssm_key,
                    kl_scale=self.rssm_kl_scale,
                    kl_free=self.rssm_kl_free,
                    kl_balance=self.rssm_kl_balance,
                )
                training_state = training_state.replace(
                    rssm_state=new_rssm_st,
                    disag_state=new_disag_st,
                )
                # Attach rssm_rewards to raw_transitions before flattening so that
                # update_networks can retrieve them keyed by "rssm_reward".
                old_se = raw_transitions.extras["state_extras"]
                new_se = {**old_se, "rssm_reward": rssm_rewards}
                raw_transitions = raw_transitions._replace(
                    extras={"state_extras": new_se}
                )

            # Flatten and process for GCP + explore policy updates
            transitions, _ = gcp_actor.process_transitions(
                raw_transitions, process_key, self.batch_size, self.discounting,
                state_size, tuple(train_env.goal_indices),
                train_env.goal_reach_thresh, self.use_her,
            )
            (training_state, _), metrics = jax.lax.scan(
                update_networks, (training_state, train_key), transitions
            )

            if _use_rssm:
                # update_networks scan produces (num_batches,) shaped metrics;
                # expand RSSM scalars to match so training_epoch's outer scan
                # can stack them uniformly.
                num_batches = jax.tree_util.tree_leaves(metrics)[0].shape[0]
                for k, v in rssm_metrics.items():
                    metrics[k] = jnp.full((num_batches,), v)

            return (training_state, env_state, buffer_state, updated_gps, macro_next), metrics

        # ── training_epoch ────────────────────────────────────────────────────
        @jax.jit
        def training_epoch(training_state, env_state, buffer_state, key,
                           goal_proposer_state, macro_step):
            # Snapshot cumulative counters *before* the epoch so we can compute
            # epoch-level deltas (rather than a lifetime average).
            pre_completions   = jnp.sum(env_state.info['go_completions_total'])
            pre_successes     = jnp.sum(env_state.info['go_successes_total'])
            pre_success_steps = jnp.sum(env_state.info['go_success_steps_total'])

            def f(carry, _):
                ts, es, bs, k, gps, ms = carry
                k, train_key = jax.random.split(k)
                (ts, es, bs, gps, ms), metrics = training_step(ts, es, bs, train_key, gps, ms)
                return (ts, es, bs, k, gps, ms), metrics

            (training_state, env_state, buffer_state, key, goal_proposer_state, macro_step), metrics = jax.lax.scan(
                f,
                (training_state, env_state, buffer_state, key, goal_proposer_state, macro_step),
                (),
                length=num_training_steps_per_epoch,
            )

            # Go Explore phase metrics — epoch-level (current policy performance)
            epoch_completions   = jnp.sum(env_state.info['go_completions_total']) - pre_completions
            epoch_successes     = jnp.sum(env_state.info['go_successes_total']) - pre_successes
            epoch_success_steps = jnp.sum(env_state.info['go_success_steps_total']) - pre_success_steps

            go_success_rate = jnp.where(epoch_completions > 0,
                                        epoch_successes / epoch_completions,
                                        0.0)
            avg_go_steps    = jnp.where(epoch_successes > 0,
                                        epoch_success_steps / epoch_successes,
                                        0.0)

            scan_shape = jax.tree_util.tree_leaves(metrics)[0].shape if metrics else (1,)
            metrics["go_phase_success_rate"] = jnp.broadcast_to(go_success_rate, scan_shape)
            metrics["avg_go_phase_steps"]    = jnp.broadcast_to(avg_go_steps, scan_shape)
            metrics["buffer_current_size"]   = jnp.broadcast_to(replay_buffer.size(buffer_state), scan_shape)

            return training_state, env_state, buffer_state, goal_proposer_state, macro_step, metrics

        # ── prefill ───────────────────────────────────────────────────────────
        macro_init = jnp.int32(0)
        key, prefill_key = jax.random.split(key)
        training_state, env_state, buffer_state, _, goal_proposer_state, macro_step = prefill_replay_buffer(
            training_state, env_state, buffer_state, prefill_key, goal_proposer_state, macro_init
        )

        # ── Evaluator ─────────────────────────────────────────────────────────
        def eval_actor_step(training_state, env, env_state, extra_fields=()):
            actions = gcp_actor.sample_actions(
                training_state.actor_state.params,
                env_state.obs,
                jax.random.PRNGKey(0),
                is_deterministic=True,
            )
            nstate = env.step(env_state, actions)
            state_extras = {x: nstate.info[x] for x in extra_fields}
            return nstate, Transition(
                observation=env_state.obs,
                action=actions,
                reward=nstate.reward,
                discount=1 - nstate.done,
                next_observation=None,
                extras={"state_extras": state_extras},
            )

        evaluator = ActorEvaluator(
            eval_actor_step,
            eval_env,
            num_eval_envs=config.num_eval_envs,
            episode_length=config.episode_length,
            key=eval_env_key,
        )

        # ── Main training loop ────────────────────────────────────────────────
        training_walltime      = 0
        last_visualization_step = -1
        logging.info("starting training....")

        for ne in range(config.num_evals):
            t = time.time()
            key, epoch_key = jax.random.split(key)

            training_state, env_state, buffer_state, goal_proposer_state, macro_step, metrics = training_epoch(
                training_state, env_state, buffer_state, epoch_key, goal_proposer_state, macro_step
            )

            metrics = jax.tree_util.tree_map(jnp.mean, metrics)
            metrics = jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

            epoch_training_time  = time.time() - t
            training_walltime   += epoch_training_time

            sps = (env_steps_per_actor_step * num_training_steps_per_epoch) / epoch_training_time

            metrics_dict = {}
            for name, value in metrics.items():
                if hasattr(value, 'item'):
                    metrics_dict[f"training/{name}"] = float(value.item())
                elif hasattr(value, '__float__'):
                    metrics_dict[f"training/{name}"] = float(value)
                else:
                    metrics_dict[f"training/{name}"] = value

            metrics = {
                "training/sps": sps,
                "training/walltime": training_walltime,
                "training/envsteps": training_state.env_steps.item(),
                **metrics_dict,
            }
            current_step = int(training_state.env_steps.item())

            metrics = evaluator.run_evaluation(training_state, metrics)
            logging.info("step: %d", current_step)

            # Visualize trajectories every 1M steps
            if current_step // 1_000_000 > last_visualization_step // 1_000_000:
                key, viz_key = jax.random.split(key)
                buffer_state = all_visualizations(
                    replay_buffer=replay_buffer,
                    buffer_state=buffer_state,
                    env=unwrapped_env,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                    rng_key=viz_key,
                )
                _, phase_transitions = replay_buffer.sample(buffer_state)
                visualize_go_explore_phases(
                    phase_transitions,
                    unwrapped_env.x_bounds,
                    unwrapped_env.y_bounds,
                    state_size=state_size,
                    goal_indices=tuple(train_env.goal_indices),
                )
                last_visualization_step = current_step

            do_render = ne % config.visualization_interval == 0
            if self.agent_type == "crl":
                make_policy = lambda param: lambda obs, rng: gcp_actor.apply(param, obs)
            else:
                make_policy = lambda param: lambda obs, rng: (
                    gcp_actor.sample_actions(param, obs, rng, is_deterministic=True), {}
                )

            # Build full GCP critic params for checkpointing
            full_critic_params = {}
            for i, cs in enumerate(training_state.critic_states):
                for lname, lparams in cs.params.items():
                    if lname in ["sa_encoder", "g_encoder"]:
                        full_critic_params[f"{lname}_{i}"] = lparams
                    else:
                        full_critic_params[f"critic_{i}_{lname}"] = lparams

            params = (
                training_state.alpha_state.params,
                training_state.actor_state.params,
                full_critic_params,
            )

            if config.checkpoint_logdir:
                path = f"{config.checkpoint_logdir}/step_{int(training_state.env_steps)}.pkl"
                save_params(path, params)

            progress_fn(
                current_step,
                metrics,
                make_policy,
                training_state.actor_state.params,
                unwrapped_env,
                do_render=do_render,
            )

        total_steps = current_step
        logging.info("total steps: %s", total_steps)
        return make_policy, params, metrics
