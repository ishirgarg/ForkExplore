import time

import jax
import jax.numpy as jp
import numpy as np
from brax import envs
from brax.envs.base import State, Wrapper
from brax.training import acting
from brax.training.types import Metrics, PolicyParams


class EvalWrapper(Wrapper):
    """Brax env with eval metrics."""

    def reset(self, rng: jax.Array) -> State:
        reset_state = self.env.reset(rng)
        reset_state.metrics['reward'] = reset_state.reward
        eval_metrics = envs.training.EvalMetrics(
            episode_metrics=jax.tree_util.tree_map(
                jp.zeros_like, reset_state.metrics
            ),
            active_episodes=jp.ones_like(reset_state.reward),
            episode_steps=jp.zeros_like(reset_state.reward),
        )
        reset_state.info['eval_metrics'] = eval_metrics
        return reset_state

    def step(self, state: State, action: jax.Array, rng=None) -> State:
        state_metrics = state.info['eval_metrics']
        if not isinstance(state_metrics, envs.training.EvalMetrics):
            raise ValueError(
                f'Incorrect type for state_metrics: {type(state_metrics)}'
            )
        del state.info['eval_metrics']
        nstate = self.env.step(state, action, rng)
        nstate.metrics['reward'] = nstate.reward
        episode_steps = jp.where(
            state_metrics.active_episodes,
            nstate.info['steps'],
            state_metrics.episode_steps,
        )
        episode_metrics = jax.tree_util.tree_map(
            lambda a, b: a + b * state_metrics.active_episodes,
            state_metrics.episode_metrics,
            nstate.metrics,
        )
        active_episodes = state_metrics.active_episodes * (1 - nstate.done)

        eval_metrics = envs.training.EvalMetrics(
            episode_metrics=episode_metrics,
            active_episodes=active_episodes,
            episode_steps=episode_steps,
        )
        nstate.info['eval_metrics'] = eval_metrics
        return nstate

# TODO: make only single type of Evaluator
class Evaluator(acting.Evaluator):
    """
    Class for running evaluation epochs in a training process.

    This class extends the base acting.Evaluator and is designed to evaluate the
    performance of policies based on given parameters and training metrics.
    It provides detailed evaluations by aggregating various metrics over episodes
    and calculating statistical summaries such as mean, standard deviation,
    maximum, and minimum values.

    This is an evaluator that behaves in the exact same way as brax Evaluator,
    but additionally it aggregates metrics with max, min.
    It also logs in how many episodes there was any success.
    """

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(policy_params, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            (np.std, "_std"),
            (np.max, "_max"),
            (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (fn(value) if aggregate_episodes else value)
                    for name, value in eval_metrics.episode_metrics.items()
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray


def generate_unroll(actor_step, training_state, env, env_state, unroll_length, extra_fields=()):
    """Collect trajectories of given unroll_length."""

    @jax.jit
    def f(carry, unused_t):
        state = carry
        nstate, transition = actor_step(training_state, env, state, extra_fields=extra_fields)
        return nstate, transition

    final_state, data = jax.lax.scan(f, env_state, (), length=unroll_length)
    return final_state, data


class ActorEvaluator:
    """Single GPU evaluator that evaluates an arbitrary actor function. Used by the CRL agent."""

    def __init__(self, actor_step, eval_env, num_eval_envs, episode_length, key):
        self._key = key
        self._eval_walltime = 0.0

        eval_env = EvalWrapper(eval_env)

        def generate_eval_unroll(training_state, key):
            reset_keys = jax.random.split(key, num_eval_envs)
            eval_first_state = eval_env.reset(reset_keys)
            return generate_unroll(
                actor_step,
                training_state,
                eval_env,
                eval_first_state,
                unroll_length=episode_length,
            )[0]

        self._generate_eval_unroll = jax.jit(generate_eval_unroll)
        self._steps_per_unroll = episode_length * num_eval_envs

    def run_evaluation(self, training_state, training_metrics, aggregate_episodes=True):
        """Run one epoch of evaluation."""
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(training_state, unroll_key)
        eval_metrics = eval_state.info["eval_metrics"]
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        aggregating_fns = [
            (np.mean, ""),
            # (np.std, "_std"),
            # (np.max, "_max"),
            # (np.min, "_min"),
        ]

        for fn, suffix in aggregating_fns:
            metrics.update(
                {
                    f"eval/episode_{name}{suffix}": (
                        fn(eval_metrics.episode_metrics[name])
                        if aggregate_episodes
                        else eval_metrics.episode_metrics[name]
                    )
                    for name in [
                        "reward",
                        "success",
                        "success_easy",
                        "dist",
                        "distance_from_origin",
                    ]
                }
            )

        # We check in how many env there was at least one step where there was success
        if "success" in eval_metrics.episode_metrics:
            metrics["eval/episode_success_any"] = np.mean(eval_metrics.episode_metrics["success"] > 0.0)

        metrics["eval/avg_episode_length"] = np.mean(eval_metrics.episode_steps)
        metrics["eval/epoch_eval_time"] = epoch_eval_time
        metrics["eval/sps"] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {"eval/walltime": self._eval_walltime, **training_metrics, **metrics}

        return metrics
