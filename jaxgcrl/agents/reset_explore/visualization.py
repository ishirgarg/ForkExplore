import numpy as np
import matplotlib.pyplot as plt
import wandb

from jaxgcrl.agents.go_explore.visualization import (
    plot_positions_with_heatmap,
    handle_goal_proposer_visualization,
)


_last_viz_env_steps_reset = -1


def handle_reset_explore_visualization(
    dyn_log_np: dict,
    init_log_np: dict,
    goals_np,
    starts_np,
    init_mask_np,
    viz_idx_np,
    goal_proposer_name: str,
    goal_proposer_name_initial: str,
    x_bounds,
    y_bounds,
    env_steps: int = -1,
):
    """
    Host-side visualization for ResetExplore proposals.
    - Dispatches candidate visualization to the correct proposer (initial vs dynamic)
      for one selected environment (viz_idx).
    - Logs three heatmaps of proposed goals: initial-only, dynamic-only, combined.
    """
    global _last_viz_env_steps_reset

    # Throttle like GoExplore: only once per 1M env steps
    if env_steps >= 0:
        if _last_viz_env_steps_reset >= 0 and (env_steps - _last_viz_env_steps_reset) < 1_000_000:
            return
        _last_viz_env_steps_reset = env_steps

    # Select which proposer generated the goal for the visualized env
    viz_idx = int(np.asarray(viz_idx_np))
    init_mask = np.asarray(init_mask_np).astype(bool)

    if init_mask[viz_idx]:
        selected = {k: v[viz_idx] for k, v in init_log_np.items()}
        gp_name = goal_proposer_name_initial
    else:
        selected = {k: v[viz_idx] for k, v in dyn_log_np.items()}
        gp_name = goal_proposer_name

    handle_goal_proposer_visualization(
        selected,
        gp_name,
        x_bounds,
        y_bounds,
        env_steps=-1,
    )

    # Build three heatmaps for proposed goals and proposed reset starts
    goals = np.asarray(goals_np)
    starts = np.asarray(starts_np)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    if goals.size == 0:
        init_goals_np = np.zeros((0, 2))
        dyn_goals_np = np.zeros((0, 2))
    else:
        init_goals_np = goals[init_mask]
        dyn_goals_np = goals[~init_mask]

    if starts.size == 0:
        init_starts_np = np.zeros((0, 2))
        dyn_starts_np = np.zeros((0, 2))
    else:
        init_starts_np = starts[init_mask]
        dyn_starts_np = starts[~init_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_positions_with_heatmap(
        init_goals_np,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — initial resets (n={len(init_goals_np)})",
        ax=axes[0],
        alpha_points=0.4,
        alpha_heatmap=0.55,
        point_size=2.0,
    )
    plot_positions_with_heatmap(
        dyn_goals_np,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — dynamic resets (n={len(dyn_goals_np)})",
        ax=axes[1],
        alpha_points=0.4,
        alpha_heatmap=0.55,
        point_size=2.0,
    )
    plot_positions_with_heatmap(
        goals,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — combined (n={len(goals)})",
        ax=axes[2],
        alpha_points=0.25,
        alpha_heatmap=0.45,
        point_size=1.5,
    )
    plt.tight_layout()
    wandb.log({"reset_explore/proposed_goals_heatmaps": wandb.Image(fig)})
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_positions_with_heatmap(
        init_starts_np,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — initial resets (n={len(init_starts_np)})",
        ax=axes[0],
        alpha_points=0.4,
        alpha_heatmap=0.55,
        point_size=2.0,
    )
    plot_positions_with_heatmap(
        dyn_starts_np,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — dynamic resets (n={len(dyn_starts_np)})",
        ax=axes[1],
        alpha_points=0.4,
        alpha_heatmap=0.55,
        point_size=2.0,
    )
    plot_positions_with_heatmap(
        starts,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — combined (n={len(starts)})",
        ax=axes[2],
        alpha_points=0.25,
        alpha_heatmap=0.45,
        point_size=1.5,
    )
    plt.tight_layout()
    wandb.log({"reset_explore/proposed_starts_heatmaps": wandb.Image(fig)})
    plt.close(fig)
