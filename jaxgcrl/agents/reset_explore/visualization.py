from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import wandb

from jaxgcrl.agents.go_explore.visualization import handle_goal_proposer_visualization


_last_viz_env_steps_reset = -1


def plot_positions_scatter(
    positions: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    title: str,
    ax: plt.Axes,
    *,
    color_values: Optional[np.ndarray] = None,
    colorbar_label: str = "value",
    point_size: float = 18.0,
    cmap: str = "viridis",
) -> None:
    """Scatter of ``(x, y)`` points; color encodes ``color_values`` (one per point)."""
    xb = np.asarray(x_bounds, dtype=np.float64)
    yb = np.asarray(y_bounds, dtype=np.float64)
    pos = np.asarray(positions)
    ax.set_xlim(float(xb[0]), float(xb[1]))
    ax.set_ylim(float(yb[0]), float(yb[1]))
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("X position", fontsize=11)
    ax.set_ylabel("Y position", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    if pos.size == 0:
        return
    n = len(pos)
    if color_values is None:
        c = np.arange(n, dtype=np.float64)
        label = "point index"
    else:
        c = np.asarray(color_values, dtype=np.float64).reshape(-1)
        label = colorbar_label
    sc = ax.scatter(
        pos[:, 0],
        pos[:, 1],
        c=c,
        cmap=cmap,
        s=point_size,
        edgecolors="black",
        linewidths=0.35,
        alpha=0.9,
    )
    ax.get_figure().colorbar(
        sc, ax=ax, fraction=0.046, pad=0.04, label=label,
    )


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
    - Logs scatter panels for proposed goals and reset starts (color = env index).
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

    goals = np.asarray(goals_np)
    starts = np.asarray(starts_np)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    if goals.size == 0:
        init_goals_np = np.zeros((0, 2))
        dyn_goals_np = np.zeros((0, 2))
        idx_init = np.zeros((0,), dtype=np.int64)
        idx_dyn = np.zeros((0,), dtype=np.int64)
    else:
        init_goals_np = goals[init_mask]
        dyn_goals_np = goals[~init_mask]
        idx_init = np.where(init_mask)[0]
        idx_dyn = np.where(~init_mask)[0]

    if starts.size == 0:
        init_starts_np = np.zeros((0, 2))
        dyn_starts_np = np.zeros((0, 2))
    else:
        init_starts_np = starts[init_mask]
        dyn_starts_np = starts[~init_mask]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_positions_scatter(
        init_goals_np,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — initial resets (n={len(init_goals_np)})",
        ax=axes[0],
        color_values=idx_init.astype(np.float64),
        colorbar_label="env index",
    )
    plot_positions_scatter(
        dyn_goals_np,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — dynamic resets (n={len(dyn_goals_np)})",
        ax=axes[1],
        color_values=idx_dyn.astype(np.float64),
        colorbar_label="env index",
    )
    plot_positions_scatter(
        goals,
        x_bounds,
        y_bounds,
        title=f"Proposed goals — combined (n={len(goals)})",
        ax=axes[2],
        color_values=np.arange(len(goals), dtype=np.float64),
        colorbar_label="env index",
    )
    plt.tight_layout()
    wandb.log({"reset_explore/proposed_goals_heatmaps": wandb.Image(fig)})
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plot_positions_scatter(
        init_starts_np,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — initial resets (n={len(init_starts_np)})",
        ax=axes[0],
        color_values=idx_init.astype(np.float64),
        colorbar_label="env index",
    )
    plot_positions_scatter(
        dyn_starts_np,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — dynamic resets (n={len(dyn_starts_np)})",
        ax=axes[1],
        color_values=idx_dyn.astype(np.float64),
        colorbar_label="env index",
    )
    plot_positions_scatter(
        starts,
        x_bounds,
        y_bounds,
        title=f"Proposed reset starts — combined (n={len(starts)})",
        ax=axes[2],
        color_values=np.arange(len(starts), dtype=np.float64),
        colorbar_label="env index",
    )
    plt.tight_layout()
    wandb.log({"reset_explore/proposed_starts_heatmaps": wandb.Image(fig)})
    plt.close(fig)


# ── Fork mode (``ResetExplore`` with ``fork_type`` set) ─────────────────────

_last_fork_viz_env_steps: int = -1


def maybe_log_fork_state_redistribution(
    states_before: np.ndarray,
    states_after: np.ndarray,
    exploration_values: np.ndarray,
    env_steps: int,
    interval: int,
    x_bounds: Optional[np.ndarray] = None,
    y_bounds: Optional[np.ndarray] = None,
) -> None:
    """Log at most once per ``interval`` environment steps (fork path only)."""
    global _last_fork_viz_env_steps
    if interval <= 0:
        return
    e = int(env_steps)
    if _last_fork_viz_env_steps >= 0 and (e // interval) <= (_last_fork_viz_env_steps // interval):
        return
    _last_fork_viz_env_steps = e
    plot_fork_state_redistribution(
        states_before,
        states_after,
        exploration_values,
        env_steps=e,
        x_bounds=x_bounds,
        y_bounds=y_bounds,
    )


def plot_fork_state_redistribution(
    states_before: np.ndarray,
    states_after: np.ndarray,
    exploration_values: np.ndarray,
    env_steps: int,
    x_bounds: Optional[np.ndarray] = None,
    y_bounds: Optional[np.ndarray] = None,
) -> None:
    """Before / after scatter (env index color), plus after-state scatter by exploration value."""
    before = np.asarray(states_before)
    after = np.asarray(states_after)
    expl = np.asarray(exploration_values).reshape(-1)
    if before.ndim != 2 or after.ndim != 2:
        raise ValueError("states_before and states_after must be 2D arrays")
    if expl.shape[0] != after.shape[0]:
        raise ValueError("exploration_values length must match number of final states")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    if x_bounds is not None and y_bounds is not None:
        xb = np.asarray(x_bounds, dtype=np.float64)
        yb = np.asarray(y_bounds, dtype=np.float64)
        plot_positions_scatter(
            before,
            xb,
            yb,
            title="Before fork",
            ax=axes[0],
            color_values=np.arange(len(before), dtype=np.float64),
            colorbar_label="env index",
            point_size=22.0,
        )
        plot_positions_scatter(
            after,
            xb,
            yb,
            title="After fork",
            ax=axes[1],
            color_values=np.arange(len(after), dtype=np.float64),
            colorbar_label="env index",
            point_size=22.0,
        )
    else:
        n0, n1 = len(before), len(after)
        sc0 = axes[0].scatter(
            before[:, 0], before[:, 1], c=np.arange(n0), cmap="viridis",
            s=22, edgecolors="black", linewidths=0.35, alpha=0.9,
        )
        fig.colorbar(sc0, ax=axes[0], fraction=0.046, pad=0.04, label="env index")
        axes[0].set_title("Before fork")
        axes[0].set_xlabel("dim 0")
        axes[0].set_ylabel("dim 1")
        axes[0].grid(True, alpha=0.3)

        sc1 = axes[1].scatter(
            after[:, 0], after[:, 1], c=np.arange(n1), cmap="viridis",
            s=22, edgecolors="black", linewidths=0.35, alpha=0.9,
        )
        fig.colorbar(sc1, ax=axes[1], fraction=0.046, pad=0.04, label="env index")
        axes[1].set_title("After fork")
        axes[1].set_xlabel("dim 0")
        axes[1].set_ylabel("dim 1")
        axes[1].grid(True, alpha=0.3)

    ax3 = axes[2]
    x = after[:, 0].astype(np.float64)
    y = after[:, 1].astype(np.float64)
    sc = ax3.scatter(
        x, y, c=expl, cmap="viridis", s=28, edgecolors="black", linewidths=0.35, alpha=0.9,
    )
    fig.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04, label="Exploration value")
    ax3.set_title("After fork — exploration value")
    ax3.set_xlabel("X position")
    ax3.set_ylabel("Y position")
    ax3.grid(True, alpha=0.3)
    ax3.set_aspect("equal", adjustable="box")
    if x_bounds is not None and y_bounds is not None:
        ax3.set_xlim(float(x_bounds[0]), float(x_bounds[1]))
        ax3.set_ylim(float(y_bounds[0]), float(y_bounds[1]))

    fig.suptitle(f"Fork state redistribution (env steps ≈ {env_steps})", fontsize=12)
    plt.tight_layout()

    wandb.log({"reset_explore/fork_state_redistribution": wandb.Image(fig)}, step=int(env_steps))

    plt.close(fig)
