"""Visualization functions for trajectory analysis."""

import logging
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import wandb
from .utils import sample_trajectories_from_buffer, sample_trajectory_sequences


def create_kde_heatmap(
    positions: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    grid_resolution: int = 100,
    bandwidth: float = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a KDE heatmap for 2D positions.
    
    Args:
        positions: (N, 2) array of [x, y] positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        grid_resolution: Resolution of the heatmap grid
        bandwidth: Bandwidth for KDE (None for automatic)
        
    Returns:
        Tuple of (X, Y, Z) where X, Y are meshgrids and Z is the density
    """
    if len(positions) == 0:
        # Return empty heatmap
        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        x = np.linspace(x_min, x_max, grid_resolution)
        y = np.linspace(y_min, y_max, grid_resolution)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        return X, Y, Z
    
    # Convert bounds to numpy
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Create grid
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute KDE
    try:
        kde = gaussian_kde(positions.T, bw_method=bandwidth)
        positions_grid = np.vstack([X.ravel(), Y.ravel()])
        Z = kde(positions_grid).reshape(X.shape)
    except (np.linalg.LinAlgError, ValueError):
        # Fallback if KDE fails (e.g., not enough points or singular matrix)
        Z = np.zeros_like(X)
    
    return X, Y, Z


def plot_positions_with_heatmap(
    positions: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    title: str,
    ax: plt.Axes = None,
    alpha_points: float = 0.3,
    alpha_heatmap: float = 0.5,
    point_size: float = 1.0,
    grid_resolution: int = 100,
) -> plt.Axes:
    """
    Plot positions as scatter points with KDE heatmap overlay.
    
    Args:
        positions: (N, 2) array of [x, y] positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        title: Plot title
        ax: Matplotlib axes (creates new if None)
        alpha_points: Transparency for scatter points
        alpha_heatmap: Transparency for heatmap
        point_size: Size of scatter points
        grid_resolution: Resolution of the heatmap grid
        
    Returns:
        Matplotlib axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Create KDE heatmap
    X, Y, Z = create_kde_heatmap(positions, x_bounds, y_bounds, grid_resolution)
    
    # Plot heatmap
    if np.any(Z > 0):
        ax.contourf(X, Y, Z, levels=20, alpha=alpha_heatmap, cmap='viridis')
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
    
    # Plot scatter points
    if len(positions) > 0:
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=point_size,
            alpha=alpha_points,
            c='red',
            edgecolors='darkred',
            linewidths=0.5,
        )
    
    # Set bounds and labels
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    return ax


def plot_trajectory_sequences(
    trajectory_states: np.ndarray,
    trajectory_goals: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    fig: plt.Figure = None,
) -> plt.Figure:
    """
    Plot trajectory sequences in a 2x2 grid, showing start, intermediate states, final state, and goal.
    
    Args:
        trajectory_states: (num_trajectories, 8, 2) array of [x, y] positions
                         [start, 6 intermediate states, final]
        trajectory_goals: (num_trajectories, 2) array of [x, y] goal positions
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        fig: Matplotlib figure (creates new if None)
        
    Returns:
        Matplotlib figure
    """
    if fig is None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    else:
        # Extract axes from existing figure if it has a 2x2 grid
        axes = fig.subplots(2, 2) if len(fig.axes) == 0 else np.array(fig.axes).reshape(2, 2)
    
    if len(trajectory_states) == 0:
        return fig
    
    # Colors for different trajectories
    colors = ['blue', 'green', 'red', 'purple']
    
    # Set bounds
    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Plot each trajectory in its own subplot
    num_trajectories = min(len(trajectory_states), 4)
    for i in range(num_trajectories):
        row = i // 2
        col = i % 2
        ax = axes[row, col]
        
        states = trajectory_states[i]
        goal = trajectory_goals[i]
        color = colors[i % len(colors)]
        
        # Plot trajectory path: start -> intermediate -> final
        # States shape: (8, 2) = [start, 6 intermediate, final]
        ax.plot(states[:, 0], states[:, 1], 'o-', color=color, 
                linewidth=2, markersize=6, alpha=0.7)
        
        # Mark start state
        ax.plot(states[0, 0], states[0, 1], 'o', color=color, 
                markersize=10, markeredgecolor='black', markeredgewidth=2, label='Start')
        
        # Mark final state
        ax.plot(states[-1, 0], states[-1, 1], 's', color=color, 
                markersize=10, markeredgecolor='black', markeredgewidth=2, label='Final')
        
        # Plot line from final state to goal (different style)
        ax.plot([states[-1, 0], goal[0]], [states[-1, 1], goal[1]], 
                '--', color=color, linewidth=2, alpha=0.5, label='To Goal')
        
        # Mark goal
        ax.plot(goal[0], goal[1], '*', color=color, 
                markersize=15, markeredgecolor='black', markeredgewidth=1, label='Goal')
        
        # Set bounds and labels for each subplot
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel('X Position', fontsize=10)
        ax.set_ylabel('Y Position', fontsize=10)
        ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='best', fontsize=8)
    
    # Hide unused subplots if we have fewer than 4 trajectories
    for i in range(num_trajectories, 4):
        row = i // 2
        col = i % 2
        axes[row, col].axis('off')
    
    return fig


def visualize_trajectories(
    all_positions: np.ndarray,
    final_positions: np.ndarray,
    goal_positions: np.ndarray,
    trajectory_states: np.ndarray,
    trajectory_goals: np.ndarray,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    save_path: str = None,
    figsize: Tuple[int, int] = (24, 16),
) -> plt.Figure:
    """
    Create visualization with three plots in first row and 2x2 grid of trajectories in second row.
    
    Args:
        all_positions: (N, 2) array of [x, y] positions from all states
        final_positions: (M, 2) array of [x, y] positions from final states
        goal_positions: (N, 2) array of [x, y] goal positions from all observations
        trajectory_states: (num_trajectories, 8, 2) array of trajectory sequences
        trajectory_goals: (num_trajectories, 2) array of goal positions for trajectories
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
        save_path: Path to save the figure (optional)
        figsize: Figure size (width, height)
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # First row: 3 plots (all states, final states, goals)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Plot 1: All states
    num_all_points = len(all_positions)
    plot_positions_with_heatmap(
        all_positions,
        x_bounds,
        y_bounds,
        title=f'All States in Trajectories (n={num_all_points})',
        ax=ax1,
        alpha_points=0.2,
        alpha_heatmap=0.4,
        point_size=0.5,
    )
    
    # Plot 2: Final states
    num_final_points = len(final_positions)
    plot_positions_with_heatmap(
        final_positions,
        x_bounds,
        y_bounds,
        title=f'Final States of Trajectories (n={num_final_points})',
        ax=ax2,
        alpha_points=0.4,
        alpha_heatmap=0.5,
        point_size=2.0,
    )
    
    # Plot 3: Goals
    num_goal_points = len(goal_positions)
    plot_positions_with_heatmap(
        goal_positions,
        x_bounds,
        y_bounds,
        title=f'Goals in Trajectories (n={num_goal_points})',
        ax=ax3,
        alpha_points=0.3,
        alpha_heatmap=0.5,
        point_size=1.0,
    )
    
    # Second row: 2x2 grid for trajectory sequences (spans columns 0-1)
    traj_gs = gs[1, :2].subgridspec(2, 2, hspace=0.3, wspace=0.3)
    traj_axes = []
    for i in range(2):
        row = []
        for j in range(2):
            row.append(fig.add_subplot(traj_gs[i, j]))
        traj_axes.append(row)
    traj_axes = np.array(traj_axes)
    
    # Plot trajectories in 2x2 grid
    if len(trajectory_states) > 0:
        x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
        y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
        
        colors = ['blue', 'green', 'red', 'purple']
        num_trajectories = min(len(trajectory_states), 4)
        
        for i in range(num_trajectories):
            row = i // 2
            col = i % 2
            ax = traj_axes[row, col]
            
            states = trajectory_states[i]
            goal = trajectory_goals[i]
            color = colors[i % len(colors)]
            
            # Plot trajectory path: start -> intermediate -> final
            ax.plot(states[:, 0], states[:, 1], 'o-', color=color, 
                    linewidth=2, markersize=6, alpha=0.7)
            
            # Mark start state
            ax.plot(states[0, 0], states[0, 1], 'o', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=2, label='Start')
            
            # Mark final state
            ax.plot(states[-1, 0], states[-1, 1], 's', color=color, 
                    markersize=10, markeredgecolor='black', markeredgewidth=2, label='Final')
            
            # Plot line from final state to goal (different style)
            ax.plot([states[-1, 0], goal[0]], [states[-1, 1], goal[1]], 
                    '--', color=color, linewidth=2, alpha=0.5, label='To Goal')
            
            # Mark goal
            ax.plot(goal[0], goal[1], '*', color=color, 
                    markersize=15, markeredgecolor='black', markeredgewidth=1, label='Goal')
            
            # Set bounds and labels for each subplot
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X Position', fontsize=10)
            ax.set_ylabel('Y Position', fontsize=10)
            ax.set_title(f'Trajectory {i+1}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='box')
            ax.legend(loc='best', fontsize=8)
        
        # Hide unused subplots if we have fewer than 4 trajectories
        for i in range(num_trajectories, 4):
            row = i // 2
            col = i % 2
            traj_axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    return fig


def all_visualizations(
    replay_buffer: Any,
    buffer_state: Any,
    env: Any,
    state_size: int,
    goal_indices: Tuple[int, ...],
    rng_key: jax.Array,
) -> Any:
    """
    Create all trajectory visualizations from the replay buffer.
    
    This function handles sampling trajectories, creating visualizations,
    and logging to wandb.
    
    Args:
        replay_buffer: The replay buffer instance
        buffer_state: Current buffer state
        env: Environment instance (must have x_bounds and y_bounds attributes)
        state_size: Size of state dimension
        goal_indices: Indices for x, y positions (typically [0, 1])
        rng_key: Random key for sampling
        
    Returns:
        Updated buffer_state after sampling operations
    """
    # Check if environment has bounds (maze environments)
    if not (hasattr(env, 'x_bounds') and hasattr(env, 'y_bounds')):
        return buffer_state
    
    # Sample trajectories from buffer
    buffer_size = replay_buffer.size(buffer_state)
    if buffer_size == 0:
        return buffer_state
    
    buffer_state, all_positions, final_positions, goal_positions = sample_trajectories_from_buffer(
        replay_buffer,
        buffer_state,
        state_size=state_size,
        goal_indices=goal_indices,
        rng_key=rng_key,
    )
    
    # Sample trajectory sequences for detailed plotting
    buffer_state, trajectory_states, trajectory_goals = sample_trajectory_sequences(
        replay_buffer,
        buffer_state,
        state_size=state_size,
        goal_indices=goal_indices,
        rng_key=rng_key,
        num_trajectories=4,
    )
    
    # Only visualize if we have data
    if len(all_positions) == 0 and len(final_positions) == 0 and len(goal_positions) == 0:
        return buffer_state
    
    # Create visualization (don't save to file)
    fig = visualize_trajectories(
        all_positions,
        final_positions,
        goal_positions,
        trajectory_states,
        trajectory_goals,
        env.x_bounds,
        env.y_bounds,
        save_path=None,
    )
    
    wandb.log({"trajectory_visualization": wandb.Image(fig)})
    plt.close(fig)

    return buffer_state


def visualize_go_explore_phases(
    transitions: Any,
    x_bounds: jnp.ndarray,
    y_bounds: jnp.ndarray,
    state_size: int,
    goal_indices: Tuple[int, ...],
    num_pairs: int = 4,
) -> None:
    # ── 1. Flatten buffer sample ──────────────────────────────────────────────
    obs_flat      = np.array(jnp.reshape(transitions.observation,
                                         (-1, transitions.observation.shape[-1])))
    phase_flat    = np.array(jnp.reshape(transitions.extras["state_extras"]["phase"], (-1,)))
    traj_id_flat  = np.array(jnp.reshape(transitions.extras["state_extras"]["traj_id"], (-1,)))
    trunc_flat    = np.array(jnp.reshape(transitions.extras["state_extras"]["truncation"], (-1,)))

    goal_idx_list = list(goal_indices)
    positions     = obs_flat[:, :state_size][:, goal_idx_list]        # (N, 2)
    goals         = obs_flat[:, state_size:][:, :len(goal_idx_list)]  # (N, 2)

    go_mask      = phase_flat == 0
    explore_mask = phase_flat == 1

    go_pos      = positions[go_mask]
    explore_pos = positions[explore_mask]
    go_goals    = goals[go_mask]

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # ── 2. Find paired (go traj N, explore traj N+1) ─────────────────────────
    # GoExploreWrapper increments traj_id by +1 on go→explore, so the explore
    # sub-trajectory immediately following go traj N has traj_id = N+1.
    def _traj_positions(traj_id):
        mask = traj_id_flat == traj_id
        return positions[mask], trunc_flat[mask]

    go_traj_ids = np.unique(traj_id_flat[go_mask])
    pairs = []          # list of (go_positions, explore_positions, go_goal)
    for go_tid in go_traj_ids:
        exp_tid = go_tid + 1.0
        if not np.any(traj_id_flat == exp_tid):
            continue
        gp, gtrunc = _traj_positions(go_tid)
        ep, _      = _traj_positions(exp_tid)
        if len(gp) == 0 or len(ep) == 0:
            continue
        # Goal is constant within go traj; take the first
        gg = goals[go_mask][traj_id_flat[go_mask] == go_tid][0]
        pairs.append((gp, ep, gg))
        if len(pairs) >= num_pairs:
            break

    # ── 3. Build figure ───────────────────────────────────────────────────────
    n_detail = max(1, len(pairs))
    fig = plt.figure(figsize=(18, 6 + 5 * ((n_detail + 1) // 2)))
    top_gs   = fig.add_gridspec(2, 1, hspace=0.45)
    top_row  = top_gs[0].subgridspec(1, 3, wspace=0.3)
    bot_row  = top_gs[1].subgridspec((n_detail + 1) // 2, 2, hspace=0.4, wspace=0.3)

    def _setup(ax, title):
        ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.3)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("X"); ax.set_ylabel("Y")

    # ── Row 0, col 0: go-phase positions with goal markers ───────────────────
    ax0 = fig.add_subplot(top_row[0, 0])
    if len(go_pos) > 0:
        ax0.scatter(go_pos[:, 0], go_pos[:, 1],
                    s=1, alpha=0.3, c="steelblue", label="Go states")
        # Sub-sample goals to avoid overplotting
        step = max(1, len(go_goals) // 200)
        ax0.scatter(go_goals[::step, 0], go_goals[::step, 1],
                    s=15, alpha=0.5, c="gold", marker="*",
                    edgecolors="darkorange", linewidths=0.4, label="Goals", zorder=5)
    _setup(ax0, f"Go phase — states (n={len(go_pos)})")
    ax0.legend(fontsize=7, loc="upper right")

    # ── Row 0, col 1: explore-phase KDE heatmap ──────────────────────────────
    ax1 = fig.add_subplot(top_row[0, 1])
    plot_positions_with_heatmap(
        explore_pos, x_bounds, y_bounds,
        title=f"Explore phase — states (n={len(explore_pos)})",
        ax=ax1, alpha_points=0.2, alpha_heatmap=0.55, point_size=0.5,
    )

    # ── Row 0, col 2: explore reach vs go goal (final explore pos per pair) ──
    ax2 = fig.add_subplot(top_row[0, 2])
    if pairs:
        final_exp  = np.array([p[1][-1] for p in pairs])
        pair_goals = np.array([p[2] for p in pairs])
        ax2.scatter(pair_goals[:, 0], pair_goals[:, 1],
                    s=60, c="gold", marker="*",
                    edgecolors="darkorange", linewidths=0.8,
                    label="Go goal", zorder=6)
        ax2.scatter(final_exp[:, 0], final_exp[:, 1],
                    s=40, c="tomato", marker="D",
                    edgecolors="darkred", linewidths=0.8,
                    label="Explore end", zorder=6)
        for gp_pt, ep_pt in zip(pair_goals, final_exp):
            ax2.annotate("", xy=ep_pt, xytext=gp_pt,
                         arrowprops=dict(arrowstyle="->", color="gray",
                                         lw=0.8, alpha=0.6))
    _setup(ax2, "Explore reach vs. go goal")
    ax2.legend(fontsize=7, loc="upper right")

    # ── Row 1: paired trajectory detail ──────────────────────────────────────
    colors_go      = plt.cm.Blues(np.linspace(0.55, 0.9, max(len(pairs), 1)))
    colors_explore = plt.cm.Oranges(np.linspace(0.55, 0.9, max(len(pairs), 1)))

    for k, (gp, ep, gg) in enumerate(pairs):
        row_k, col_k = k // 2, k % 2
        ax = fig.add_subplot(bot_row[row_k, col_k])

        # Go-phase path
        ax.plot(gp[:, 0], gp[:, 1], "-o", color=colors_go[k],
                linewidth=1.5, markersize=3, alpha=0.8, label="Go")
        ax.plot(*gp[0],  "o", color=colors_go[k],
                markersize=8, markeredgecolor="black", markeredgewidth=1.2)

        # Explore-phase path
        ax.plot(ep[:, 0], ep[:, 1], "-o", color=colors_explore[k],
                linewidth=1.5, markersize=3, alpha=0.8, label="Explore")
        ax.plot(*ep[-1], "s", color=colors_explore[k],
                markersize=8, markeredgecolor="black", markeredgewidth=1.2)

        # Goal
        ax.plot(*gg, "*", color="gold", markersize=12,
                markeredgecolor="darkorange", markeredgewidth=1, label="Goal", zorder=7)

        # Dashed connector: final go state → proposed goal (shows residual gap)
        ax.plot([gp[-1, 0], gg[0]], [gp[-1, 1], gg[1]],
                "--", color="gray", linewidth=1, alpha=0.6)

        # Title: show which go traj_id this pair belongs to
        go_tid_label = int(go_traj_ids[k]) if k < len(go_traj_ids) else k
        _setup(ax, f"Pair {k + 1}  (go traj_id={go_tid_label})")
        ax.legend(fontsize=7, loc="upper right")

    # Hide any unused bot-row slots
    total_slots = ((n_detail + 1) // 2) * 2
    for k in range(len(pairs), total_slots):
        try:
            fig.add_subplot(bot_row[k // 2, k % 2]).axis("off")
        except Exception:
            pass

    plt.suptitle("Go Explore — Phase Breakdown", fontsize=13, fontweight="bold", y=1.01)
    wandb.log({"go_explore_phase_viz": wandb.Image(fig)})
    plt.close(fig)


# Module-level variable to track last visualized env_steps (for go_explore only)
_last_viz_env_steps = -1

# Separate tracker for fork visualizations so it throttles independently.
_last_fork_viz_env_steps = -1
_last_fork_grid_viz_env_steps = -1
_FORK_VIZ_INTERVAL = 500_000


def visualize_fork_grid(
    positions: np.ndarray,
    scores: np.ndarray,
    top_k_mask: np.ndarray,
    bottom_k_mask: np.ndarray,
    go_mask: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    env_steps: int = -1,
) -> None:
    """Scatter of all envs coloured by exploration score, with distinguishing
    outlines for top-k (lime), bottom-k (orange), and go-phase (red) envs."""
    global _last_fork_grid_viz_env_steps
    if env_steps >= 0:
        if (_last_fork_grid_viz_env_steps >= 0
                and (env_steps - _last_fork_grid_viz_env_steps) < _FORK_VIZ_INTERVAL):
            return
        _last_fork_grid_viz_env_steps = env_steps

    positions = np.asarray(positions)
    scores = np.asarray(scores)
    top_k_mask = np.asarray(top_k_mask).astype(bool)
    bottom_k_mask = np.asarray(bottom_k_mask).astype(bool)
    go_mask = np.asarray(go_mask).astype(bool)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    finite_mask = np.isfinite(scores)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Explore-phase envs: coloured by exploration score.
    if np.any(finite_mask):
        sc = ax.scatter(
            positions[finite_mask, 0], positions[finite_mask, 1],
            c=scores[finite_mask], cmap="viridis",
            s=140, alpha=0.9,
            edgecolors="black", linewidths=0.4,
            zorder=3,
        )
        plt.colorbar(sc, ax=ax, label="Exploration score")

    # Go-phase envs: greyed fill (no score).
    if np.any(go_mask):
        ax.scatter(
            positions[go_mask, 0], positions[go_mask, 1],
            c="lightgrey", s=140, alpha=0.6,
            zorder=2,
        )

    # Outlines layered above fills.
    if np.any(go_mask):
        ax.scatter(
            positions[go_mask, 0], positions[go_mask, 1],
            facecolors="none", edgecolors="red",
            s=260, linewidths=2.2,
            label=f"Go phase (n={int(go_mask.sum())})", zorder=5,
        )
    if np.any(top_k_mask):
        ax.scatter(
            positions[top_k_mask, 0], positions[top_k_mask, 1],
            facecolors="none", edgecolors="lime",
            s=380, linewidths=3.0,
            label=f"Top-k (n={int(top_k_mask.sum())})", zorder=6,
        )
    if np.any(bottom_k_mask):
        ax.scatter(
            positions[bottom_k_mask, 0], positions[bottom_k_mask, 1],
            facecolors="none", edgecolors="darkorange",
            s=380, linewidths=3.0,
            label=f"Bottom-k (n={int(bottom_k_mask.sum())})", zorder=6,
        )

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title(
        "Fork grid — exploration scores + top-k / bottom-k / go outlines",
        fontsize=13, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    wandb.log({"fork_grid": wandb.Image(fig)})
    plt.close(fig)


def visualize_fork_scatter_kde(
    pre_positions: np.ndarray,
    post_positions: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """Two-panel scatter+KDE of explore-phase positions before and after a fork event."""
    pre_positions = np.asarray(pre_positions)
    post_positions = np.asarray(post_positions)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plot_positions_with_heatmap(
        pre_positions, x_bounds, y_bounds,
        title="Fork: Explore States (Pre-Resample)", ax=ax1, point_size=20.0, alpha_points=0.7,
    )
    plot_positions_with_heatmap(
        post_positions, x_bounds, y_bounds,
        title="Fork: Explore States (Post-Resample)", ax=ax2, point_size=20.0, alpha_points=0.7,
    )
    plt.tight_layout()
    wandb.log({"fork_scatter_kde": wandb.Image(fig)})
    plt.close(fig)


def visualize_fork_scores(
    positions: np.ndarray,
    scores: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """Single-panel scatter of explore-phase positions, colored by fork score."""
    positions = np.asarray(positions)
    scores = np.asarray(scores)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    fig, ax = plt.subplots(figsize=(10, 10))
    if len(positions) > 0:
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            c=scores,
            cmap="viridis",
            s=60,
            alpha=0.85,
            edgecolors="black",
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label="Fork score")

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Position", fontsize=12)
    ax.set_ylabel("Y Position", fontsize=12)
    ax.set_title("Fork: Explore States Colored by Score", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    wandb.log({"fork_scores": wandb.Image(fig)})
    plt.close(fig)


def handle_fork_visualization(
    pre_positions: np.ndarray,
    post_positions: np.ndarray,
    scores: np.ndarray,
    in_explore: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    env_steps: int,
) -> None:
    """Throttled dispatcher for fork-related plots.

    Only logs when ``env_steps`` has advanced by at least ``_FORK_VIZ_INTERVAL``
    since the previous call.  Steps can jump by more than 1 per training
    iteration, so we use an accumulator instead of a modulo check.
    """
    global _last_fork_viz_env_steps

    if _last_fork_viz_env_steps >= 0 and (env_steps - _last_fork_viz_env_steps) < _FORK_VIZ_INTERVAL:
        return
    _last_fork_viz_env_steps = env_steps

    in_explore = np.asarray(in_explore).astype(bool)
    if not np.any(in_explore):
        return

    pre_positions = np.asarray(pre_positions)[in_explore]
    post_positions = np.asarray(post_positions)[in_explore]
    scores = np.asarray(scores)[in_explore]

    visualize_fork_scatter_kde(pre_positions, post_positions, x_bounds, y_bounds)
    visualize_fork_scores(pre_positions, scores, x_bounds, y_bounds)


def handle_goal_proposer_visualization(
    log_data: dict,
    goal_proposer_name: str,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    env_steps: int = -1,
) -> None:
    """
    Generic handler for goal proposer visualization.
    Dispatches to appropriate visualization function based on goal_proposer_name.
    
    Args:
        log_data: Dictionary with visualization data from goal proposer
        goal_proposer_name: Name of the goal proposer (e.g., "q_epistemic", "rb")
        x_bounds: Environment x bounds [x_min, x_max]
        y_bounds: Environment y bounds [y_min, y_max]
        env_steps: Current environment steps (for go_explore: only visualize if >= 1M steps since last)
    """
    global _last_viz_env_steps
    
    if not log_data:  # Empty dict means no visualization
        return
    
    # For go_explore: only visualize if env_steps is provided and it's been >= 1M steps since last viz
    if env_steps >= 0:
        if _last_viz_env_steps >= 0 and (env_steps - _last_viz_env_steps) < 1_000_000:
            return
        _last_viz_env_steps = env_steps
    
    if goal_proposer_name == "q_epistemic":
        # Extract data for q_epistemic visualization
        candidate_goals = log_data["candidate_goals"]
        first_obs_position = log_data["first_obs_position"]
        q_means = log_data["q_means"]
        q_stds = log_data["q_stds"]
        selected_goal = log_data.get("selected_goal")

        visualize_q_epistemic_candidates(
            candidate_goals,
            first_obs_position,
            q_means,
            q_stds,
            x_bounds,
            y_bounds,
            selected_goal,
        )
    elif goal_proposer_name == "ucgr":
        visualize_ucgr_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            minlse_scores=log_data["minlse_scores"],
            selected_goal=log_data["selected_goal"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    elif goal_proposer_name == "max_critic_to_env":
        visualize_max_critic_to_env_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            q_means=log_data["q_means"],
            env_goal=log_data["selected_goal"],
            selected_state_goal=log_data["selected_state_goal"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    elif goal_proposer_name == "mega":
        visualize_mega_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            densities=log_data["densities"],
            selected_goal=log_data["selected_goal"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            title_prefix="MEGA",
        )
    elif goal_proposer_name == "omega":
        visualize_omega_candidates(
            candidate_goals=log_data["candidate_goals"],
            first_obs_position=log_data["first_obs_position"],
            densities=log_data["densities"],
            selected_goal=log_data["selected_goal"],
            mega_goal=log_data["mega_goal"],
            env_goal=log_data["env_goal"],
            alpha=log_data["alpha"],
            kl_div=log_data["kl_div"],
            x_bounds=x_bounds,
            y_bounds=y_bounds,
        )
    # Add more goal proposer visualizations here as needed


def visualize_q_epistemic_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    q_means: np.ndarray,
    q_stds: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    selected_goal: np.ndarray,
) -> None:
    """
    Visualize Q-epistemic goal proposer candidates with mean and std Q-values.
    
    Creates two plots side by side:
    - Left: candidate states and first observation colored by mean Q-value
    - Right: candidate states and first observation colored by std Q-value
    
    Args:
        candidate_goals: (num_candidates, 2) array of [x, y] candidate goal positions
        first_obs_position: (2,) array of [x, y] position from first observation
        q_means: (num_candidates,) array of mean Q-values for each candidate
        q_stds: (num_candidates,) array of std Q-values for each candidate
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Ensure numpy arrays
    candidate_goals = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    q_means = np.asarray(q_means)
    q_stds = np.asarray(q_stds)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)
    selected_goal = np.asarray(selected_goal)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Left plot: colored by mean Q-value
    if len(candidate_goals) > 0:
        scatter1 = ax1.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_means,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter1, ax=ax1, label='Mean Q-value')
    
    # Plot first observation position
    ax1.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('X Position', fontsize=12)
    ax1.set_ylabel('Y Position', fontsize=12)
    ax1.set_title('Q-Epistemic Candidates (Mean Q-value)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')
    ax1.legend()
    
    ax1.scatter(
        selected_goal[0],
        selected_goal[1],
        c="lime",
        s=150,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Selected Goal",
        zorder=11,
    )

    # Right plot: colored by std Q-value
    if len(candidate_goals) > 0:
        scatter2 = ax2.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_stds,
            cmap='plasma',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter2, ax=ax2, label='Std Q-value')
    
    # Plot first observation position
    ax2.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    # Optionally plot selected goal on right subplot
    if selected_goal is not None and selected_goal.size > 0:
        ax2.scatter(
            selected_goal[0],
            selected_goal[1],
            c="lime",
            s=150,
            marker="X",
            edgecolors="black",
            linewidths=1.5,
            label="Selected Goal",
            zorder=11,
        )

    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(y_min, y_max)
    ax2.set_xlabel('X Position', fontsize=12)
    ax2.set_ylabel('Y Position', fontsize=12)
    ax2.set_title('Q-Epistemic Candidates (Std Q-value)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"q_epistemic_candidates": wandb.Image(fig)})
    plt.close(fig)


def visualize_max_critic_to_env_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    q_means: np.ndarray,
    env_goal: np.ndarray,
    selected_state_goal: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """
    Visualize max_critic_to_env goal proposer candidates colored by mean Q-value.
    
    Shows:
    - Candidate states colored by their mean Q-value (with the random env goal g)
    - The random environment goal g (selected goal)
    - The state that maximizes Q(w, g) (selected_state_goal)
    - The first observation position
    
    Args:
        candidate_goals: (num_candidates, 2) array of [x, y] candidate goal positions
        first_obs_position: (2,) array of [x, y] position from first observation
        q_means: (num_candidates,) array of mean Q-values for each candidate
        env_goal: (2,) array of [x, y] position of the random environment goal g
        selected_state_goal: (2,) array of [x, y] position of the state that maximizes Q
        x_bounds: [x_min, x_max] bounds for x-axis
        y_bounds: [y_min, y_max] bounds for y-axis
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Ensure numpy arrays
    candidate_goals = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    q_means = np.asarray(q_means)
    env_goal = np.asarray(env_goal)
    selected_state_goal = np.asarray(selected_state_goal)
    x_bounds = np.asarray(x_bounds)
    y_bounds = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])
    
    # Plot candidate states colored by mean Q-value
    if len(candidate_goals) > 0:
        scatter = ax.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=q_means,
            cmap='viridis',
            s=50,
            alpha=0.7,
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax, label='Mean Q-value')
    
    # Plot first observation position
    ax.scatter(
        first_obs_position[0],
        first_obs_position[1],
        c='red',
        s=100,
        marker='*',
        edgecolors='black',
        linewidths=1.5,
        label='First Observation',
        zorder=10,
    )
    
    # Plot the random environment goal g (selected goal)
    ax.scatter(
        env_goal[0],
        env_goal[1],
        c="lime",
        s=150,
        marker="X",
        edgecolors="black",
        linewidths=1.5,
        label="Selected Goal (Env Goal g)",
        zorder=11,
    )
    
    # Plot the state that maximizes Q(w, g)
    ax.scatter(
        selected_state_goal[0],
        selected_state_goal[1],
        c="orange",
        s=150,
        marker="D",
        edgecolors="black",
        linewidths=1.5,
        label="Max Q State w",
        zorder=11,
    )
    
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel('X Position', fontsize=12)
    ax.set_ylabel('Y Position', fontsize=12)
    ax.set_title('Max Critic to Env: Candidates Colored by Q(w, g)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({"max_critic_to_env_candidates": wandb.Image(fig)})
    plt.close(fig)


def visualize_ucgr_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    minlse_scores: np.ndarray,
    selected_goal: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """
    Visualize UCGR goal-proposer candidates coloured by their MinLSE score.

    Two side-by-side panels:
    - Left:  candidates coloured by raw MinLSE score (lower = harder = frontier).
             Cool colours → hard; warm colours → easy.  Selected goal marked in lime.
    - Right: candidates coloured by normalised *difficulty* (1 − normalised score),
             so the selected (hardest) goal is the brightest point.  Useful for
             seeing at a glance where the frontier of the agent's capability lies.

    Both panels also show the agent's start position as a red star.

    Args:
        candidate_goals:    (num_candidates, 2) [x, y] positions of candidate goals.
        first_obs_position: (2,) [x, y] of the agent's start position this step.
        minlse_scores:      (num_candidates,) MinLSE reachability score per candidate.
                            Lower score → harder to reach → more useful for training.
        selected_goal:      (2,) [x, y] of the chosen (lowest-score) goal.
        x_bounds:           [x_min, x_max] for the environment.
        y_bounds:           [y_min, y_max] for the environment.
    """
    # ── Ensure plain numpy ───────────────────────────────────────────────────
    candidate_goals    = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    minlse_scores      = np.asarray(minlse_scores)
    selected_goal      = np.asarray(selected_goal)
    x_bounds           = np.asarray(x_bounds)
    y_bounds           = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # ── Normalised difficulty (inverted score, 0-1) ──────────────────────────
    score_min, score_max = minlse_scores.min(), minlse_scores.max()
    score_range = score_max - score_min
    if score_range > 0:
        # difficulty ∈ [0, 1]; 1 = hardest (lowest MinLSE score)
        difficulty = 1.0 - (minlse_scores - score_min) / score_range
    else:
        difficulty = np.ones_like(minlse_scores) * 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # ── Shared helper ────────────────────────────────────────────────────────
    def _setup_ax(ax, title):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # ── Left panel: raw MinLSE score (cool = hard, warm = easy) ─────────────
    if len(candidate_goals) > 0:
        sc1 = ax1.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=minlse_scores,
            cmap="coolwarm_r",   # reversed: blue (cool) = low score = hard
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("MinLSE Score  (↓ harder to reach)", fontsize=10)

    # Start position
    ax1.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    # Selected goal
    ax1.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=180, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (hardest)", zorder=11,
    )
    _setup_ax(ax1, "UCGR Candidates — MinLSE Score")
    ax1.legend(fontsize=10, loc="best")

    # ── Right panel: normalised difficulty (bright = hardest) ────────────────
    if len(candidate_goals) > 0:
        sc2 = ax2.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=difficulty,
            cmap="plasma",
            vmin=0.0, vmax=1.0,
            s=50,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Normalised Difficulty  (↑ harder)", fontsize=10)

    # Start position
    ax2.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    # Selected goal
    ax2.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=180, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (hardest)", zorder=11,
    )
    _setup_ax(ax2, "UCGR Candidates — Normalised Difficulty")
    ax2.legend(fontsize=10, loc="best")

    plt.tight_layout()
    wandb.log({"ucgr_candidates": wandb.Image(fig)})
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# MEGA / OMEGA visualisation
# ─────────────────────────────────────────────────────────────────────────────

def visualize_mega_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    densities: np.ndarray,
    selected_goal: np.ndarray,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
    title_prefix: str = "MEGA",
) -> None:
    """Visualise MEGA goal-proposer candidates coloured by KDE density.

    Two side-by-side panels:
    - Left:  candidates coloured by raw KDE density estimate.
             Cool (low) = sparse = frontier; warm (high) = well-visited.
             The selected (minimum-density) goal is highlighted in lime.
    - Right: candidates coloured by normalised *sparsity* (1 − normalised density),
             so the selected goal is the brightest point.  Useful for seeing
             the frontier of the achievable goal set at a glance.

    Both panels also show the agent's start position as a red star.

    Args:
        candidate_goals:    (num_candidates, 2) candidate goal positions [x, y].
        first_obs_position: (2,) agent's start position this step [x, y].
        densities:          (num_candidates,) KDE density per candidate.
                            Lower → sparser → chosen by MEGA.
        selected_goal:      (2,) the chosen minimum-density goal [x, y].
        x_bounds:           [x_min, x_max] for the environment.
        y_bounds:           [y_min, y_max] for the environment.
        title_prefix:       "MEGA" or "OMEGA" (used in panel titles / wandb key).
    """
    candidate_goals    = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    densities          = np.asarray(densities)
    selected_goal      = np.asarray(selected_goal)
    x_bounds           = np.asarray(x_bounds)
    y_bounds           = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    # Normalised sparsity: 1 − normalised density  →  1 = most sparse = frontier
    d_min, d_max = densities.min(), densities.max()
    d_range = d_max - d_min
    if d_range > 0:
        sparsity = 1.0 - (densities - d_min) / d_range
    else:
        sparsity = np.ones_like(densities) * 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    def _setup_ax(ax, title):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # ── Left panel: raw KDE density (cool = sparse = frontier) ──────────────
    if len(candidate_goals) > 0:
        sc1 = ax1.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=densities,
            cmap="coolwarm",   # blue (cool) = low density = frontier
            s=55,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("KDE Density  (↓ = frontier)", fontsize=10)

    ax1.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    ax1.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=200, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (min density)", zorder=11,
    )
    _setup_ax(ax1, f"{title_prefix} Candidates — KDE Density")
    ax1.legend(fontsize=10, loc="best")

    # ── Right panel: normalised sparsity (bright = hardest / most sparse) ───
    if len(candidate_goals) > 0:
        sc2 = ax2.scatter(
            candidate_goals[:, 0],
            candidate_goals[:, 1],
            c=sparsity,
            cmap="plasma",
            vmin=0.0, vmax=1.0,
            s=55,
            alpha=0.75,
            edgecolors="black",
            linewidths=0.4,
            zorder=5,
        )
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Normalised Sparsity  (↑ = frontier)", fontsize=10)

    ax2.scatter(
        first_obs_position[0], first_obs_position[1],
        c="red", s=120, marker="*",
        edgecolors="black", linewidths=1.5,
        label="Start obs", zorder=10,
    )
    ax2.scatter(
        selected_goal[0], selected_goal[1],
        c="lime", s=200, marker="X",
        edgecolors="black", linewidths=1.5,
        label="Selected goal (min density)", zorder=11,
    )
    _setup_ax(ax2, f"{title_prefix} Candidates — Normalised Sparsity")
    ax2.legend(fontsize=10, loc="best")

    plt.tight_layout()
    wandb_key = f"{title_prefix.lower()}_candidates"
    wandb.log({wandb_key: wandb.Image(fig)})
    plt.close(fig)


def visualize_omega_candidates(
    candidate_goals: np.ndarray,
    first_obs_position: np.ndarray,
    densities: np.ndarray,
    selected_goal: np.ndarray,
    mega_goal: np.ndarray,
    env_goal: np.ndarray,
    alpha: float,
    kl_div: float,
    x_bounds: np.ndarray,
    y_bounds: np.ndarray,
) -> None:
    """Visualise OMEGA goal-proposer candidates.

    Extends the MEGA visualisation with OMEGA-specific overlays:
    - The MEGA (min-density) goal is marked with a blue diamond.
    - The randomly sampled environment (desired) goal is marked with an
      orange triangle.  If OMEGA chose the env goal this step, the selected
      goal (lime ✕) will coincide with the env goal; otherwise with the MEGA goal.
    - The mixing weight α and estimated KL divergence are annotated on the plot.

    Logs two images to wandb:
    - ``omega_candidates``  — the main candidate scatter (two panels, as in MEGA).
    - ``omega_alpha``       — a simple scalar-over-time log (just ``alpha`` this step).

    Args:
        candidate_goals:    (num_candidates, 2) candidate goal positions.
        first_obs_position: (2,) agent's start position.
        densities:          (num_candidates,) KDE density per candidate.
        selected_goal:      (2,) goal that was actually proposed this step.
        mega_goal:          (2,) minimum-density goal (MEGA choice).
        env_goal:           (2,) randomly drawn desired goal (env choice).
        alpha:              Mixing weight in [0, 1]; 1 → always use env goal.
        kl_div:             Estimated KL(p_dg ‖ p_ag) used to compute alpha.
        x_bounds:           [x_min, x_max] for the environment.
        y_bounds:           [y_min, y_max] for the environment.
    """
    candidate_goals    = np.asarray(candidate_goals)
    first_obs_position = np.asarray(first_obs_position)
    densities          = np.asarray(densities)
    selected_goal      = np.asarray(selected_goal)
    mega_goal          = np.asarray(mega_goal)
    env_goal           = np.asarray(env_goal)
    alpha              = float(alpha)
    kl_div             = float(kl_div)
    x_bounds           = np.asarray(x_bounds)
    y_bounds           = np.asarray(y_bounds)

    x_min, x_max = float(x_bounds[0]), float(x_bounds[1])
    y_min, y_max = float(y_bounds[0]), float(y_bounds[1])

    d_min, d_max = densities.min(), densities.max()
    d_range = d_max - d_min
    sparsity = (
        1.0 - (densities - d_min) / d_range if d_range > 0
        else np.ones_like(densities) * 0.5
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

    def _setup_ax(ax, title):
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("X Position", fontsize=12)
        ax.set_ylabel("Y Position", fontsize=12)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    def _add_omega_overlays(ax):
        """Add start, mega, env, and selected goal markers."""
        # Agent start
        ax.scatter(
            first_obs_position[0], first_obs_position[1],
            c="red", s=120, marker="*", edgecolors="black", linewidths=1.5,
            label="Start obs", zorder=10,
        )
        # MEGA candidate (min-density)
        ax.scatter(
            mega_goal[0], mega_goal[1],
            c="deepskyblue", s=180, marker="D",
            edgecolors="black", linewidths=1.5,
            label=f"MEGA goal (min ρ)", zorder=11,
        )
        # Env (desired) goal
        ax.scatter(
            env_goal[0], env_goal[1],
            c="orange", s=180, marker="^",
            edgecolors="black", linewidths=1.5,
            label="Env goal (desired)", zorder=11,
        )
        # Actually selected goal (lime ✕)
        ax.scatter(
            selected_goal[0], selected_goal[1],
            c="lime", s=220, marker="X",
            edgecolors="black", linewidths=1.5,
            label="Selected goal", zorder=12,
        )

    # ── Left panel: raw KDE density ──────────────────────────────────────────
    if len(candidate_goals) > 0:
        sc1 = ax1.scatter(
            candidate_goals[:, 0], candidate_goals[:, 1],
            c=densities, cmap="coolwarm",
            s=55, alpha=0.75, edgecolors="black", linewidths=0.4, zorder=5,
        )
        cb1 = plt.colorbar(sc1, ax=ax1)
        cb1.set_label("KDE Density  (↓ = frontier)", fontsize=10)

    _add_omega_overlays(ax1)
    info_text = f"α = {alpha:.3f}   KL = {kl_div:.2f}"
    ax1.text(
        0.02, 0.98, info_text,
        transform=ax1.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _setup_ax(ax1, "OMEGA Candidates — KDE Density")
    ax1.legend(fontsize=9, loc="lower right")

    # ── Right panel: normalised sparsity ─────────────────────────────────────
    if len(candidate_goals) > 0:
        sc2 = ax2.scatter(
            candidate_goals[:, 0], candidate_goals[:, 1],
            c=sparsity, cmap="plasma", vmin=0.0, vmax=1.0,
            s=55, alpha=0.75, edgecolors="black", linewidths=0.4, zorder=5,
        )
        cb2 = plt.colorbar(sc2, ax=ax2)
        cb2.set_label("Normalised Sparsity  (↑ = frontier)", fontsize=10)

    _add_omega_overlays(ax2)
    ax2.text(
        0.02, 0.98, info_text,
        transform=ax2.transAxes, fontsize=11, verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )
    _setup_ax(ax2, "OMEGA Candidates — Normalised Sparsity")
    ax2.legend(fontsize=9, loc="lower right")

    plt.tight_layout()
    wandb.log({
        "omega_candidates": wandb.Image(fig),
        "omega_alpha":      alpha,
        "omega_kl_div":     kl_div,
    })
    plt.close(fig)