import os

import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless: save figures without a display
import matplotlib.pyplot as plt

from config import Config
from env import ContinuousHomeostaticEnv


# Type index -> (label, colour) for plotting.
TYPE_STYLE = {0: ("food", "tab:green"), 1: ("water", "tab:blue")}
INTERNAL_LABELS = ["Food", "Water"]


def rollout(agent, env, deterministic=True, max_steps=None):
    """Run one episode and record everything needed for the plots.

    If ``agent`` is None, a random policy is used (handy before training).
    """
    max_steps = max_steps or env.max_steps
    state, _ = env.reset()

    positions = [env.agent_pos.copy()]
    internal = [env.internal_state.copy()]
    rewards = []
    consume_points = []          # agent positions where consumption occurred
    consume_caps = [env.resource_cap.copy()]
    within_history = []          # per-step spatial presence (T, M)
    delivered_history = []       # per-step amount consumed per resource (T, M)

    for _ in range(max_steps):
        if agent is None:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, add_noise=not deterministic)

        state, reward, terminated, truncated, info = env.step(action)

        positions.append(info["agent_pos"])
        internal.append(info["internal_state"])
        rewards.append(reward)
        consume_caps.append(info["resource_cap"])
        within_history.append(info["within_radius"])
        delivered_history.append(info["delivered"])
        if info["consuming"]:
            consume_points.append(info["agent_pos"])

        if terminated or truncated:
            break

    return {
        "positions": np.array(positions),
        "internal": np.array(internal),
        "rewards": np.array(rewards),
        "consume_points": np.array(consume_points) if consume_points else np.empty((0, 2)),
        "consume_caps": np.array(consume_caps),
        "within_history": np.array(within_history),       # (T, M) bool
        "delivered_history": np.array(delivered_history),  # (T, M) float
        "terminated": terminated,
    }


def resource_labels(env):
    """Human-readable per-resource labels like food1, water1, food2, ..."""
    counts = {}
    labels = []
    for t in env.resource_type:
        name = TYPE_STYLE.get(int(t), (f"type{int(t)}", None))[0]
        counts[name] = counts.get(name, 0) + 1
        labels.append(f"{name}{counts[name]}")
    return labels


def resource_stats(data, env):
    """Per-resource visit count, average stay time, and capacity dynamics."""
    within = data["within_history"]               # (T, M) bool
    caps = data["consume_caps"]                    # (T+1, M)
    n = env.n_resources

    visits = np.zeros(n, dtype=int)
    avg_stay = np.zeros(n)
    for i in range(n):
        col = within[:, i].astype(int) if within.size else np.zeros(0, dtype=int)
        # A visit = a rising edge (outside -> inside the radius).
        rising = np.sum((col[1:] == 1) & (col[:-1] == 0)) + (col[0] if len(col) else 0)
        visits[i] = int(rising)
        # Average stay = total steps inside / number of visits.
        avg_stay[i] = (col.sum() / rising) if rising > 0 else 0.0

    return {
        "labels": resource_labels(env),
        "types": env.resource_type,
        "visits": visits,
        "avg_stay": avg_stay,
        "max_capacity": env.resource_max_cap,
        "final_capacity": caps[-1].copy(),
        "regen_rate": env.resource_regen,
        "caps_over_time": caps,
    }


def plot_trajectory(data, env, save_path):
    """Plot the agent's path, resource sites, and consumption events."""
    fig, ax = plt.subplots(figsize=(8, 8))
    pos = data["positions"]

    # Agent path, coloured by progress (time)
    ax.plot(pos[:, 0], pos[:, 1], "-", color="0.6", lw=1, alpha=0.7, zorder=1)
    ax.scatter(pos[:, 0], pos[:, 1], c=np.arange(len(pos)), cmap="viridis",
               s=8, zorder=2, label="path (time)")
    ax.plot(pos[0, 0], pos[0, 1], "ko", ms=10, label="start", zorder=4)
    ax.plot(pos[-1, 0], pos[-1, 1], "kX", ms=12, label="end", zorder=4)

    # Resource sites: colour by type, consumption radius as a ring
    seen = set()
    for i in range(env.n_resources):
        t = int(env.resource_type[i])
        label, colour = TYPE_STYLE.get(t, (f"type{t}", "tab:red"))
        x, y = env.resource_pos[i]
        ax.add_patch(plt.Circle((x, y), env.consume_radius, color=colour,
                                 alpha=0.12, zorder=0))
        ax.plot(x, y, "*", color=colour, ms=18, zorder=3,
                label=label if label not in seen else None)
        seen.add(label)

    # Consumption events
    cp = data["consume_points"]
    if len(cp):
        ax.scatter(cp[:, 0], cp[:, 1], marker="o", facecolors="none",
                   edgecolors="red", s=40, lw=1.2, zorder=5, label="consumption")

    lim = env.world_size + 1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_title(f"Agent Trajectory ({'died' if data['terminated'] else 'survived'})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_internal_states(data, env, save_path):
    """Plot each internal state over time; it should oscillate around H*, not collapse."""
    internal = data["internal"]
    steps = np.arange(len(internal))

    fig, ax = plt.subplots(figsize=(10, 5))
    for i in range(env.n_internal):
        label = INTERNAL_LABELS[i] if i < len(INTERNAL_LABELS) else f"State {i + 1}"
        ax.plot(steps, internal[:, i], lw=1.5, label=label)

    ax.axhline(env.H_star[0], color="k", ls="--", lw=1, alpha=0.7, label="target H*")
    ax.axhline(0.0, color="r", ls=":", lw=1, alpha=0.6, label="death (0)")
    ax.set_ylim(-0.05, 2.05)
    ax.set_title("Internal State Over Time")
    ax.set_xlabel("step")
    ax.set_ylabel("internal state level")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def plot_resource_stats(data, env, save_path):
    """Per-resource diagnostics: visits, average stay time, and capacity dynamics
    (max capacity, remaining capacity, regeneration)."""
    s = resource_stats(data, env)
    labels = s["labels"]
    colours = [TYPE_STYLE.get(int(t), ("?", "tab:red"))[1] for t in s["types"]]
    x = np.arange(env.n_resources)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # (1) Visits per resource
    ax = axes[0, 0]
    ax.bar(x, s["visits"], color=colours)
    ax.set_title("Visits per resource (radius entries)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("visits")

    # (2) Average stay time per resource
    ax = axes[0, 1]
    ax.bar(x, s["avg_stay"], color=colours)
    ax.set_title("Average stay time per visit")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("steps inside / visit")

    # (3) Remaining capacity over time (depletion + regeneration dynamics)
    ax = axes[1, 0]
    caps = s["caps_over_time"]
    steps = np.arange(len(caps))
    for i in range(env.n_resources):
        ax.plot(steps, caps[:, i], color=colours[i], lw=1.3, label=labels[i])
    ax.set_title("Remaining capacity over time")
    ax.set_xlabel("step"); ax.set_ylabel("remaining capacity")
    ax.legend(fontsize=7, ncol=2)

    # (4) Capacity summary: max vs final remaining, with regen rate annotated
    ax = axes[1, 1]
    w = 0.4
    ax.bar(x - w / 2, s["max_capacity"], width=w, label="max capacity", color=colours, alpha=0.5)
    ax.bar(x + w / 2, s["final_capacity"], width=w, label="final remaining", color=colours)
    for i in range(env.n_resources):
        ax.text(x[i], s["max_capacity"][i] + 0.02, f"regen\n{s['regen_rate'][i]:.3f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_title("Capacity: max vs final remaining (+ regen rate)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("capacity")
    ax.legend(fontsize=8)

    fig.suptitle("Per-resource statistics", fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    return save_path


def visualize(agent, env=None, config=Config, prefix="trained"):
    """Roll out one episode and save the trajectory, internal-state, and
    per-resource statistics plots."""
    env = env or ContinuousHomeostaticEnv(config)
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    data = rollout(agent, env, deterministic=True)

    traj_path = plot_trajectory(data, env, os.path.join(config.PLOT_DIR, f"{prefix}_trajectory.png"))
    state_path = plot_internal_states(data, env, os.path.join(config.PLOT_DIR, f"{prefix}_internal_states.png"))
    res_path = plot_resource_stats(data, env, os.path.join(config.PLOT_DIR, f"{prefix}_resource_stats.png"))

    survived = not data["terminated"]
    print(
        f"[visualize] {prefix}: {'survived' if survived else 'died'} after "
        f"{len(data['positions']) - 1} steps, "
        f"{len(data['consume_points'])} consumption steps, "
        f"return={data['rewards'].sum():.2f}"
    )
    print(f"[visualize] saved: {traj_path}")
    print(f"[visualize] saved: {state_path}")
    print(f"[visualize] saved: {res_path}")
    return data


if __name__ == "__main__":
    # Standalone: load the trained agent if available, else visualize a random policy.
    from agents import make_agent

    cfg = Config
    device = cfg.get_device()
    env = ContinuousHomeostaticEnv(cfg)
    agent = make_agent(env, device, cfg)
    try:
        agent.load(cfg.ACTOR_PATH, cfg.CRITIC_PATH)
        print(f"[visualize] loaded trained weights from {cfg.ACTOR_PATH}")
    except FileNotFoundError:
        print("[visualize] no saved model found; visualizing a random policy.")
        agent = None

    visualize(agent, env, cfg, prefix="trained" if agent is not None else "random")
