import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import Config
from env import ContinuousHomeostaticEnv


def collect(agent, env, n_episodes=10, gamma=Config.GAMMA, deterministic=True, use_target=False):
    """Roll out episodes and gather data for both diagnostics.

    Returns per-step states/actions/rewards/Q with their Monte-Carlo returns,
    plus per-episode returns and the drive telescoping terms.
    """
    all_states, all_actions, all_rewards, all_returns = [], [], [], []
    episode_returns = []
    drive_initial, drive_final, effort_sums, bonus_sums = [], [], [], []
    ep_lengths, ep_terminated = [], []

    for _ in range(n_episodes):
        state, _ = env.reset()
        drive_initial.append(env.drive(env.internal_state))

        ep_states, ep_actions, ep_rewards = [], [], []
        effort_sum = 0.0
        bonus_sum = 0.0
        terminated = False
        info = {}

        for _ in range(env.max_steps):
            action = (env.action_space.sample() if agent is None
                      else agent.select_action(state, add_noise=not deterministic, use_target=use_target))
            ep_states.append(np.asarray(state, dtype=np.float32))
            ep_actions.append(np.asarray(action, dtype=np.float32))

            state, reward, terminated, truncated, info = env.step(action)
            ep_rewards.append(reward)
            effort_sum += env.effort_penalty * np.sum(np.asarray(action) ** 2)
            # Reproduce the state-dependent survival bonus actually paid this step.
            bonus_sum += env._survival_bonus(info["drive"], terminated)
            if terminated or truncated:
                break

        # Monte-Carlo discounted returns G_t for this episode
        returns = np.zeros(len(ep_rewards), dtype=np.float64)
        g = 0.0
        for t in reversed(range(len(ep_rewards))):
            g = ep_rewards[t] + gamma * g
            returns[t] = g

        all_states.extend(ep_states)
        all_actions.extend(ep_actions)
        all_rewards.extend(ep_rewards)
        all_returns.extend(returns.tolist())
        episode_returns.append(float(np.sum(ep_rewards)))
        drive_final.append(info.get("drive", np.nan))
        effort_sums.append(effort_sum)
        bonus_sums.append(bonus_sum)
        ep_lengths.append(len(ep_rewards))
        ep_terminated.append(bool(terminated))

    states = np.array(all_states)
    actions = np.array(all_actions)
    rewards = np.array(all_rewards)
    mc_returns = np.array(all_returns)
    q_estimates = (agent.evaluate_q(states, actions) if agent is not None
                   else np.full(len(states), np.nan))

    return {
        "states": states, "actions": actions, "rewards": rewards,
        "mc_returns": mc_returns, "q_estimates": q_estimates,
        "episode_returns": np.array(episode_returns),
        "drive_initial": np.array(drive_initial),
        "drive_final": np.array(drive_final),
        "effort_sums": np.array(effort_sums),
        "bonus_sums": np.array(bonus_sums),
        "ep_lengths": np.array(ep_lengths),
        "ep_terminated": np.array(ep_terminated),
        "dt": env.dt, "gamma": gamma,
        "reward_scale": env.reward_scale, "survival_bonus": env.survival_bonus,
    }


def analyze(data, config=Config, save_path=None):
    """Print findings for the two scenarios and save a diagnostic figure."""
    rewards = data["rewards"]
    ep_ret = data["episode_returns"]
    q = data["q_estimates"]
    g = data["mc_returns"]

    print("\n================ DIAGNOSTICS ================")

    # ---- Scenario 2: reward scale / all-negative returns ----
    frac_neg_steps = float(np.mean(rewards < 0))
    frac_neg_eps = float(np.mean(ep_ret < 0))
    print("\n[Reward scale & return sign]")
    print(f"  per-step reward : mean={rewards.mean():.3f}  std={rewards.std():.3f}  "
          f"min={rewards.min():.3f}  max={rewards.max():.3f}")
    print(f"  fraction of steps with negative reward    : {frac_neg_steps:6.1%}")
    print(f"  episode return  : mean={ep_ret.mean():.2f}  min={ep_ret.min():.2f}  max={ep_ret.max():.2f}")
    print(f"  fraction of episodes with negative return : {frac_neg_eps:6.1%}")

    # Telescoping identity for the current reward (survival bonus is now per-step
    # and state-dependent, so it is summed directly rather than bonus * alive_steps):
    #   return = scale * [ (D_init - D_final) - effort_sum + bonus_sum ]
    telescoped = data["reward_scale"] * (
        (data["drive_initial"] - data["drive_final"])
        - data["effort_sums"]
        + data["bonus_sums"]
    )
    print(f"  telescoped return check scale*[(D_init-D_final) - effort + bonus_sum] : "
          f"mean={np.nanmean(telescoped):.2f}  (matches episode return)")
    if frac_neg_eps >= 0.8:
        print("  -> VERDICT: returns are still predominantly NEGATIVE. Agents are dying "
              "or drifting from H* faster than the survival bonus can offset. Consider "
              "raising SURVIVAL_BONUS, lowering EFFORT_PENALTY, or longer training.")
    else:
        print("  -> VERDICT: returns are no longer uniformly negative -- surviving, "
              "homeostatic trajectories now earn positive return (reward scale fixed).")

    # ---- Scenario 1: critic under/over-estimation ----
    print("\n[Critic value bias  (Q vs Monte-Carlo return)]")
    if np.all(np.isnan(q)):
        print("  no trained critic available (random policy) -- skipping bias check.")
        bias = norm_bias = np.nan
    else:
        bias = float(np.mean(q - g))
        std_g = float(np.std(g)) + 1e-8
        norm_bias = bias / std_g
        corr = float(np.corrcoef(q, g)[0, 1]) if len(q) > 1 else np.nan
        print(f"  mean Q          : {q.mean():.3f}")
        print(f"  mean MC return  : {g.mean():.3f}")
        print(f"  mean bias (Q-G) : {bias:+.3f}   normalized (bias/std_G): {norm_bias:+.2f}")
        print(f"  corr(Q, G)      : {corr:+.2f}")
        if norm_bias < -0.25:
            print("  -> VERDICT: critic UNDERESTIMATES returns (Q systematically below "
                  "realized returns). Common with a slow/cold critic; ensure "
                  "LEARNING_STARTS isn't starving it and consider a higher CRITIC_LR "
                  "or more updates per step.")
        elif norm_bias > 0.25:
            print("  -> VERDICT: critic OVERESTIMATES returns (the classic DDPG failure "
                  "mode). Consider TD3-style clipped double-Q / target smoothing.")
        else:
            print("  -> VERDICT: critic value bias is small; calibration looks healthy.")

    # ---- Figure ----
    os.makedirs(config.PLOT_DIR, exist_ok=True)
    save_path = save_path or os.path.join(config.PLOT_DIR, "diagnostics.png")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (1) Q vs MC return
    ax = axes[0]
    if not np.all(np.isnan(q)):
        ax.scatter(g, q, s=6, alpha=0.3)
        lo, hi = float(min(g.min(), np.nanmin(q))), float(max(g.max(), np.nanmax(q)))
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="Q = G (calibrated)")
        ax.legend(fontsize=8)
    ax.set_title("Critic Q vs Monte-Carlo return")
    ax.set_xlabel("MC return G")
    ax.set_ylabel("critic Q")

    # (2) per-step reward histogram
    ax = axes[1]
    ax.hist(rewards, bins=60, color="tab:orange")
    ax.axvline(0, color="k", ls="--", lw=1)
    ax.set_title("Per-step reward distribution")
    ax.set_xlabel("reward")
    ax.set_ylabel("count")

    # (3) episode returns
    ax = axes[2]
    ax.bar(np.arange(len(ep_ret)), ep_ret,
           color=["tab:red" if r < 0 else "tab:green" for r in ep_ret])
    ax.axhline(0, color="k", ls="--", lw=1)
    ax.set_title("Episode returns")
    ax.set_xlabel("episode")
    ax.set_ylabel("undiscounted return")

    fig.tight_layout()
    fig.savefig(save_path, dpi=130)
    plt.close(fig)
    print(f"\n[diagnostics] saved figure: {save_path}")
    print("=============================================\n")
    return {"step_reward_mean": float(rewards.mean()),
            "frac_neg_episodes": frac_neg_eps,
            "critic_bias": bias, "critic_norm_bias": norm_bias}


def compare_actor_heads(agent, env, n_episodes=3, gamma=Config.GAMMA):
    """Compare actor vs actor_target on the same environment."""
    actor_data = collect(agent, env, n_episodes=n_episodes, gamma=gamma, use_target=False)
    target_data = collect(agent, env, n_episodes=n_episodes, gamma=gamma, use_target=True)
    actor_ret = actor_data["episode_returns"]
    target_ret = target_data["episode_returns"]
    actor_survival = 1.0 - actor_data["ep_terminated"].mean()
    target_survival = 1.0 - target_data["ep_terminated"].mean()
    print(
        "[policy head check] "
        f"actor mean return={actor_ret.mean():.2f}, survival={actor_survival:.1%} | "
        f"actor_target mean return={target_ret.mean():.2f}, survival={target_survival:.1%}"
    )


def run(agent, env=None, config=Config, n_episodes=10, save_path=None):
    env = env or ContinuousHomeostaticEnv(config)
    if agent is not None:
        compare_actor_heads(agent, env, n_episodes=min(3, n_episodes), gamma=config.GAMMA)
    data = collect(agent, env, n_episodes=n_episodes, gamma=config.GAMMA)
    return analyze(data, config, save_path=save_path)


if __name__ == "__main__":
    from agents import make_agent

    cfg = Config
    device = cfg.get_device()
    env = ContinuousHomeostaticEnv(cfg)
    agent = make_agent(env, device, cfg)
    try:
        agent.load(cfg.ACTOR_PATH, cfg.CRITIC_PATH)
        print(f"[diagnostics] loaded trained weights from {cfg.ACTOR_PATH}")
    except FileNotFoundError:
        print("[diagnostics] no saved model found; diagnosing a random policy.")
        agent = None

    run(agent, env, cfg, n_episodes=10)
