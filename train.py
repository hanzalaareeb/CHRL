import os
from collections import defaultdict, deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from env import ContinuousHomeostaticEnv


INTERNAL_LABELS = ["Food", "Water"]

LOSS_TAGS = {
    "actor_loss": "Loss/Actor",
    "critic_loss": "Loss/Critic",
    "q_value": "Q/Average",
    "q1": "Q/Q1",
    "q2": "Q/Q2",
    "q_diff": "Q/Q1_minus_Q2",
}


def exploration_sigma(episode, total_episodes, config):
    """Linearly decay exploration across the whole training budget."""
    decay_eps = max(1, int(config.EXPLORATION_DECAY_FRAC * total_episodes))
    frac = min(1.0, episode / decay_eps)
    return config.EXPLORATION_NOISE + frac * (
        config.EXPLORATION_NOISE_FINAL - config.EXPLORATION_NOISE
    )


def evaluate(agent, env, use_target=False):
    """Run one deterministic episode with either the actor or actor_target."""
    state, _ = env.reset()
    total_reward = 0.0
    steps = 0
    terminated = False
    info = {}
    done = False
    while not done:
        action = agent.select_action(state, add_noise=False, use_target=use_target)
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    return {
        "reward": float(total_reward),
        "steps": steps,
        "final_drive": info.get("drive", float("nan")),
        "survived": float(not terminated),
    }


def save_policy_checkpoint(agent, actor_path, critic_path, use_target=False):
    """Save the chosen actor head together with the current critic."""
    torch.save(agent.actor_state_dict(use_target=use_target), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)


def make_tracker():
    """Run-level counters that persist across curriculum stages."""
    return {
        "episodes_with_consumption": 0,
        "episodes_surviving_500": 0,
        "episodes_surviving_1000": 0,
        "current_survival_streak": 0,
        "longest_survival_streak": 0,
        "best_drive_achieved": float("inf"),
        "best_eval_score": (-1.0, -float("inf"), -float("inf")),
        "best_eval_policy": None,
        "best_eval_stage": None,
        "best_eval_episode": None,
        "success_history": deque(maxlen=Config.SUCCESS_RATE_WINDOW),
    }


def episode_metrics_template(env):
    return {
        "distance_travelled": 0.0,
        "food_visits": 0,
        "water_visits": 0,
        "consumption_events": 0,
        "total_consumption": 0.0,
        "first_food_step": -1,
        "first_water_step": -1,
        "first_consumption_step": -1,
        "time_inside_resource_radius": 0,
        "unique_resources_mask": np.zeros(env.n_resources, dtype=bool),
        "prev_within": np.zeros(env.n_resources, dtype=bool),
        "best_drive": float("inf"),
    }


def collect_episode_transition_metrics(metrics, step_idx, prev_pos, info, resource_type):
    current_pos = np.asarray(info["agent_pos"], dtype=np.float64)
    metrics["distance_travelled"] += float(np.linalg.norm(current_pos - prev_pos))

    within = np.asarray(info["within_radius"], dtype=bool)
    delivered = np.asarray(info["delivered"], dtype=np.float64)
    rising = within & ~metrics["prev_within"]

    food_mask = resource_type == 0
    water_mask = resource_type == 1

    metrics["food_visits"] += int(np.sum(rising[food_mask]))
    metrics["water_visits"] += int(np.sum(rising[water_mask]))
    metrics["consumption_events"] += int(np.any(delivered > 0.0))
    metrics["total_consumption"] += float(np.sum(delivered))
    metrics["time_inside_resource_radius"] += int(np.any(within))
    metrics["unique_resources_mask"] |= within
    metrics["best_drive"] = min(metrics["best_drive"], float(info["drive"]))

    if metrics["first_food_step"] < 0 and np.any(within[food_mask]):
        metrics["first_food_step"] = step_idx
    if metrics["first_water_step"] < 0 and np.any(within[water_mask]):
        metrics["first_water_step"] = step_idx
    if metrics["first_consumption_step"] < 0 and np.any(delivered > 0.0):
        metrics["first_consumption_step"] = step_idx

    metrics["prev_within"] = within
    return current_pos


def write_episode_metrics(writer, episode, info, agent, loss_history, n_internal):
    writer.add_scalar("Reward/Episode", info["episode_reward"], episode)
    writer.add_scalar("Episode/Length", info["episode_steps"], episode)
    writer.add_scalar("Episode/ConsumptionEvents", info["consumption_events"], episode)
    writer.add_scalar("Episode/DistanceTravelled", info["distance_travelled"], episode)
    writer.add_scalar("Episode/UniqueResourcesVisited", info["unique_resources_visited"], episode)
    writer.add_scalar("Episode/TimeInsideRadius", info["time_inside_resource_radius"], episode)
    writer.add_scalar("Episode/FoodVisits", info["food_visits"], episode)
    writer.add_scalar("Episode/WaterVisits", info["water_visits"], episode)
    writer.add_scalar("Episode/AvgConsumptionPerVisit", info["avg_consumption_per_visit"], episode)
    writer.add_scalar("Episode/FirstFoodStep", info["first_food_step_for_log"], episode)
    writer.add_scalar("Episode/FirstWaterStep", info["first_water_step_for_log"], episode)
    writer.add_scalar("Episode/FirstConsumptionStep", info["first_consumption_step_for_log"], episode)
    writer.add_scalar("Episode/ConsumedAny", float(info["consumed_any"]), episode)
    writer.add_scalar("Episode/Survived500", float(info["survived_500"]), episode)
    writer.add_scalar("Episode/Survived1000", float(info["survived_1000"]), episode)
    writer.add_scalar("Homeostasis/Final_Drive", info["final_drive"], episode)
    writer.add_scalar("Homeostasis/Best_Drive", info["best_drive"], episode)
    writer.add_scalar("Replay/SuccessRatio", info["replay_success_ratio"], episode)
    writer.add_scalar("Success/RateLast100", info["success_rate_last100"], episode)
    writer.add_scalar("Success/LongestStreak", info["longest_survival_streak"], episode)
    writer.add_scalar("Success/EpisodesWithConsumption", info["episodes_with_consumption"], episode)
    writer.add_scalar("Success/EpisodesSurviving500", info["episodes_surviving_500"], episode)
    writer.add_scalar("Success/EpisodesSurviving1000", info["episodes_surviving_1000"], episode)
    writer.add_scalar("Homeostasis/AverageFinalDrive", info["average_final_drive"], episode)

    for key, values in loss_history.items():
        if values:
            tag = LOSS_TAGS.get(key, f"Loss/{key}")
            writer.add_scalar(tag, np.mean(values), episode)

    writer.add_scalar("Action/Average", np.mean(info["action_abs_sum"]) / info["episode_steps"], episode)
    writer.add_scalar("Exploration/Sigma", agent.noise.sigma, episode)
    writer.add_scalar("Distance/Nearest_Food", info["nearest_food_sum"] / info["episode_steps"], episode)
    writer.add_scalar("Distance/Nearest_Water", info["nearest_water_sum"] / info["episode_steps"], episode)
    for i in range(n_internal):
        label = INTERNAL_LABELS[i] if i < len(INTERNAL_LABELS) else f"State{i + 1}"
        writer.add_scalar(f"InternalState/{label}", info["states_sum"][i] / info["episode_steps"], episode)


def maybe_log_evals(agent, writer, episode, stage_name, stage_env, full_env, tracker, config):
    if (episode + 1) % config.EVAL_INTERVAL != 0:
        return

    evals = {
        "stage_actor": evaluate(agent, stage_env, use_target=False),
        "stage_actor_target": evaluate(agent, stage_env, use_target=True),
        "full_actor": evaluate(agent, full_env, use_target=False),
        "full_actor_target": evaluate(agent, full_env, use_target=True),
    }

    for name, ev in evals.items():
        prefix = f"Eval/{stage_name}/{name}"
        writer.add_scalar(f"{prefix}/Reward", ev["reward"], episode)
        writer.add_scalar(f"{prefix}/Length", ev["steps"], episode)
        writer.add_scalar(f"{prefix}/FinalDrive", ev["final_drive"], episode)
        writer.add_scalar(f"{prefix}/Survived", ev["survived"], episode)

    actor_ev = evals["full_actor"]
    target_ev = evals["full_actor_target"]
    actor_score = (actor_ev["survived"], actor_ev["reward"], -actor_ev["final_drive"])
    target_score = (target_ev["survived"], target_ev["reward"], -target_ev["final_drive"])
    best_name, best_ev, use_target = (
        ("actor_target", target_ev, True) if target_score > actor_score else ("actor", actor_ev, False)
    )
    best_score = target_score if use_target else actor_score
    if best_score > tracker["best_eval_score"]:
        tracker["best_eval_score"] = best_score
        tracker["best_eval_policy"] = best_name
        tracker["best_eval_stage"] = stage_name
        tracker["best_eval_episode"] = episode + 1
        save_policy_checkpoint(agent, config.BEST_ACTOR_PATH, config.BEST_CRITIC_PATH, use_target=use_target)

    tqdm.write(
        f"  [eval @ ep {episode + 1}] stage(actor={evals['stage_actor']['reward']:.2f}, "
        f"target={evals['stage_actor_target']['reward']:.2f}) | "
        f"full(actor={actor_ev['reward']:.2f}, target={target_ev['reward']:.2f}) | "
        f"best={best_name}"
    )


def run_episodes(
    agent,
    env,
    writer,
    config,
    n_episodes,
    total_episodes,
    tracker,
    full_eval_env,
    episode_offset=0,
    total_steps=0,
    stage_name="stage",
):
    """Run a curriculum stage and return updated counters."""
    n_internal = env.n_internal
    resource_type = env.resource_type
    rolling_success = tracker["success_history"]
    stage_success = deque(maxlen=config.STAGE_ADVANCE_WINDOW)
    final_drives = []

    for local_ep in tqdm(range(n_episodes), desc=f"Stage {stage_name}"):
        episode = episode_offset + local_ep
        state, _ = env.reset()
        agent.noise.reset()

        base_sigma = exploration_sigma(episode, total_episodes, config)
        noise_floor = env.config.EXPLORATION_NOISE_FINAL
        if hasattr(env, "stage_noise_floor"):
            noise_floor = max(noise_floor, env.stage_noise_floor)
        agent.noise.sigma = max(base_sigma, noise_floor)

        episode_reward = 0.0
        episode_steps = 0
        info = {}
        done = False
        loss_history = defaultdict(list)
        action_abs_sum = np.zeros(agent.action_dim)
        states_sum = np.zeros(n_internal)
        nearest_food_sum = 0.0
        nearest_water_sum = 0.0
        prev_pos = env.agent_pos.copy()
        episode_metrics = episode_metrics_template(env)
        episode_buffer_indices = []

        while not done:
            if total_steps < config.START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, add_noise=True)

            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer_idx = agent.replay_buffer.add(state, action, reward, next_state, float(terminated))
            episode_buffer_indices.append(buffer_idx)

            state = next_state
            episode_reward += reward
            total_steps += 1
            episode_steps += 1

            action_abs_sum += np.abs(action)
            states_sum += info["internal_state"]
            dists = info["dists"]
            nearest_food_sum += dists[resource_type == 0].min()
            nearest_water_sum += dists[resource_type == 1].min()
            prev_pos = collect_episode_transition_metrics(
                episode_metrics, episode_steps, prev_pos, info, resource_type
            )

            if agent.replay_buffer.size > config.LEARNING_STARTS:
                losses = agent.update()
                if losses is not None:
                    for key, value in losses.items():
                        loss_history[key].append(value)

        consumed_any = episode_metrics["consumption_events"] > 0
        survived_500 = episode_steps >= 500
        survived_1000 = bool((not terminated) and truncated and episode_steps >= env.max_steps)
        success = survived_1000
        total_visits = episode_metrics["food_visits"] + episode_metrics["water_visits"]
        avg_consumption_per_visit = episode_metrics["total_consumption"] / total_visits if total_visits else 0.0
        unique_resources_visited = int(np.sum(episode_metrics["unique_resources_mask"]))

        if success:
            agent.replay_buffer.mark_success(episode_buffer_indices)
            agent.replay_buffer.save(config.BUFFER_PATH)

        tracker["episodes_with_consumption"] += int(consumed_any)
        tracker["episodes_surviving_500"] += int(survived_500)
        tracker["episodes_surviving_1000"] += int(survived_1000)
        tracker["current_survival_streak"] = tracker["current_survival_streak"] + 1 if success else 0
        tracker["longest_survival_streak"] = max(
            tracker["longest_survival_streak"], tracker["current_survival_streak"]
        )
        tracker["best_drive_achieved"] = min(tracker["best_drive_achieved"], episode_metrics["best_drive"])

        rolling_success.append(float(success))
        stage_success.append(float(success))
        final_drives.append(float(info.get("drive", np.nan)))

        episode_info = {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "consumption_events": episode_metrics["consumption_events"],
            "distance_travelled": episode_metrics["distance_travelled"],
            "food_visits": episode_metrics["food_visits"],
            "water_visits": episode_metrics["water_visits"],
            "avg_consumption_per_visit": avg_consumption_per_visit,
            "first_food_step_for_log": episode_metrics["first_food_step"] if episode_metrics["first_food_step"] >= 0 else env.max_steps,
            "first_water_step_for_log": episode_metrics["first_water_step"] if episode_metrics["first_water_step"] >= 0 else env.max_steps,
            "first_consumption_step_for_log": episode_metrics["first_consumption_step"] if episode_metrics["first_consumption_step"] >= 0 else env.max_steps,
            "time_inside_resource_radius": episode_metrics["time_inside_resource_radius"],
            "unique_resources_visited": unique_resources_visited,
            "consumed_any": consumed_any,
            "survived_500": survived_500,
            "survived_1000": survived_1000,
            "final_drive": float(info.get("drive", float("nan"))),
            "best_drive": episode_metrics["best_drive"],
            "replay_success_ratio": agent.replay_buffer.success_ratio(),
            "success_rate_last100": float(np.mean(rolling_success)) if rolling_success else 0.0,
            "longest_survival_streak": tracker["longest_survival_streak"],
            "episodes_with_consumption": tracker["episodes_with_consumption"],
            "episodes_surviving_500": tracker["episodes_surviving_500"],
            "episodes_surviving_1000": tracker["episodes_surviving_1000"],
            "average_final_drive": float(np.nanmean(final_drives)),
            "action_abs_sum": action_abs_sum,
            "nearest_food_sum": nearest_food_sum,
            "nearest_water_sum": nearest_water_sum,
            "states_sum": states_sum,
        }
        write_episode_metrics(writer, episode, episode_info, agent, loss_history, n_internal)

        writer.add_scalar(f"Stage/{stage_name}/SuccessRate", float(np.mean(stage_success)), episode)
        writer.add_scalar(f"Stage/{stage_name}/EpisodesCompleted", local_ep + 1, episode)

        maybe_log_evals(agent, writer, episode, stage_name, env, full_eval_env, tracker, config)

        if (episode + 1) % config.LOG_INTERVAL == 0:
            q_hist = loss_history.get("q_value") or loss_history.get("q1") or []
            avg_q = np.mean(q_hist) if q_hist else float("nan")
            tqdm.write(
                f"Episode {episode + 1:4d} | Reward: {episode_reward:8.2f} | "
                f"Steps: {episode_steps:4d} | Consumed: {episode_metrics['consumption_events']:4d} | "
                f"FirstConsume: {episode_info['first_consumption_step_for_log']:4d} | "
                f"Dist: {episode_metrics['distance_travelled']:6.2f} | "
                f"Final Drive: {episode_info['final_drive']:.4f} | Avg Q: {avg_q:7.3f}"
            )

        if len(stage_success) == config.STAGE_ADVANCE_WINDOW:
            if float(np.mean(stage_success)) >= config.STAGE_ADVANCE_THRESHOLD:
                tqdm.write(
                    f"  [stage advance] '{stage_name}' cleared with "
                    f"{100.0 * float(np.mean(stage_success)):.1f}% success over the last "
                    f"{config.STAGE_ADVANCE_WINDOW} episodes."
                )
                return total_steps, episode + 1

    return total_steps, episode_offset + n_episodes


def summarize_final_eval(agent, full_eval_env, tracker, config):
    actor_ev = evaluate(agent, full_eval_env, use_target=False)
    target_ev = evaluate(agent, full_eval_env, use_target=True)
    print(
        "\n[final policy comparison] "
        f"actor: reward={actor_ev['reward']:.2f}, steps={actor_ev['steps']}, "
        f"final_drive={actor_ev['final_drive']:.3f}, survived={'yes' if actor_ev['survived'] else 'no'} | "
        f"actor_target: reward={target_ev['reward']:.2f}, steps={target_ev['steps']}, "
        f"final_drive={target_ev['final_drive']:.3f}, survived={'yes' if target_ev['survived'] else 'no'}"
    )
    if tracker["best_eval_policy"] is not None:
        print(
            f"[best checkpoint] {tracker['best_eval_policy']} at episode {tracker['best_eval_episode']} "
            f"during stage '{tracker['best_eval_stage']}' -> "
            f"{config.BEST_ACTOR_PATH}, {config.BEST_CRITIC_PATH}"
        )


def train(agent, env, config=Config):
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    tracker = make_tracker()
    full_eval_env = ContinuousHomeostaticEnv(
        config, resources=config.RESOURCES, regen_delay=config.REGEN_DELAY, survival_bonus=0.0
    )
    total_steps, _ = run_episodes(
        agent,
        env,
        writer,
        config,
        config.MAX_EPISODES,
        config.MAX_EPISODES,
        tracker,
        full_eval_env,
        stage_name="single-stage",
    )
    agent.save(config.ACTOR_PATH, config.CRITIC_PATH)
    writer.add_scalar("Run/TotalSteps", total_steps, config.MAX_EPISODES - 1)
    writer.close()
    summarize_final_eval(agent, full_eval_env, tracker, config)
    print(
        f"Training finished! Models saved to '{config.ACTOR_PATH}' and '{config.CRITIC_PATH}'."
    )


def train_curriculum(agent, config=Config):
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    tracker = make_tracker()
    total_episodes = sum(stage["episodes"] for stage in config.STAGES)
    total_steps = 0
    episode_offset = 0
    full_eval_env = ContinuousHomeostaticEnv(
        config, resources=config.RESOURCES, regen_delay=config.REGEN_DELAY, survival_bonus=0.0
    )

    for stage in config.STAGES:
        env = ContinuousHomeostaticEnv(
            config,
            resources=stage["resources"],
            regen_delay=stage.get("regen_delay", 0),
            survival_bonus=stage.get("survival_bonus", config.SURVIVAL_BONUS),
        )
        env.stage_noise_floor = stage.get("noise_floor", config.EXPLORATION_NOISE_FINAL)
        print(
            f"\n=== Curriculum stage '{stage['name']}': {env.n_resources} resources, "
            f"{stage['episodes']} episodes, regen_delay={env.regen_delay}, "
            f"survival_bonus={env.survival_bonus:.4f}, noise_floor={env.stage_noise_floor:.3f} ==="
        )
        total_steps, episode_offset = run_episodes(
            agent,
            env,
            writer,
            config,
            stage["episodes"],
            total_episodes,
            tracker,
            full_eval_env,
            episode_offset=episode_offset,
            total_steps=total_steps,
            stage_name=stage["name"],
        )

    agent.save(config.ACTOR_PATH, config.CRITIC_PATH)
    if episode_offset > 0:
        writer.add_scalar("Run/TotalSteps", total_steps, episode_offset - 1)
    writer.close()
    summarize_final_eval(agent, full_eval_env, tracker, config)
    print(
        f"Curriculum finished ({episode_offset} episodes)! Models saved to "
        f"'{config.ACTOR_PATH}' and '{config.CRITIC_PATH}'."
    )
