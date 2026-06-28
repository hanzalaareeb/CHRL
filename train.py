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


def stage3_reset_sigma(local_ep, config):
    decay_eps = max(1, int(config.STAGE3_EXPLORATION_RESET_EPISODES))
    frac = min(1.0, local_ep / decay_eps)
    return config.STAGE3_EXPLORATION_RESET + frac * (
        config.EXPLORATION_NOISE - config.STAGE3_EXPLORATION_RESET
    )


def stage_number(stage_name):
    digits = "".join(ch for ch in stage_name if ch.isdigit())
    return int(digits) if digits else 999


def evaluate(agent, env, use_target=False, n_episodes=1, seed_base=Config.SEED):
    """Run deterministic evaluation episodes and return averaged metrics."""
    rewards, stepss, drives, survivals = [], [], [], []
    consumptions, first_consumptions, entereds, consumed_anys, before_200s = [], [], [], [], []
    nearests, minimums = [], []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_base + ep)
        total_reward = 0.0
        steps = 0
        terminated = False
        info = {}
        done = False
        consumption_events = 0
        first_consumption_step = env.max_steps
        resource_entered = False
        nearest_resource_distance = float("inf")
        minimum_distance_reached = float("inf")

        while not done:
            action = agent.select_action(state, add_noise=False, use_target=use_target)
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            dists = np.asarray(info["dists"], dtype=np.float64)
            nearest_resource_distance = min(nearest_resource_distance, float(dists.min()))
            minimum_distance_reached = min(minimum_distance_reached, float(dists.min()))
            resource_entered = resource_entered or bool(np.any(info["within_radius"]))
            if np.any(info["delivered"] > 0.0):
                consumption_events += 1
                if first_consumption_step == env.max_steps:
                    first_consumption_step = steps

        rewards.append(float(total_reward))
        stepss.append(steps)
        drives.append(info.get("drive", float("nan")))
        survivals.append(float(not terminated))
        consumptions.append(float(consumption_events))
        first_consumptions.append(float(first_consumption_step))
        entereds.append(float(resource_entered))
        consumed_anys.append(float(consumption_events > 0))
        before_200s.append(float(first_consumption_step < threshold))
        nearests.append(float(nearest_resource_distance))
        minimums.append(float(minimum_distance_reached))

    return {
        "reward": float(np.mean(rewards)),
        "steps": float(np.mean(stepss)),
        "final_drive": float(np.mean(drives)),
        "survived": float(np.mean(survivals)),
        "avg_consumption": float(np.mean(consumptions)),
        "first_consumption": float(np.mean(first_consumptions)),
        "resource_entered": float(np.mean(entereds)),
        "consumed_any_rate": float(np.mean(consumed_anys)),
        "consumed_before_200_rate": float(np.mean(before_200s)),
        "nearest_resource_distance": float(np.mean(nearests)),
        "minimum_distance_reached": float(np.mean(minimums)),
    }


def save_policy_checkpoint(agent, actor_path, critic_path, use_target=False):
    """Save the chosen actor head together with the current critic."""
    torch.save(agent.actor_state_dict(use_target=use_target), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)


def stage_checkpoint_paths(config, stage_name):
    digits = "".join(ch for ch in stage_name if ch.isdigit())
    stage_id = digits or stage_name.lower().replace(" ", "_")
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    return (
        os.path.join(checkpoint_dir, f"stage{stage_id}_best_actor.pth"),
        os.path.join(checkpoint_dir, f"stage{stage_id}_best_critic.pth"),
    )


def stage_periodic_checkpoint_paths(config, stage_name, episode_number):
    digits = "".join(ch for ch in stage_name if ch.isdigit())
    stage_id = digits or stage_name.lower().replace(" ", "_")
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    return (
        os.path.join(checkpoint_dir, f"stage{stage_id}_ep{episode_number:04d}_actor.pth"),
        os.path.join(checkpoint_dir, f"stage{stage_id}_ep{episode_number:04d}_critic.pth"),
    )


def transition_checkpoint_paths(config, label):
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    safe = label.lower().replace(" ", "_").replace("-", "_")
    return (
        os.path.join(checkpoint_dir, f"{safe}_actor.pth"),
        os.path.join(checkpoint_dir, f"{safe}_critic.pth"),
    )


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
        "best_eval_consumption": -float("inf"),
        "stage_progress": {},
        "stage_best_scores": {},
        "success_history": deque(maxlen=Config.SUCCESS_RATE_WINDOW),
        "meaningful_stage2_indices": [],
    }


def ensure_stage_progress(tracker, stage_name):
    return tracker["stage_progress"].setdefault(
        stage_name,
        {
            "best_consumption": -float("inf"),
            "plateau_evals": 0,
        },
    )


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
    writer.add_scalar("Episode/EnteredResource", float(info["entered_resource"]), episode)
    writer.add_scalar("Episode/ConsumedBefore200", float(info["consumed_before_200"]), episode)
    writer.add_scalar("Episode/Survived500", float(info["survived_500"]), episode)
    writer.add_scalar("Episode/Survived1000", float(info["survived_1000"]), episode)
    writer.add_scalar("Homeostasis/Final_Drive", info["final_drive"], episode)
    writer.add_scalar("Homeostasis/Best_Drive", info["best_drive"], episode)
    writer.add_scalar("Replay/SuccessRatio", info["replay_success_ratio"], episode)
    writer.add_scalar("Replay/Size", info["replay_size"], episode)
    writer.add_scalar("Replay/AverageAge", info["replay_average_age"], episode)
    writer.add_scalar("Replay/Stage2Fraction", info["replay_stage2_fraction"], episode)
    writer.add_scalar("Success/RateLast100", info["success_rate_last100"], episode)
    writer.add_scalar("Success/LongestStreak", info["longest_survival_streak"], episode)
    writer.add_scalar("Success/EpisodesWithConsumption", info["episodes_with_consumption"], episode)
    writer.add_scalar("Success/EpisodesSurviving500", info["episodes_surviving_500"], episode)
    writer.add_scalar("Success/EpisodesSurviving1000", info["episodes_surviving_1000"], episode)
    writer.add_scalar("Homeostasis/AverageFinalDrive", info["average_final_drive"], episode)
    writer.add_scalar("Capability/EnteredResourceRate", info["entered_resource_rate"], episode)
    writer.add_scalar("Capability/ConsumedAnyRate", info["consumed_any_rate"], episode)
    writer.add_scalar("Capability/ConsumedBefore200Rate", info["consumed_before_200_rate"], episode)

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
        return {"should_stop": False, "evals": None}

    evals = {
        "stage_actor": evaluate(agent, stage_env, use_target=False, n_episodes=config.EVAL_EPISODES),
        "stage_actor_target": evaluate(agent, stage_env, use_target=True, n_episodes=config.EVAL_EPISODES),
        "full_actor": evaluate(agent, full_env, use_target=False, n_episodes=config.EVAL_EPISODES),
        "full_actor_target": evaluate(agent, full_env, use_target=True, n_episodes=config.EVAL_EPISODES),
    }

    for name, ev in evals.items():
        prefix = f"Eval/{stage_name}/{name}"
        writer.add_scalar(f"{prefix}/Reward", ev["reward"], episode)
        writer.add_scalar(f"{prefix}/Length", ev["steps"], episode)
        writer.add_scalar(f"{prefix}/FinalDrive", ev["final_drive"], episode)
        writer.add_scalar(f"{prefix}/Survived", ev["survived"], episode)
        writer.add_scalar(f"{prefix}/AvgConsumption", ev["avg_consumption"], episode)
        writer.add_scalar(f"{prefix}/FirstConsumption", ev["first_consumption"], episode)
        writer.add_scalar(f"{prefix}/ResourceEntered", ev["resource_entered"], episode)
        writer.add_scalar(f"{prefix}/ConsumedAnyRate", ev["consumed_any_rate"], episode)
        writer.add_scalar(f"{prefix}/ConsumedBefore200Rate", ev["consumed_before_200_rate"], episode)
        writer.add_scalar(f"{prefix}/NearestResource", ev["nearest_resource_distance"], episode)
        writer.add_scalar(f"{prefix}/MinimumDistance", ev["minimum_distance_reached"], episode)

    actor_ev = evals["full_actor"]
    target_ev = evals["full_actor_target"]
    actor_score = (actor_ev["avg_consumption"], actor_ev["survived"], -actor_ev["final_drive"])
    target_score = (target_ev["avg_consumption"], target_ev["survived"], -target_ev["final_drive"])
    best_name, best_ev, use_target = (
        ("actor_target", target_ev, True) if target_score > actor_score else ("actor", actor_ev, False)
    )
    best_score = target_score if use_target else actor_score
    if best_score > tracker["best_eval_score"]:
        tracker["best_eval_score"] = best_score
        tracker["best_eval_consumption"] = best_ev["avg_consumption"]
        tracker["best_eval_policy"] = best_name
        tracker["best_eval_stage"] = stage_name
        tracker["best_eval_episode"] = episode + 1
        save_policy_checkpoint(agent, config.BEST_ACTOR_PATH, config.BEST_CRITIC_PATH, use_target=use_target)

    stage_best = tracker["stage_best_scores"].get(stage_name, (-1.0, -float("inf"), -float("inf")))
    if best_score > stage_best:
        tracker["stage_best_scores"][stage_name] = best_score
        stage_actor_path, stage_critic_path = stage_checkpoint_paths(config, stage_name)
        save_policy_checkpoint(agent, stage_actor_path, stage_critic_path, use_target=use_target)

    stage_progress = ensure_stage_progress(tracker, stage_name)
    improved = best_ev["avg_consumption"] > (
        stage_progress["best_consumption"] + config.EARLY_STOPPING_MIN_DELTA
    )
    if improved:
        stage_progress["best_consumption"] = best_ev["avg_consumption"]
        stage_progress["plateau_evals"] = 0
    else:
        stage_progress["plateau_evals"] += 1

    tqdm.write(
        f"  [eval @ ep {episode + 1}] stage(cons actor={evals['stage_actor']['avg_consumption']:.2f}, "
        f"target={evals['stage_actor_target']['avg_consumption']:.2f}; "
        f"entered actor={100.0 * evals['stage_actor']['resource_entered']:.1f}%, "
        f"consumed actor={100.0 * evals['stage_actor']['consumed_any_rate']:.1f}%, "
        f"before200 actor={100.0 * evals['stage_actor']['consumed_before_200_rate']:.1f}%) | "
        f"full(cons actor={actor_ev['avg_consumption']:.2f}, target={target_ev['avg_consumption']:.2f}; "
        f"nearest actor={actor_ev['nearest_resource_distance']:.2f}, "
        f"target={target_ev['nearest_resource_distance']:.2f}) | "
        f"best={best_name}"
    )
    if tracker["best_eval_consumption"] > 0.0 and best_ev["avg_consumption"] < 0.5 * tracker["best_eval_consumption"]:
        tqdm.write(
            f"  [warning] performance collapse: current consumption={best_ev['avg_consumption']:.2f} "
            f"vs global best={tracker['best_eval_consumption']:.2f}"
        )
    should_stop = (
        config.EARLY_STOPPING
        and stage_number(stage_name) >= config.EARLY_STOPPING_MIN_STAGE
        and stage_progress["plateau_evals"] >= config.EARLY_STOPPING_PATIENCE
    )
    if should_stop:
        tqdm.write(
            f"  [early stop] avg consumption did not improve for "
            f"{stage_progress['plateau_evals']} evaluations in stage '{stage_name}'."
        )
    return {
        "should_stop": should_stop,
        "evals": evals,
        "best_name": best_name,
        "best_eval": best_ev,
        "stage_plateau_evals": stage_progress["plateau_evals"],
        "stage_best_consumption": stage_progress["best_consumption"],
    }


def log_stage_transition_eval(agent, writer, episode, from_stage_name, to_stage_name, full_env, config):
    actor_ev = evaluate(agent, full_env, use_target=False, n_episodes=config.EVAL_EPISODES)
    target_ev = evaluate(agent, full_env, use_target=True, n_episodes=config.EVAL_EPISODES)
    for name, ev in (("actor", actor_ev), ("actor_target", target_ev)):
        prefix = f"Transition/{from_stage_name}_to_{to_stage_name}/{name}"
        writer.add_scalar(f"{prefix}/Survived", ev["survived"], episode)
        writer.add_scalar(f"{prefix}/AvgConsumption", ev["avg_consumption"], episode)
        writer.add_scalar(f"{prefix}/ConsumedAnyRate", ev["consumed_any_rate"], episode)
        writer.add_scalar(f"{prefix}/ConsumedBefore200Rate", ev["consumed_before_200_rate"], episode)
        writer.add_scalar(f"{prefix}/NearestResource", ev["nearest_resource_distance"], episode)
        writer.add_scalar(f"{prefix}/MinimumDistance", ev["minimum_distance_reached"], episode)
        writer.add_scalar(f"{prefix}/FinalDrive", ev["final_drive"], episode)
    best_name, best_ev, use_target = (
        ("actor_target", target_ev, True)
        if (target_ev["avg_consumption"], target_ev["survived"], -target_ev["final_drive"])
        > (actor_ev["avg_consumption"], actor_ev["survived"], -actor_ev["final_drive"])
        else ("actor", actor_ev, False)
    )
    checkpoint_label = f"pre_{to_stage_name}"
    actor_path, critic_path = transition_checkpoint_paths(config, checkpoint_label)
    save_policy_checkpoint(agent, actor_path, critic_path, use_target=use_target)
    tqdm.write(
        f"  [transition eval {from_stage_name} -> {to_stage_name}] "
        f"actor(cons={actor_ev['avg_consumption']:.2f}, survived={actor_ev['survived']:.2f}, "
        f"nearest={actor_ev['nearest_resource_distance']:.2f}) | "
        f"target(cons={target_ev['avg_consumption']:.2f}, survived={target_ev['survived']:.2f}, "
        f"nearest={target_ev['nearest_resource_distance']:.2f}) | saved={best_name}"
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
    recent_rewards = deque(maxlen=config.EVAL_INTERVAL)
    recent_consumed = deque(maxlen=config.EVAL_INTERVAL)
    recent_entered = deque(maxlen=config.EVAL_INTERVAL)
    recent_before_200 = deque(maxlen=config.EVAL_INTERVAL)
    ensure_stage_progress(tracker, stage_name)
    stage_id = stage_number(stage_name)
    stage3_mix_pool = tracker.get("meaningful_stage2_indices", [])

    for local_ep in tqdm(range(n_episodes), desc=f"Stage {stage_name}"):
        episode = episode_offset + local_ep
        state, _ = env.reset()
        agent.noise.reset()

        base_sigma = exploration_sigma(episode, total_episodes, config)
        if stage_id == 3 and local_ep < config.STAGE3_EXPLORATION_RESET_EPISODES:
            base_sigma = max(base_sigma, stage3_reset_sigma(local_ep, config))
        noise_floor = env.config.EXPLORATION_NOISE_FINAL
        if hasattr(env, "stage_noise_floor"):
            noise_floor = max(noise_floor, env.stage_noise_floor)
        agent.noise.sigma = max(base_sigma, noise_floor)
        tqdm.write(
            f"[noise] episode={episode + 1} stage={stage_name} "
            f"sigma={agent.noise.sigma:.4f} floor={noise_floor:.4f}"
        )

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

            buffer_idx = agent.replay_buffer.add(
                state, action, reward, next_state, float(terminated), step_id=total_steps, stage_id=stage_id
            )
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
                preferred_indices = None
                preferred_fraction = 0.0
                if (
                    stage_id == 3
                    and local_ep < config.STAGE3_MIX_EPISODES
                    and len(stage3_mix_pool) > 0
                ):
                    preferred_indices = stage3_mix_pool
                    preferred_fraction = config.STAGE3_MIX_FRACTION
                losses = agent.update(
                    preferred_indices=preferred_indices,
                    preferred_fraction=preferred_fraction,
                )
                if losses is not None:
                    for key, value in losses.items():
                        loss_history[key].append(value)

        consumed_any = episode_metrics["consumption_events"] > 0
        entered_resource = episode_metrics["time_inside_resource_radius"] > 0
        consumed_before_200 = (
            0 <= episode_metrics["first_consumption_step"] < config.MEANINGFUL_CONSUMPTION_STEP
        )
        survived_500 = episode_steps >= 500
        survived_1000 = bool((not terminated) and truncated and episode_steps >= env.max_steps)
        success = consumed_any or entered_resource or consumed_before_200
        total_visits = episode_metrics["food_visits"] + episode_metrics["water_visits"]
        avg_consumption_per_visit = episode_metrics["total_consumption"] / total_visits if total_visits else 0.0
        unique_resources_visited = int(np.sum(episode_metrics["unique_resources_mask"]))

        if success:
            agent.replay_buffer.mark_success(episode_buffer_indices)
            agent.replay_buffer.save(config.BUFFER_PATH)
        if stage_id == 2 and success:
            tracker["meaningful_stage2_indices"].extend(episode_buffer_indices)

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
        recent_rewards.append(float(episode_reward))
        recent_consumed.append(int(consumed_any))
        recent_entered.append(int(entered_resource))
        recent_before_200.append(int(consumed_before_200))

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
            "entered_resource": entered_resource,
            "consumed_before_200": consumed_before_200,
            "survived_500": survived_500,
            "survived_1000": survived_1000,
            "final_drive": float(info.get("drive", float("nan"))),
            "best_drive": episode_metrics["best_drive"],
            "replay_success_ratio": agent.replay_buffer.success_ratio(),
            "replay_size": agent.replay_buffer.size,
            "replay_average_age": agent.replay_buffer.average_age(total_steps),
            "replay_stage2_fraction": agent.replay_buffer.stage_fraction(2),
            "success_rate_last100": float(np.mean(rolling_success)) if rolling_success else 0.0,
            "longest_survival_streak": tracker["longest_survival_streak"],
            "episodes_with_consumption": tracker["episodes_with_consumption"],
            "episodes_surviving_500": tracker["episodes_surviving_500"],
            "episodes_surviving_1000": tracker["episodes_surviving_1000"],
            "average_final_drive": float(np.nanmean(final_drives)),
            "entered_resource_rate": float(np.mean(recent_entered)) if recent_entered else 0.0,
            "consumed_any_rate": float(np.mean(recent_consumed)) if recent_consumed else 0.0,
            "consumed_before_200_rate": float(np.mean(recent_before_200)) if recent_before_200 else 0.0,
            "action_abs_sum": action_abs_sum,
            "nearest_food_sum": nearest_food_sum,
            "nearest_water_sum": nearest_water_sum,
            "states_sum": states_sum,
        }
        write_episode_metrics(writer, episode, episode_info, agent, loss_history, n_internal)

        writer.add_scalar(f"Stage/{stage_name}/SuccessRate", float(np.mean(stage_success)), episode)
        writer.add_scalar(f"Stage/{stage_name}/EpisodesCompleted", local_ep + 1, episode)

        eval_result = maybe_log_evals(agent, writer, episode, stage_name, env, full_eval_env, tracker, config)
        should_stop = eval_result["should_stop"]

        if (episode + 1) % config.LOG_INTERVAL == 0:
            q_hist = loss_history.get("q_value") or loss_history.get("q1") or []
            avg_q = np.mean(q_hist) if q_hist else float("nan")
            episodes_with_consumption = int(np.sum(recent_consumed))
            episodes_without_consumption = len(recent_consumed) - episodes_with_consumption
            tqdm.write(
                f"Episode {episode + 1:4d} | Reward: {episode_reward:8.2f} | "
                f"Steps: {episode_steps:4d} | Consumed: {episode_metrics['consumption_events']:4d} | "
                f"FirstConsume: {episode_info['first_consumption_step_for_log']:4d} | "
                f"Dist: {episode_metrics['distance_travelled']:6.2f} | "
                f"Final Drive: {episode_info['final_drive']:.4f} | Avg Q: {avg_q:7.3f}"
            )
            tqdm.write(
                f"  [log20] replay_size={agent.replay_buffer.size} | "
                f"success_transitions={100.0 * agent.replay_buffer.success_ratio():.1f}% | "
                f"episodes_with_consumption={episodes_with_consumption} | "
                f"episodes_without_consumption={episodes_without_consumption} | "
                f"entered_resource={100.0 * (float(np.mean(recent_entered)) if recent_entered else 0.0):.1f}% | "
                f"consumed_any={100.0 * (float(np.mean(recent_consumed)) if recent_consumed else 0.0):.1f}% | "
                f"consumed_before_200={100.0 * (float(np.mean(recent_before_200)) if recent_before_200 else 0.0):.1f}% | "
                f"avg_reward={float(np.mean(recent_rewards)) if recent_rewards else 0.0:.2f} | "
                f"replay_avg_age={agent.replay_buffer.average_age(total_steps):.1f}"
            )

        if len(stage_success) == config.STAGE_ADVANCE_WINDOW:
            if float(np.mean(stage_success)) >= config.STAGE_ADVANCE_THRESHOLD:
                tqdm.write(
                    f"  [stage advance] '{stage_name}' cleared with "
                    f"{100.0 * float(np.mean(stage_success)):.1f}% success over the last "
                    f"{config.STAGE_ADVANCE_WINDOW} episodes."
                )
                return total_steps, episode + 1

        stage3_early_stop_locked = (
            stage_id == 3 and (local_ep + 1) < config.STAGE3_MIN_EPISODES_BEFORE_EARLY_STOP
        )
        if should_stop and stage3_early_stop_locked:
            tqdm.write(
                f"  [early stop deferred] stage 3 has completed only {local_ep + 1} episodes; "
                f"minimum is {config.STAGE3_MIN_EPISODES_BEFORE_EARLY_STOP}."
            )
            should_stop = False

        if should_stop:
            return total_steps, episode + 1

        if (
            stage_number(stage_name) == 3
            and (local_ep + 1) % config.STAGE3_CHECKPOINT_INTERVAL == 0
        ):
            actor_path, critic_path = stage_periodic_checkpoint_paths(config, stage_name, episode + 1)
            agent.save(actor_path, critic_path)
            tqdm.write(
                f"  [stage3 checkpoint] saved episode {episode + 1} -> "
                f"{actor_path}, {critic_path}"
            )

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
    for stage_name in sorted(tracker["stage_best_scores"]):
        stage_actor_path, stage_critic_path = stage_checkpoint_paths(config, stage_name)
        print(f"[stage checkpoint] '{stage_name}' -> {stage_actor_path}, {stage_critic_path}")


def train(agent, env, config=Config):
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    tracker = make_tracker()
    tracker["stage_progress"]["single-stage"] = {
        "best_consumption": -float("inf"),
        "plateau_evals": 0,
    }
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

    for stage_idx, stage in enumerate(config.STAGES):
        tracker["stage_progress"][stage["name"]] = {
            "best_consumption": -float("inf"),
            "plateau_evals": 0,
        }
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
        if stage_number(stage["name"]) == 3:
            previous_stage_name = config.STAGES[max(0, stage_idx - 1)]["name"]
            log_stage_transition_eval(
                agent,
                writer,
                max(episode_offset - 1, 0),
                previous_stage_name,
                stage["name"],
                full_eval_env,
                config,
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
    threshold = getattr(env.config, "MEANINGFUL_CONSUMPTION_STEP", Config.MEANINGFUL_CONSUMPTION_STEP)
