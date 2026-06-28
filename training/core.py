import os
from collections import defaultdict, deque

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from config import Config
from env import ContinuousHomeostaticEnv
from training.checkpoints import (
    stage_checkpoint_paths,
    stage_periodic_checkpoint_paths,
    transition_checkpoint_paths,
)
from training.evaluation import evaluate
from training.logging import TrainingLogger
from training.metrics import EpisodeMetrics


def exploration_sigma(episode, total_episodes, config):
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


def save_policy_checkpoint(agent, actor_path, critic_path, use_target=False):
    torch.save(agent.actor_state_dict(use_target=use_target), actor_path)
    torch.save(agent.critic.state_dict(), critic_path)


def make_tracker():
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
        "meaningful_stage1_indices": [],
        "meaningful_stage2_indices": [],
        "reference_evals": {},
    }


def ensure_stage_progress(tracker, stage_name):
    return tracker["stage_progress"].setdefault(
        stage_name,
        {"best_consumption": -float("inf"), "plateau_evals": 0},
    )


def stage_success_mode(stage_name, config):
    for stage in getattr(config, "STAGES", []):
        if stage["name"] == stage_name:
            return stage.get("success_mode", "consumption")
    return "consumption"


def episode_success(stage_name, metrics, config):
    mode = stage_success_mode(stage_name, config)
    if mode == "navigation":
        return (
            metrics["entered_resource"]
            or metrics["consumed_any"]
            or metrics["minimum_distance_reached"] <= config.NAVIGATION_SUCCESS_DISTANCE
        )
    if mode == "dual_resource":
        return metrics["dual_resource_success"]
    if mode == "dual_resource_local":
        return metrics["dual_resource_success"] and metrics["same_cluster_dual_resource"]
    return metrics["replay_success"]


def build_episode_summary(metrics, env, info, tracker, total_steps, config):
    consumed_before_200 = metrics.consumed_before(config.MEANINGFUL_CONSUMPTION_STEP)
    consumed_food_before_threshold = metrics.consumed_food_before(config.DUAL_RESOURCE_SUCCESS_STEP_THRESHOLD)
    consumed_water_before_threshold = metrics.consumed_water_before(config.DUAL_RESOURCE_SUCCESS_STEP_THRESHOLD)
    dual_resource_success = (
        metrics.both_resources_consumed
        or metrics.food_to_water_success
        or metrics.water_to_food_success
        or (consumed_food_before_threshold and consumed_water_before_threshold)
    )
    replay_success = metrics.consumed_any or metrics.entered_resource or consumed_before_200
    initial_deficits = env.H_star - metrics.initial_internal_state
    start_dominant_need_type = int(np.argmax(initial_deficits))
    first_consumed_matches_dominant_need = (
        metrics.first_consumed_type >= 0 and metrics.first_consumed_type == start_dominant_need_type
    )
    summary = {
        "consumed_any": metrics.consumed_any,
        "entered_resource": metrics.entered_resource,
        "consumed_before_200": consumed_before_200,
        "consumed_food_before_threshold": consumed_food_before_threshold,
        "consumed_water_before_threshold": consumed_water_before_threshold,
        "dual_resource_success": dual_resource_success,
        "both_resources_consumed": metrics.both_resources_consumed,
        "survived_500": metrics.episode_steps >= 500,
        "survived_1000": bool(metrics.episode_steps >= env.max_steps),
        "first_food_step_for_log": metrics.first_food_step if metrics.first_food_step >= 0 else env.max_steps,
        "first_water_step_for_log": metrics.first_water_step if metrics.first_water_step >= 0 else env.max_steps,
        "first_consumption_step_for_log": metrics.first_consumption_step if metrics.first_consumption_step >= 0 else env.max_steps,
        "first_consumed_type_for_log": metrics.first_consumed_type,
        "first_consumed_matches_dominant_need": first_consumed_matches_dominant_need,
        "second_resource_nearby_ignored": metrics.second_resource_nearby_ignored,
        "first_opposite_resource_distance_for_log": (
            metrics.first_opposite_resource_distance if np.isfinite(metrics.first_opposite_resource_distance) else float(env.max_steps)
        ),
        "initial_food": float(metrics.initial_internal_state[0]),
        "initial_water": float(metrics.initial_internal_state[1]),
        "start_dominant_need_type": start_dominant_need_type,
        "replay_success": replay_success,
        "replay_success_ratio": tracker["replay_buffer"].success_ratio(),
        "replay_size": tracker["replay_buffer"].size,
        "replay_average_age": tracker["replay_buffer"].average_age(total_steps),
        "replay_stage2_fraction": tracker["replay_buffer"].stage_fraction(2),
        "replay_anchor_fraction": tracker["replay_buffer"].stage_fraction(1) + tracker["replay_buffer"].stage_fraction(2),
        "min_resource_distance": float(getattr(env, "current_min_resource_distance", float("nan"))),
        "cluster_left_visits": int(metrics.cluster_visit_counts[0]) if len(metrics.cluster_visit_counts) > 0 else 0,
        "cluster_right_visits": int(metrics.cluster_visit_counts[1]) if len(metrics.cluster_visit_counts) > 1 else 0,
        "same_cluster_dual_resource": metrics.same_cluster_dual_resource,
        "final_info": info,
    }
    return summary


def maybe_log_evals(agent, logger, episode, stage_name, stage_env, full_env, tracker, config):
    if (episode + 1) % config.EVAL_INTERVAL != 0:
        return {"should_stop": False, "evals": None, "collapse_detected": False}

    evals = {
        "stage_actor": evaluate(agent, stage_env, use_target=False, n_episodes=config.EVAL_EPISODES),
        "stage_actor_target": evaluate(agent, stage_env, use_target=True, n_episodes=config.EVAL_EPISODES),
        "full_actor": evaluate(agent, full_env, use_target=False, n_episodes=config.EVAL_EPISODES),
        "full_actor_target": evaluate(agent, full_env, use_target=True, n_episodes=config.EVAL_EPISODES),
    }
    reference = tracker["reference_evals"].get(stage_name)
    logger.log_evaluation(episode, stage_name, evals, reference=reference)

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
    improved = best_ev["avg_consumption"] > (stage_progress["best_consumption"] + config.EARLY_STOPPING_MIN_DELTA)
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
        f"target={target_ev['nearest_resource_distance']:.2f}) | best={best_name}"
    )

    collapse_detected = False
    if tracker["best_eval_consumption"] > 0.0 and best_ev["avg_consumption"] < 0.5 * tracker["best_eval_consumption"]:
        tqdm.write(
            f"  [warning] performance collapse: current consumption={best_ev['avg_consumption']:.2f} "
            f"vs global best={tracker['best_eval_consumption']:.2f}"
        )
    reference_fraction = config.STAGE4_REFERENCE_COLLAPSE_FRACTION
    if stage_number(stage_name) == 2:
        reference_fraction = config.STAGE2_REFERENCE_COLLAPSE_FRACTION
    if reference is not None:
        checks = [
            best_ev["avg_consumption"] >= reference_fraction * reference["avg_consumption"],
            best_ev["food_to_water_success"] >= reference_fraction * reference["food_to_water_success"],
            best_ev["water_to_food_success"] >= reference_fraction * reference["water_to_food_success"],
            best_ev["alternating_visit_count"] >= reference_fraction * reference["alternating_visit_count"],
        ]
        collapse_detected = not all(checks)
        if collapse_detected:
            tqdm.write(
                f"  [warning] {stage_name} below reference: "
                f"cons={best_ev['avg_consumption']:.2f}/{reference['avg_consumption']:.2f}, "
                f"f->w={100.0 * best_ev['food_to_water_success']:.1f}%/{100.0 * reference['food_to_water_success']:.1f}%, "
                f"w->f={100.0 * best_ev['water_to_food_success']:.1f}%/{100.0 * reference['water_to_food_success']:.1f}%, "
                f"alt={best_ev['alternating_visit_count']:.2f}/{reference['alternating_visit_count']:.2f}"
            )

    base_should_stop = (
        config.EARLY_STOPPING
        and stage_number(stage_name) >= config.EARLY_STOPPING_MIN_STAGE
        and stage_progress["plateau_evals"] >= config.EARLY_STOPPING_PATIENCE
    )
    stage2_should_stop = (
        config.STAGE2_EARLY_STOPPING
        and stage_number(stage_name) == 2
        and stage_progress["plateau_evals"] >= config.STAGE2_EARLY_STOPPING_PATIENCE
    )
    should_stop = base_should_stop or stage2_should_stop or collapse_detected
    if should_stop:
        tqdm.write(
            f"  [early stop] avg consumption did not improve for "
            f"{stage_progress['plateau_evals']} evaluations in stage '{stage_name}'."
        )
    return {
        "should_stop": should_stop,
        "evals": evals,
        "collapse_detected": collapse_detected,
    }


def log_stage_transition_eval(agent, logger, episode, from_stage_name, to_stage_name, full_env, config):
    actor_ev = evaluate(agent, full_env, use_target=False, n_episodes=config.EVAL_EPISODES)
    target_ev = evaluate(agent, full_env, use_target=True, n_episodes=config.EVAL_EPISODES)
    logger.log_stage_transition(episode, from_stage_name, to_stage_name, actor_ev, target_ev)
    best_name, _, use_target = (
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
    logger,
    config,
    n_episodes,
    total_episodes,
    tracker,
    full_eval_env,
    episode_offset=0,
    total_steps=0,
    stage_name="stage",
):
    resource_type = env.resource_type
    rolling_success = tracker["success_history"]
    stage_success = deque(maxlen=config.STAGE_ADVANCE_WINDOW)
    final_drives = []
    recent_rewards = deque(maxlen=config.EVAL_INTERVAL)
    recent_consumed = deque(maxlen=config.EVAL_INTERVAL)
    recent_entered = deque(maxlen=config.EVAL_INTERVAL)
    recent_before_200 = deque(maxlen=config.EVAL_INTERVAL)
    recent_dual_resource = deque(maxlen=config.EVAL_INTERVAL)
    recent_both = deque(maxlen=config.EVAL_INTERVAL)
    recent_food_first = deque(maxlen=config.EVAL_INTERVAL)
    recent_water_first = deque(maxlen=config.EVAL_INTERVAL)
    recent_first_match = deque(maxlen=config.EVAL_INTERVAL)
    recent_alt = deque(maxlen=config.EVAL_INTERVAL)
    ensure_stage_progress(tracker, stage_name)
    stage_id = stage_number(stage_name)
    stage3_mix_pool = tracker.get("meaningful_stage2_indices", [])
    stage1_mix_pool = tracker.get("meaningful_stage1_indices", [])

    for local_ep in tqdm(range(n_episodes), desc=f"Stage {stage_name}"):
        episode = episode_offset + local_ep
        state, _ = env.reset()
        agent.noise.reset()
        metrics = EpisodeMetrics(
            action_dim=agent.action_dim,
            n_internal=env.n_internal,
            n_resources=env.n_resources,
            max_steps=env.max_steps,
            initial_internal_state=env.internal_state.copy(),
        )
        prev_pos = env.agent_pos.copy()

        base_sigma = exploration_sigma(episode, total_episodes, config)
        if stage_id == 3 and local_ep < config.STAGE3_EXPLORATION_RESET_EPISODES:
            base_sigma = max(base_sigma, stage3_reset_sigma(local_ep, config))
        noise_floor = getattr(env, "stage_noise_floor", env.config.EXPLORATION_NOISE_FINAL)
        agent.noise.sigma = max(base_sigma, noise_floor)
        tqdm.write(
            f"[noise] episode={episode + 1} stage={stage_name} "
            f"sigma={agent.noise.sigma:.4f} floor={noise_floor:.4f}"
        )

        info = {}
        done = False
        loss_history = defaultdict(list)
        episode_buffer_indices = []

        while not done:
            action = env.action_space.sample() if total_steps < config.START_STEPS else agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            buffer_idx = agent.replay_buffer.add(
                state, action, reward, next_state, float(terminated), step_id=total_steps, stage_id=stage_id
            )
            episode_buffer_indices.append(buffer_idx)
            metrics.action_abs_sum += np.abs(action)
            prev_pos = metrics.update_from_step(reward, info, prev_pos, resource_type, action=action)

            state = next_state
            total_steps += 1

            if agent.replay_buffer.size > config.LEARNING_STARTS:
                preferred_indices = None
                preferred_fraction = 0.0
                if stage_id == 2 and local_ep < config.STAGE2_MIX_EPISODES and len(stage1_mix_pool) > 0:
                    preferred_indices = stage1_mix_pool
                    preferred_fraction = config.STAGE2_MIX_FRACTION
                elif stage_id == 3 and local_ep < config.STAGE3_MIX_EPISODES and len(stage3_mix_pool) > 0:
                    preferred_indices = stage3_mix_pool
                    preferred_fraction = config.STAGE3_MIX_FRACTION
                elif stage_id == 4 and local_ep < config.STAGE4_MIX_EPISODES and len(stage3_mix_pool) > 0:
                    preferred_indices = stage3_mix_pool
                    preferred_fraction = config.STAGE4_MIX_FRACTION
                losses = agent.update(preferred_indices=preferred_indices, preferred_fraction=preferred_fraction)
                if losses is not None:
                    for key, value in losses.items():
                        loss_history[key].append(value)

        summary = build_episode_summary(metrics, env, info, tracker, total_steps, config)
        success = episode_success(
            stage_name,
            {
                "entered_resource": summary["entered_resource"],
                "consumed_any": summary["consumed_any"],
                "minimum_distance_reached": metrics.minimum_distance_reached,
                "dual_resource_success": summary["dual_resource_success"],
                "same_cluster_dual_resource": summary["same_cluster_dual_resource"],
                "replay_success": summary["replay_success"],
            },
            config,
        )

        if summary["replay_success"]:
            agent.replay_buffer.mark_success(episode_buffer_indices)
            agent.replay_buffer.save(config.BUFFER_PATH)
        if stage_id == 1 and summary["replay_success"]:
            tracker["meaningful_stage1_indices"].extend(episode_buffer_indices)
        if stage_id == 2 and summary["replay_success"]:
            tracker["meaningful_stage2_indices"].extend(episode_buffer_indices)

        tracker["episodes_with_consumption"] += int(summary["consumed_any"])
        tracker["episodes_surviving_500"] += int(summary["survived_500"])
        tracker["episodes_surviving_1000"] += int(summary["survived_1000"])
        tracker["current_survival_streak"] = tracker["current_survival_streak"] + 1 if success else 0
        tracker["longest_survival_streak"] = max(tracker["longest_survival_streak"], tracker["current_survival_streak"])
        tracker["best_drive_achieved"] = min(tracker["best_drive_achieved"], metrics.best_drive)

        rolling_success.append(float(success))
        stage_success.append(float(success))
        final_drives.append(metrics.final_drive)
        recent_rewards.append(metrics.episode_reward)
        recent_consumed.append(int(summary["consumed_any"]))
        recent_entered.append(int(summary["entered_resource"]))
        recent_before_200.append(int(summary["consumed_before_200"]))
        recent_dual_resource.append(int(summary["dual_resource_success"]))
        recent_both.append(int(summary["both_resources_consumed"]))
        recent_food_first.append(summary["first_food_step_for_log"])
        recent_water_first.append(summary["first_water_step_for_log"])
        recent_first_match.append(int(summary["first_consumed_matches_dominant_need"]))
        recent_alt.append(int(metrics.alternating_visit_count))

        summary.update(
            {
                "success_rate_last100": float(np.mean(rolling_success)) if rolling_success else 0.0,
                "longest_survival_streak": tracker["longest_survival_streak"],
                "episodes_with_consumption": tracker["episodes_with_consumption"],
                "episodes_surviving_500": tracker["episodes_surviving_500"],
                "episodes_surviving_1000": tracker["episodes_surviving_1000"],
                "average_final_drive": float(np.nanmean(final_drives)),
                "entered_resource_rate": float(np.mean(recent_entered)) if recent_entered else 0.0,
                "consumed_any_rate": float(np.mean(recent_consumed)) if recent_consumed else 0.0,
                "consumed_before_200_rate": float(np.mean(recent_before_200)) if recent_before_200 else 0.0,
                "dual_resource_rate": float(np.mean(recent_dual_resource)) if recent_dual_resource else 0.0,
            }
        )
        logger.log_episode(episode, metrics, summary, agent, loss_history)
        logger.writer.add_scalar(f"Stage/{stage_name}/SuccessRate", float(np.mean(stage_success)), episode)
        logger.writer.add_scalar(f"Stage/{stage_name}/EpisodesCompleted", local_ep + 1, episode)

        eval_result = maybe_log_evals(agent, logger, episode, stage_name, env, full_eval_env, tracker, config)
        should_stop = eval_result["should_stop"]
        collapse_detected = bool(eval_result.get("collapse_detected", False))

        if (episode + 1) % config.LOG_INTERVAL == 0:
            q_hist = loss_history.get("q_value") or loss_history.get("q1") or []
            avg_q = np.mean(q_hist) if q_hist else float("nan")
            episodes_with_consumption = int(np.sum(recent_consumed))
            episodes_without_consumption = len(recent_consumed) - episodes_with_consumption
            tqdm.write(
                f"Episode {episode + 1:4d} | Reward: {metrics.episode_reward:8.2f} | "
                f"Steps: {metrics.episode_steps:4d} | Consumed: {metrics.consumption_events:4d} | "
                f"FirstConsume: {summary['first_consumption_step_for_log']:4d} | "
                f"Dist: {metrics.distance_travelled:6.2f} | "
                f"Final Drive: {metrics.final_drive:.4f} | Avg Q: {avg_q:7.3f}"
            )
            tqdm.write(
                f"  [log20] replay_size={agent.replay_buffer.size} | "
                f"success_transitions={100.0 * agent.replay_buffer.success_ratio():.1f}% | "
                f"episodes_with_consumption={episodes_with_consumption} | "
                f"episodes_without_consumption={episodes_without_consumption} | "
                f"entered_resource={100.0 * summary['entered_resource_rate']:.1f}% | "
                f"consumed_any={100.0 * summary['consumed_any_rate']:.1f}% | "
                f"consumed_before_200={100.0 * summary['consumed_before_200_rate']:.1f}% | "
                f"dual_resource={100.0 * summary['dual_resource_rate']:.1f}% | "
                f"avg_reward={float(np.mean(recent_rewards)) if recent_rewards else 0.0:.2f} | "
                f"replay_avg_age={agent.replay_buffer.average_age(total_steps):.1f}"
            )
        if stage_id == 2 and (local_ep + 1) % config.STAGE2_EXTRA_LOG_INTERVAL == 0:
            tqdm.write(
                f"  [stage2 detail] both_consumed={100.0 * float(np.mean(recent_both)):.1f}% | "
                f"first_food={float(np.mean(recent_food_first)):.1f} | "
                f"first_water={float(np.mean(recent_water_first)):.1f} | "
                f"first_matches_need={100.0 * float(np.mean(recent_first_match)):.1f}% | "
                f"alternating={float(np.mean(recent_alt)):.2f} | "
                f"heading_entropy={metrics.heading_entropy:.2f} | action_var={metrics.action_variance:.4f} | "
                f"ignored_nearby={100.0 * float(summary['second_resource_nearby_ignored']):.1f}% | "
                f"clusters=({summary['cluster_left_visits']},{summary['cluster_right_visits']}) | "
                f"spawn_min_dist={summary['min_resource_distance']:.2f}"
            )
            tqdm.write(
                f"  [stage2 replay] anchor_fraction={summary['replay_anchor_fraction']:.2f} | "
                f"stage2_fraction={summary['replay_stage2_fraction']:.2f}"
            )

        if len(stage_success) == config.STAGE_ADVANCE_WINDOW and float(np.mean(stage_success)) >= config.STAGE_ADVANCE_THRESHOLD:
            tqdm.write(
                f"  [stage advance] '{stage_name}' cleared with "
                f"{100.0 * float(np.mean(stage_success)):.1f}% success over the last "
                f"{config.STAGE_ADVANCE_WINDOW} episodes."
            )
            return total_steps, episode + 1

        stage3_early_stop_locked = stage_id == 3 and (local_ep + 1) < config.STAGE3_MIN_EPISODES_BEFORE_EARLY_STOP
        stage2_early_stop_locked = stage_id == 2 and (local_ep + 1) < config.STAGE2_MIN_EPISODES_BEFORE_EARLY_STOP
        if should_stop and stage3_early_stop_locked:
            tqdm.write(
                f"  [early stop deferred] stage 3 has completed only {local_ep + 1} episodes; "
                f"minimum is {config.STAGE3_MIN_EPISODES_BEFORE_EARLY_STOP}."
            )
            should_stop = False
        if should_stop and stage2_early_stop_locked:
            tqdm.write(
                f"  [early stop deferred] stage 2 has completed only {local_ep + 1} episodes; "
                f"minimum is {config.STAGE2_MIN_EPISODES_BEFORE_EARLY_STOP}."
            )
            should_stop = False

        if should_stop and collapse_detected:
            reference_stage_name = (
                config.STAGES[0]["name"] if stage_id == 2 else config.STAGES[2]["name"] if len(config.STAGES) >= 3 else stage_name
            )
            reference_actor_path, reference_critic_path = stage_checkpoint_paths(config, reference_stage_name)
            if os.path.exists(reference_actor_path) and os.path.exists(reference_critic_path):
                agent.load(reference_actor_path, reference_critic_path)
                tqdm.write(f"  [revert] restored reference checkpoint before leaving '{stage_name}'.")
        if should_stop:
            return total_steps, episode + 1

        periodic_interval = None
        periodic_label = None
        if stage_id == 3:
            periodic_interval = config.STAGE3_CHECKPOINT_INTERVAL
            periodic_label = "stage3"
        elif stage_id == 4:
            periodic_interval = config.STAGE4_CHECKPOINT_INTERVAL
            periodic_label = "stage4"
        if periodic_interval and (local_ep + 1) % periodic_interval == 0:
            actor_path, critic_path = stage_periodic_checkpoint_paths(config, stage_name, episode + 1)
            agent.save(actor_path, critic_path)
            tqdm.write(
                f"  [{periodic_label} checkpoint] saved episode {episode + 1} -> "
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
    logger = TrainingLogger(writer)
    tracker = make_tracker()
    tracker["replay_buffer"] = agent.replay_buffer
    tracker["stage_progress"]["single-stage"] = {"best_consumption": -float("inf"), "plateau_evals": 0}
    full_eval_env = ContinuousHomeostaticEnv(config, resources=config.RESOURCES, regen_delay=config.REGEN_DELAY, survival_bonus=0.0)
    total_steps, _ = run_episodes(
        agent, env, logger, config, config.MAX_EPISODES, config.MAX_EPISODES, tracker, full_eval_env, stage_name="single-stage"
    )
    agent.save(config.ACTOR_PATH, config.CRITIC_PATH)
    writer.add_scalar("Run/TotalSteps", total_steps, config.MAX_EPISODES - 1)
    writer.close()
    summarize_final_eval(agent, full_eval_env, tracker, config)
    print(f"Training finished! Models saved to '{config.ACTOR_PATH}' and '{config.CRITIC_PATH}'.")


def train_curriculum(agent, config=Config):
    writer = SummaryWriter(log_dir=config.LOG_DIR)
    logger = TrainingLogger(writer)
    tracker = make_tracker()
    tracker["replay_buffer"] = agent.replay_buffer
    total_episodes = sum(stage["episodes"] for stage in config.STAGES)
    total_steps = 0
    episode_offset = 0
    full_eval_env = ContinuousHomeostaticEnv(config, resources=config.RESOURCES, regen_delay=config.REGEN_DELAY, survival_bonus=0.0)

    for stage_idx, stage in enumerate(config.STAGES):
        tracker["stage_progress"][stage["name"]] = {"best_consumption": -float("inf"), "plateau_evals": 0}
        env = ContinuousHomeostaticEnv(
            config,
            resources=stage["resources"],
            regen_delay=stage.get("regen_delay", 0),
            survival_bonus=stage.get("survival_bonus", config.SURVIVAL_BONUS),
            resource_jitter=stage.get("resource_jitter", 0.0),
        )
        env.stage_noise_floor = stage.get("noise_floor", config.EXPLORATION_NOISE_FINAL)
        print(
            f"\n=== Curriculum stage '{stage['name']}': {env.n_resources} resources, "
            f"{stage['episodes']} episodes, regen_delay={env.regen_delay}, "
            f"survival_bonus={env.survival_bonus:.4f}, noise_floor={env.stage_noise_floor:.3f}, "
            f"resource_jitter={env.resource_jitter:.2f} ==="
        )
        if stage_number(stage["name"]) == 2:
            previous_stage_name = config.STAGES[max(0, stage_idx - 1)]["name"]
            reference_actor_path, reference_critic_path = stage_checkpoint_paths(config, previous_stage_name)
            if os.path.exists(reference_actor_path) and os.path.exists(reference_critic_path):
                agent.load(reference_actor_path, reference_critic_path)
                reference_eval = evaluate(agent, full_eval_env, use_target=False, n_episodes=config.EVAL_EPISODES)
                tracker["reference_evals"][stage["name"]] = reference_eval
                print(
                    f"  [stage2 init] loaded {previous_stage_name} best checkpoint; "
                    f"reference cons={reference_eval['avg_consumption']:.2f}, "
                    f"f->w={100.0 * reference_eval['food_to_water_success']:.1f}%, "
                    f"w->f={100.0 * reference_eval['water_to_food_success']:.1f}%, "
                    f"alt={reference_eval['alternating_visit_count']:.2f}"
                )
        if stage_number(stage["name"]) == 3:
            previous_stage_name = config.STAGES[max(0, stage_idx - 1)]["name"]
            log_stage_transition_eval(agent, logger, max(episode_offset - 1, 0), previous_stage_name, stage["name"], full_eval_env, config)
        if stage_number(stage["name"]) == 4:
            previous_stage_name = config.STAGES[max(0, stage_idx - 1)]["name"]
            reference_actor_path, reference_critic_path = stage_checkpoint_paths(config, previous_stage_name)
            if os.path.exists(reference_actor_path) and os.path.exists(reference_critic_path):
                agent.load(reference_actor_path, reference_critic_path)
                reference_eval = evaluate(agent, full_eval_env, use_target=False, n_episodes=config.EVAL_EPISODES)
                tracker["reference_evals"][stage["name"]] = reference_eval
                print(
                    f"  [stage4 init] loaded {previous_stage_name} best checkpoint; "
                    f"reference cons={reference_eval['avg_consumption']:.2f}, "
                    f"f->w={100.0 * reference_eval['food_to_water_success']:.1f}%, "
                    f"w->f={100.0 * reference_eval['water_to_food_success']:.1f}%, "
                    f"alt={reference_eval['alternating_visit_count']:.2f}"
                )
        total_steps, episode_offset = run_episodes(
            agent, env, logger, config, stage["episodes"], total_episodes, tracker, full_eval_env,
            episode_offset=episode_offset, total_steps=total_steps, stage_name=stage["name"],
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
