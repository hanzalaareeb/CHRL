import numpy as np

from training.metrics import INTERNAL_LABELS


LOSS_TAGS = {
    "actor_loss": "Loss/Actor",
    "critic_loss": "Loss/Critic",
    "q_value": "Q/Average",
    "q1": "Q/Q1",
    "q2": "Q/Q2",
    "q_diff": "Q/Q1_minus_Q2",
}


class TrainingLogger:
    def __init__(self, writer):
        self.writer = writer

    def log_episode(self, episode, metrics, summary, agent, loss_history):
        scalar_map = {
            "Reward/Episode": metrics.episode_reward,
            "Episode/Length": metrics.episode_steps,
            "Episode/ConsumptionEvents": metrics.consumption_events,
            "Episode/DistanceTravelled": metrics.distance_travelled,
            "Episode/UniqueResourcesVisited": metrics.unique_resources_visited,
            "Episode/TimeInsideRadius": metrics.time_inside_resource_radius,
            "Episode/FoodVisits": metrics.food_visits,
            "Episode/WaterVisits": metrics.water_visits,
            "Episode/AvgConsumptionPerVisit": metrics.avg_consumption_per_visit,
            "Episode/FirstFoodStep": summary["first_food_step_for_log"],
            "Episode/FirstWaterStep": summary["first_water_step_for_log"],
            "Episode/FirstConsumptionStep": summary["first_consumption_step_for_log"],
            "Episode/FirstConsumedType": summary["first_consumed_type_for_log"],
            "Episode/FirstConsumedMatchesDominantNeed": float(summary["first_consumed_matches_dominant_need"]),
            "Episode/SecondResourceNearbyIgnored": float(summary["second_resource_nearby_ignored"]),
            "Episode/FirstOppositeResourceDistance": summary["first_opposite_resource_distance_for_log"],
            "Episode/InitialFood": summary["initial_food"],
            "Episode/InitialWater": summary["initial_water"],
            "Episode/FinalFood": metrics.final_food,
            "Episode/FinalWater": metrics.final_water,
            "Episode/StartDominantNeedType": summary["start_dominant_need_type"],
            "Episode/BothResourcesConsumed": float(summary["both_resources_consumed"]),
            "Episode/FoodToWaterSuccess": float(metrics.food_to_water_success),
            "Episode/WaterToFoodSuccess": float(metrics.water_to_food_success),
            "Episode/AlternatingVisitCount": metrics.alternating_visit_count,
            "Episode/ConsumedAny": float(summary["consumed_any"]),
            "Episode/EnteredResource": float(summary["entered_resource"]),
            "Episode/ConsumedBefore200": float(summary["consumed_before_200"]),
            "Episode/ConsumedFoodBeforeThreshold": float(summary["consumed_food_before_threshold"]),
            "Episode/ConsumedWaterBeforeThreshold": float(summary["consumed_water_before_threshold"]),
            "Episode/DualResourceSuccess": float(summary["dual_resource_success"]),
            "Episode/Survived500": float(summary["survived_500"]),
            "Episode/Survived1000": float(summary["survived_1000"]),
            "Homeostasis/Final_Drive": metrics.final_drive,
            "Homeostasis/Best_Drive": metrics.best_drive,
            "Replay/SuccessRatio": summary["replay_success_ratio"],
            "Replay/Size": summary["replay_size"],
            "Replay/AverageAge": summary["replay_average_age"],
            "Replay/Stage2Fraction": summary["replay_stage2_fraction"],
            "Replay/AnchorFraction": summary["replay_anchor_fraction"],
            "Success/RateLast100": summary["success_rate_last100"],
            "Success/DualResourceRateLastWindow": summary["dual_resource_rate"],
            "Success/LongestStreak": summary["longest_survival_streak"],
            "Success/EpisodesWithConsumption": summary["episodes_with_consumption"],
            "Success/EpisodesSurviving500": summary["episodes_surviving_500"],
            "Success/EpisodesSurviving1000": summary["episodes_surviving_1000"],
            "Homeostasis/AverageFinalDrive": summary["average_final_drive"],
            "Capability/EnteredResourceRate": summary["entered_resource_rate"],
            "Capability/ConsumedAnyRate": summary["consumed_any_rate"],
            "Capability/ConsumedBefore200Rate": summary["consumed_before_200_rate"],
            "Environment/MinResourceDistance": summary["min_resource_distance"],
            "Action/Variance": metrics.action_variance,
            "Action/HeadingEntropy": metrics.heading_entropy,
            "Stage2/ClusterLeftVisits": float(summary["cluster_left_visits"]),
            "Stage2/ClusterRightVisits": float(summary["cluster_right_visits"]),
            "Action/Average": np.mean(metrics.action_abs_sum) / max(metrics.episode_steps, 1),
            "Exploration/Sigma": agent.noise.sigma,
            "Distance/Nearest_Food": metrics.nearest_food_sum / max(metrics.episode_steps, 1),
            "Distance/Nearest_Water": metrics.nearest_water_sum / max(metrics.episode_steps, 1),
        }
        for tag, value in scalar_map.items():
            self.writer.add_scalar(tag, value, episode)

        for i in range(metrics.n_internal):
            label = INTERNAL_LABELS[i] if i < len(INTERNAL_LABELS) else f"State{i + 1}"
            self.writer.add_scalar(
                f"InternalState/{label}",
                metrics.states_sum[i] / max(metrics.episode_steps, 1),
                episode,
            )

        for key, values in loss_history.items():
            if values:
                tag = LOSS_TAGS.get(key, f"Loss/{key}")
                self.writer.add_scalar(tag, np.mean(values), episode)

    def log_evaluation(self, episode, stage_name, evals, reference=None):
        for name, ev in evals.items():
            prefix = f"Eval/{stage_name}/{name}"
            for metric_name, value in ev.items():
                tag = "".join(part.title() for part in metric_name.split("_"))
                self.writer.add_scalar(f"{prefix}/{tag}", value, episode)

        if reference is not None:
            for name, ev in (("actor", evals["full_actor"]), ("actor_target", evals["full_actor_target"])):
                prefix = f"Reference/{stage_name}/{name}"
                self.writer.add_scalar(
                    f"{prefix}/ConsumptionRatio",
                    ev["avg_consumption"] / max(reference["avg_consumption"], 1e-6),
                    episode,
                )
                self.writer.add_scalar(
                    f"{prefix}/FoodToWaterRatio",
                    ev["food_to_water_success"] / max(reference["food_to_water_success"], 1e-6),
                    episode,
                )
                self.writer.add_scalar(
                    f"{prefix}/WaterToFoodRatio",
                    ev["water_to_food_success"] / max(reference["water_to_food_success"], 1e-6),
                    episode,
                )
                self.writer.add_scalar(
                    f"{prefix}/AlternatingRatio",
                    ev["alternating_visit_count"] / max(reference["alternating_visit_count"], 1e-6),
                    episode,
                )

    def log_stage_transition(self, episode, from_stage_name, to_stage_name, actor_ev, target_ev):
        for name, ev in (("actor", actor_ev), ("actor_target", target_ev)):
            prefix = f"Transition/{from_stage_name}_to_{to_stage_name}/{name}"
            for metric_name, value in ev.items():
                tag = "".join(part.title() for part in metric_name.split("_"))
                self.writer.add_scalar(f"{prefix}/{tag}", value, episode)
