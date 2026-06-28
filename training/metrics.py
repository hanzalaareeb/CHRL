from dataclasses import dataclass, field

import numpy as np

from config import Config


INTERNAL_LABELS = ["Food", "Water"]


@dataclass
class EpisodeMetrics:
    action_dim: int
    n_internal: int
    n_resources: int
    max_steps: int
    initial_internal_state: np.ndarray
    episode_reward: float = 0.0
    episode_steps: int = 0
    consumption_events: int = 0
    distance_travelled: float = 0.0
    minimum_distance_reached: float = float("inf")
    food_visits: int = 0
    water_visits: int = 0
    total_consumption: float = 0.0
    first_food_step: int = -1
    first_water_step: int = -1
    first_consumption_step: int = -1
    time_inside_resource_radius: int = 0
    best_drive: float = float("inf")
    first_consumed_type: int = -1
    action_abs_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    states_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    nearest_food_sum: float = 0.0
    nearest_water_sum: float = 0.0
    action_sq_sum: np.ndarray = field(default_factory=lambda: np.zeros(0))
    action_heading_hist: np.ndarray = field(default_factory=lambda: np.zeros(8))
    final_food: float = float("nan")
    final_water: float = float("nan")
    final_drive: float = float("nan")
    unique_resources_mask: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    prev_within: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    consumed_sequence: list[int] = field(default_factory=list)
    consumed_resource_indices: list[int] = field(default_factory=list)
    first_consumed_resource_idx: int = -1
    first_opposite_resource_distance: float = float("inf")
    second_resource_nearby_ignored: bool = False
    cluster_visit_counts: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=int))

    def __post_init__(self):
        self.action_abs_sum = np.zeros(self.action_dim)
        self.action_sq_sum = np.zeros(self.action_dim)
        self.states_sum = np.zeros(self.n_internal)
        self.unique_resources_mask = np.zeros(self.n_resources, dtype=bool)
        self.prev_within = np.zeros(self.n_resources, dtype=bool)
        self.action_heading_hist = np.zeros(8, dtype=np.int64)
        self.cluster_visit_counts = np.zeros(2, dtype=np.int64)

    def update_from_step(self, reward, info, prev_pos, resource_type, action=None):
        current_pos = np.asarray(info["agent_pos"], dtype=np.float64)
        delivered = np.asarray(info["delivered"], dtype=np.float64)
        within = np.asarray(info["within_radius"], dtype=bool)
        dists = np.asarray(info["dists"], dtype=np.float64)
        rising = within & ~self.prev_within
        resource_pos = np.asarray(info.get("resource_pos")) if info.get("resource_pos") is not None else None

        food_mask = resource_type == 0
        water_mask = resource_type == 1

        self.episode_reward += float(reward)
        self.episode_steps += 1
        self.distance_travelled += float(np.linalg.norm(current_pos - prev_pos))
        self.minimum_distance_reached = min(self.minimum_distance_reached, float(np.min(dists)))
        self.food_visits += int(np.sum(rising[food_mask]))
        self.water_visits += int(np.sum(rising[water_mask]))
        self.consumption_events += int(np.any(delivered > 0.0))
        self.total_consumption += float(np.sum(delivered))
        self.time_inside_resource_radius += int(np.any(within))
        self.unique_resources_mask |= within
        self.best_drive = min(self.best_drive, float(info["drive"]))
        self.states_sum += np.asarray(info["internal_state"], dtype=np.float64)
        self.final_food = float(info["internal_state"][0])
        self.final_water = float(info["internal_state"][1])
        self.final_drive = float(info["drive"])

        self.nearest_food_sum += float(dists[food_mask].min())
        self.nearest_water_sum += float(dists[water_mask].min())
        if action is not None:
            action = np.asarray(action, dtype=np.float64)
            self.action_sq_sum += action ** 2
            if np.linalg.norm(action) > 1e-8:
                angle = np.arctan2(action[1], action[0])
                bin_idx = int(np.floor(((angle + np.pi) / (2 * np.pi)) * len(self.action_heading_hist))) % len(self.action_heading_hist)
                self.action_heading_hist[bin_idx] += 1

        if resource_pos is not None and np.any(rising):
            rising_indices = np.where(rising)[0]
            for idx in rising_indices:
                cluster_idx = 1 if resource_pos[idx, 0] >= 0.0 else 0
                self.cluster_visit_counts[cluster_idx] += 1

        if self.first_food_step < 0 and np.any(delivered[food_mask] > 0.0):
            self.first_food_step = self.episode_steps
        if self.first_water_step < 0 and np.any(delivered[water_mask] > 0.0):
            self.first_water_step = self.episode_steps
        if self.first_consumption_step < 0 and np.any(delivered > 0.0):
            self.first_consumption_step = self.episode_steps

        consumed_mask = delivered > 0.0
        if np.any(consumed_mask):
            consumed_indices = np.where(consumed_mask)[0]
            consumed_types = sorted(set(resource_type[consumed_mask].tolist()))
            if self.first_consumed_resource_idx < 0:
                self.first_consumed_resource_idx = int(consumed_indices[0])
                first_type = int(resource_type[self.first_consumed_resource_idx])
                opposite_mask = resource_type != first_type
                self.first_opposite_resource_distance = float(np.min(dists[opposite_mask])) if np.any(opposite_mask) else float("inf")
            if self.first_consumed_type < 0 and consumed_types:
                self.first_consumed_type = int(consumed_types[0])
            for consumed_type in consumed_types:
                if not self.consumed_sequence or self.consumed_sequence[-1] != consumed_type:
                    self.consumed_sequence.append(int(consumed_type))
        if self.first_consumed_type >= 0 and self.first_opposite_resource_distance < float("inf"):
            opposite_consumed = (
                (self.first_consumed_type == 0 and self.first_water_step >= 0)
                or (self.first_consumed_type == 1 and self.first_food_step >= 0)
            )
            self.second_resource_nearby_ignored = (self.first_opposite_resource_distance <= 2.0) and (not opposite_consumed)

        self.prev_within = within
        return current_pos

    @property
    def consumed_any(self):
        return self.consumption_events > 0

    @property
    def entered_resource(self):
        return self.time_inside_resource_radius > 0

    def consumed_before(self, threshold):
        return 0 <= self.first_consumption_step < threshold

    def consumed_food_before(self, threshold):
        return 0 <= self.first_food_step < threshold

    def consumed_water_before(self, threshold):
        return 0 <= self.first_water_step < threshold

    @property
    def both_resources_consumed(self):
        return self.first_food_step >= 0 and self.first_water_step >= 0

    @property
    def food_to_water_success(self):
        return any(
            self.consumed_sequence[i] == 0 and self.consumed_sequence[i + 1] == 1
            for i in range(len(self.consumed_sequence) - 1)
        )

    @property
    def water_to_food_success(self):
        return any(
            self.consumed_sequence[i] == 1 and self.consumed_sequence[i + 1] == 0
            for i in range(len(self.consumed_sequence) - 1)
        )

    @property
    def alternating_visit_count(self):
        return max(0, len(self.consumed_sequence) - 1)

    @property
    def unique_resources_visited(self):
        return int(np.sum(self.unique_resources_mask))

    @property
    def avg_consumption_per_visit(self):
        total_visits = self.food_visits + self.water_visits
        return self.total_consumption / total_visits if total_visits else 0.0

    @property
    def action_variance(self):
        mean_sq = self.action_sq_sum / max(self.episode_steps, 1)
        mean = self.action_abs_sum / max(self.episode_steps, 1)
        return float(np.mean(np.maximum(mean_sq - mean ** 2, 0.0)))

    @property
    def heading_entropy(self):
        total = float(np.sum(self.action_heading_hist))
        if total <= 0:
            return 0.0
        p = self.action_heading_hist / total
        p = p[p > 0]
        return float(-np.sum(p * np.log(p + 1e-12)))

    @property
    def same_cluster_dual_resource(self):
        return self.both_resources_consumed and self.first_opposite_resource_distance <= 2.0


@dataclass
class EvaluationAccumulator:
    rewards: list[float] = field(default_factory=list)
    steps: list[float] = field(default_factory=list)
    final_drives: list[float] = field(default_factory=list)
    survivals: list[float] = field(default_factory=list)
    consumptions: list[float] = field(default_factory=list)
    first_consumptions: list[float] = field(default_factory=list)
    entered: list[float] = field(default_factory=list)
    consumed_any: list[float] = field(default_factory=list)
    before_threshold: list[float] = field(default_factory=list)
    nearest: list[float] = field(default_factory=list)
    minimum: list[float] = field(default_factory=list)
    first_food: list[float] = field(default_factory=list)
    first_water: list[float] = field(default_factory=list)
    food_to_water: list[float] = field(default_factory=list)
    water_to_food: list[float] = field(default_factory=list)
    alternating: list[float] = field(default_factory=list)

    def add_episode(self, episode_metrics, terminated, nearest_resource_distance):
        threshold = Config.MEANINGFUL_CONSUMPTION_STEP
        self.rewards.append(float(episode_metrics.episode_reward))
        self.steps.append(float(episode_metrics.episode_steps))
        self.final_drives.append(float(episode_metrics.final_drive))
        self.survivals.append(float(not terminated))
        self.consumptions.append(float(episode_metrics.consumption_events))
        self.first_consumptions.append(
            float(episode_metrics.first_consumption_step if episode_metrics.first_consumption_step >= 0 else episode_metrics.max_steps)
        )
        self.entered.append(float(episode_metrics.entered_resource))
        self.consumed_any.append(float(episode_metrics.consumed_any))
        self.before_threshold.append(float(episode_metrics.consumed_before(threshold)))
        self.nearest.append(float(nearest_resource_distance))
        self.minimum.append(float(episode_metrics.minimum_distance_reached))
        self.first_food.append(float(episode_metrics.first_food_step if episode_metrics.first_food_step >= 0 else episode_metrics.max_steps))
        self.first_water.append(float(episode_metrics.first_water_step if episode_metrics.first_water_step >= 0 else episode_metrics.max_steps))
        self.food_to_water.append(float(episode_metrics.food_to_water_success))
        self.water_to_food.append(float(episode_metrics.water_to_food_success))
        self.alternating.append(float(episode_metrics.alternating_visit_count))

    def summary(self):
        return {
            "reward": float(np.mean(self.rewards)),
            "steps": float(np.mean(self.steps)),
            "final_drive": float(np.mean(self.final_drives)),
            "survived": float(np.mean(self.survivals)),
            "avg_consumption": float(np.mean(self.consumptions)),
            "first_consumption": float(np.mean(self.first_consumptions)),
            "resource_entered": float(np.mean(self.entered)),
            "consumed_any_rate": float(np.mean(self.consumed_any)),
            "consumed_before_200_rate": float(np.mean(self.before_threshold)),
            "nearest_resource_distance": float(np.mean(self.nearest)),
            "minimum_distance_reached": float(np.mean(self.minimum)),
            "first_food_step": float(np.mean(self.first_food)),
            "first_water_step": float(np.mean(self.first_water)),
            "food_to_water_success": float(np.mean(self.food_to_water)),
            "water_to_food_success": float(np.mean(self.water_to_food)),
            "alternating_visit_count": float(np.mean(self.alternating)),
        }
