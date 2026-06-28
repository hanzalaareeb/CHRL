import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config


TYPE_TO_IDX = {"food": 0, "water": 1}


class ContinuousHomeostaticEnv(gym.Env):
    """Continuous-time, continuous-space homeostatic environment."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        config=Config,
        resources=None,
        regen_delay=None,
        survival_bonus=None,
        resource_jitter=0.0,
    ):
        super().__init__()
        self.config = config
        self.dt = config.DT
        self.max_steps = config.MAX_STEPS
        self.effort_penalty = config.EFFORT_PENALTY
        self.reward_scale = config.REWARD_SCALE
        self.survival_bonus = config.SURVIVAL_BONUS if survival_bonus is None else survival_bonus
        self.survival_mode = config.SURVIVAL_BONUS_MODE
        self.world_size = config.WORLD_SIZE
        self.consume_radius = config.CONSUME_RADIUS
        self.normalize_obs = config.NORMALIZE_OBS
        self.max_resources = config.MAX_RESOURCES
        self.regen_delay = config.REGEN_DELAY if regen_delay is None else regen_delay
        self.resource_jitter = float(resource_jitter)
        self.min_spawn_distance = float(getattr(config, "RESOURCE_MIN_SPAWN_DISTANCE", 0.0))
        self.spawn_resample_attempts = int(getattr(config, "RESOURCE_SPAWN_RESAMPLE_ATTEMPTS", 1))
        self.max_distance = 2.0 * self.world_size * np.sqrt(2.0)

        self.n_internal = config.N_INTERNAL
        self.H_star = np.full(self.n_internal, config.H_STAR, dtype=np.float64)
        self.decay_rates = np.asarray(config.DECAY_RATES, dtype=np.float64)
        self.init_state_low = config.INIT_STATE_LOW
        self.init_state_high = config.INIT_STATE_HIGH

        specs = config.RESOURCES if resources is None else resources
        self.n_resources = len(specs)
        if self.n_resources > self.max_resources:
            raise ValueError(f"n_resources={self.n_resources} exceeds MAX_RESOURCES={self.max_resources}")
        self.base_resource_pos = np.array([[s[0], s[1]] for s in specs], dtype=np.float64)
        self.resource_pos = self.base_resource_pos.copy()
        self.resource_type = np.array([TYPE_TO_IDX[s[2]] for s in specs], dtype=np.int64)
        self.resource_consume = np.array([s[3] for s in specs], dtype=np.float64)
        self.resource_max_cap = np.array([s[4] for s in specs], dtype=np.float64)
        self.resource_regen = np.array([s[5] for s in specs], dtype=np.float64)
        self.current_min_resource_distance = self._minimum_pair_distance(self.resource_pos)

        obs_dim = 2 + self.n_internal + 5 * self.max_resources
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(obs_dim,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        self.agent_pos = np.zeros(2)
        self.internal_state = np.copy(self.H_star)
        self.resource_cap = np.copy(self.resource_max_cap)
        self.regen_cooldown = np.zeros(self.n_resources, dtype=np.int64)
        self.time_step = 0

    def _minimum_pair_distance(self, positions):
        if len(positions) < 2:
            return float("inf")
        deltas = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(deltas, axis=-1)
        upper = dists[np.triu_indices(len(positions), k=1)]
        return float(upper.min()) if upper.size else float("inf")

    def _sample_resource_positions(self):
        if self.resource_jitter <= 0.0:
            positions = self.base_resource_pos.copy()
            return positions, self._minimum_pair_distance(positions)

        best_positions = None
        best_distance = -float("inf")
        for _ in range(max(1, self.spawn_resample_attempts)):
            jitter = self.np_random.uniform(
                low=-self.resource_jitter, high=self.resource_jitter, size=self.base_resource_pos.shape
            )
            positions = np.clip(self.base_resource_pos + jitter, -self.world_size, self.world_size)
            min_distance = self._minimum_pair_distance(positions)
            if min_distance > best_distance:
                best_positions = positions
                best_distance = min_distance
            if min_distance >= self.min_spawn_distance:
                return positions, min_distance
        return best_positions, best_distance

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.zeros(2)
        self.internal_state = self.np_random.uniform(
            self.init_state_low, self.init_state_high, size=self.n_internal
        )
        self.resource_pos, self.current_min_resource_distance = self._sample_resource_positions()
        self.resource_cap = np.copy(self.resource_max_cap)
        self.regen_cooldown = np.zeros(self.n_resources, dtype=np.int64)
        self.time_step = 0
        return self._get_obs(), {}

    def _distances(self):
        return np.linalg.norm(self.resource_pos - self.agent_pos, axis=1)

    def _get_obs(self):
        rel = self.resource_pos - self.agent_pos
        dists = np.linalg.norm(rel, axis=1, keepdims=True)
        cap_frac = (self.resource_cap / self.resource_max_cap).reshape(-1, 1)

        if self.normalize_obs:
            pos = self.agent_pos / self.world_size
            internal = (self.internal_state - self.H_star) / self.H_star
            rel = rel / self.world_size
            dists = dists / self.max_distance
        else:
            pos = self.agent_pos
            internal = self.internal_state

        active = np.ones((self.n_resources, 1))
        per_resource = np.concatenate([rel, dists, cap_frac, active], axis=1)
        pad = self.max_resources - self.n_resources
        if pad > 0:
            per_resource = np.vstack([per_resource, np.zeros((pad, 5))])
        return np.concatenate([pos, internal, per_resource.flatten()]).astype(np.float32)

    def drive(self, H):
        return np.sum((H - self.H_star) ** 2)

    def _survival_bonus(self, drive, terminated):
        if terminated:
            return 0.0
        if self.survival_mode == "min_state":
            return self.survival_bonus * float(np.min(self.internal_state))
        return self.survival_bonus * float(np.exp(-drive))

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        D_current = self.drive(self.internal_state)
        self.agent_pos = np.clip(
            self.agent_pos + action * self.dt, -self.world_size, self.world_size
        )

        dists = self._distances()
        within_radius = dists < self.consume_radius
        in_range = within_radius & (self.resource_cap > 0.0)

        cap_before = self.resource_cap.copy()
        delivered = np.where(in_range, self.resource_consume * self.dt, 0.0)
        delivered = np.minimum(delivered, self.resource_cap)
        self.resource_cap = self.resource_cap - delivered

        just_depleted = (cap_before > 0.0) & (self.resource_cap <= 1e-9) & (self.regen_delay > 0)
        self.regen_cooldown = np.where(just_depleted, self.regen_delay, self.regen_cooldown)
        can_regen = self.regen_cooldown <= 0
        self.resource_cap = np.where(
            can_regen,
            np.minimum(self.resource_cap + self.resource_regen * self.dt, self.resource_max_cap),
            self.resource_cap,
        )
        self.regen_cooldown = np.maximum(self.regen_cooldown - 1, 0)

        gain = np.zeros(self.n_internal)
        np.add.at(gain, self.resource_type, delivered)
        self.internal_state = self.internal_state - self.decay_rates * self.dt + gain
        self.internal_state = np.clip(self.internal_state, 0.0, 2.0)

        D_next = self.drive(self.internal_state)
        drive_reduction = D_current - D_next
        effort = self.effort_penalty * np.sum(action ** 2)

        self.time_step += 1
        terminated = bool(np.any(self.internal_state <= 0.0))
        truncated = bool(self.time_step >= self.max_steps)

        survival = self._survival_bonus(D_next, terminated)
        reward = self.reward_scale * (drive_reduction - effort + survival)

        info = {
            "drive": D_next,
            "internal_state": self.internal_state.copy(),
            "agent_pos": self.agent_pos.copy(),
            "resource_pos": self.resource_pos.copy(),
            "dists": dists.copy(),
            "delivered": delivered.copy(),
            "consuming": bool(np.any(delivered > 0.0)),
            "within_radius": within_radius.copy(),
            "resource_cap": self.resource_cap.copy(),
            "min_resource_distance": self.current_min_resource_distance,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
