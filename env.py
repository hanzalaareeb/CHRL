import numpy as np
import gymnasium as gym
from gymnasium import spaces

from config import Config

# ======================================
# 1. Continuous Time/Space Homeostatic Env
# ======================================

# Map resource type names to the internal-state index they replenish.
TYPE_TO_IDX = {"food": 0, "water": 1}


class ContinuousHomeostaticEnv(gym.Env):
    """CTCS-HRRL environment based on Laurencon et al.

    The agent moves continuously in 2D and must keep ``N_INTERNAL`` homeostatic
    variables (food, water) near their set-point ``H*``. Multiple resource sites
    of each type are scattered in the world. Each site holds a **finite,
    depletable capacity** that **regenerates slowly**, so the agent cannot camp
    on a single site — it must rotate between sources.

    Follows the Gymnasium API: ``reset`` returns ``(obs, info)`` and ``step``
    returns ``(obs, reward, terminated, truncated, info)``.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=Config, resources=None, regen_delay=None, survival_bonus=None):
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
        # Largest possible distance between two points in the [-W, W]^2 world.
        self.max_distance = 2.0 * self.world_size * np.sqrt(2.0)

        # --- Internal physiology ---
        self.n_internal = config.N_INTERNAL
        self.H_star = np.full(self.n_internal, config.H_STAR, dtype=np.float64)
        self.decay_rates = np.asarray(config.DECAY_RATES, dtype=np.float64)
        self.init_state_low = config.INIT_STATE_LOW
        self.init_state_high = config.INIT_STATE_HIGH

        # --- Resource sites (per-stage list -> parallel arrays) ---
        specs = config.RESOURCES if resources is None else resources
        self.n_resources = len(specs)
        if self.n_resources > self.max_resources:
            raise ValueError(f"n_resources={self.n_resources} exceeds MAX_RESOURCES={self.max_resources}")
        self.resource_pos = np.array([[s[0], s[1]] for s in specs], dtype=np.float64)
        self.resource_type = np.array([TYPE_TO_IDX[s[2]] for s in specs], dtype=np.int64)
        self.resource_consume = np.array([s[3] for s in specs], dtype=np.float64)
        self.resource_max_cap = np.array([s[4] for s in specs], dtype=np.float64)
        self.resource_regen = np.array([s[5] for s in specs], dtype=np.float64)

        # Observation layout (padded to MAX_RESOURCES so the network input is fixed
        # across curriculum stages):
        #   [agent_x, agent_y]                                   (2)
        #   [internal states]                                    (n_internal)
        #   per slot: [dx, dy, distance, cap_frac, active]       (5 * MAX_RESOURCES)
        # dx/dy preserve DIRECTION ("move left"); distance gives magnitude; cap_frac
        # tells whether the store is depleted; active marks padded (inactive) slots.
        obs_dim = 2 + self.n_internal + 5 * self.max_resources
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(obs_dim,), dtype=np.float32,
        )

        # Action: [Velocity X, Velocity Y] - Continuous spatial movement
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32,
        )

        # Dynamic state (initialised in reset)
        self.agent_pos = np.zeros(2)
        self.internal_state = np.copy(self.H_star)
        self.resource_cap = np.copy(self.resource_max_cap)
        self.regen_cooldown = np.zeros(self.n_resources, dtype=np.int64)
        self.time_step = 0

    # ------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = np.zeros(2)
        # Start with randomly depleted internal states so there is an immediate
        # drive to reduce (denser early reward than starting at perfect homeostasis).
        self.internal_state = self.np_random.uniform(
            self.init_state_low, self.init_state_high, size=self.n_internal
        )
        # All resource stores start full; no regeneration cooldown pending.
        self.resource_cap = np.copy(self.resource_max_cap)
        self.regen_cooldown = np.zeros(self.n_resources, dtype=np.int64)
        self.time_step = 0
        return self._get_obs(), {}

    def _distances(self):
        return np.linalg.norm(self.resource_pos - self.agent_pos, axis=1)

    def _get_obs(self):
        rel = self.resource_pos - self.agent_pos                 # (M, 2) -> dx, dy
        dists = np.linalg.norm(rel, axis=1, keepdims=True)       # (M, 1)
        cap_frac = (self.resource_cap / self.resource_max_cap).reshape(-1, 1)  # (M, 1), already [0,1]

        if self.normalize_obs:
            pos = self.agent_pos / self.world_size               # [-1, 1]
            internal = (self.internal_state - self.H_star) / self.H_star  # deviation, ~[-1, 1]
            rel = rel / self.world_size                          # ~[-2, 2]
            dists = dists / self.max_distance                    # [0, 1]
        else:
            pos = self.agent_pos
            internal = self.internal_state

        # Per active resource: [dx, dy, distance, capacity_fraction, active=1]
        active = np.ones((self.n_resources, 1))
        per_resource = np.concatenate([rel, dists, cap_frac, active], axis=1)

        # Pad inactive slots up to MAX_RESOURCES with a neutral, "ignore me" encoding.
        pad = self.max_resources - self.n_resources
        if pad > 0:
            per_resource = np.vstack([per_resource, np.zeros((pad, 5))])

        return np.concatenate([pos, internal, per_resource.flatten()]).astype(np.float32)

    def drive(self, H):
        # Drive function D(H) = ||H - H*||^2 (squared distance to optimal state)
        return np.sum((H - self.H_star) ** 2)

    def _survival_bonus(self, drive, terminated):
        """State-dependent survival bonus (0 on death, vanishes as the agent starves)."""
        if terminated:
            return 0.0
        if self.survival_mode == "min_state":
            # Gated by the worst reserve -> forces balancing BOTH food and water.
            return self.survival_bonus * float(np.min(self.internal_state))
        # "exp_drive": maximal at the set-point (drive=0), -> 0 as drive grows.
        return self.survival_bonus * float(np.exp(-drive))

    # ------------------------------------------------------------------
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # 1. Record current physiological drive
        D_current = self.drive(self.internal_state)

        # 2. Continuous spatial kinematics (clipped to the world bounds)
        self.agent_pos = np.clip(
            self.agent_pos + action * self.dt, -self.world_size, self.world_size
        )

        # 3. Resource consumption with finite, depletable capacity
        dists = self._distances()
        within_radius = dists < self.consume_radius          # spatial presence (visits/stay)
        in_range = within_radius & (self.resource_cap > 0.0)  # can actually consume

        cap_before = self.resource_cap.copy()
        # Amount delivered this step is rate * dt, capped by remaining capacity.
        delivered = np.where(in_range, self.resource_consume * self.dt, 0.0)
        delivered = np.minimum(delivered, self.resource_cap)
        self.resource_cap = self.resource_cap - delivered

        # 4. Regeneration, with an optional cooldown after a store is fully depleted.
        #    A store that just emptied waits `regen_delay` steps before regenerating.
        just_depleted = (cap_before > 0.0) & (self.resource_cap <= 1e-9) & (self.regen_delay > 0)
        self.regen_cooldown = np.where(just_depleted, self.regen_delay, self.regen_cooldown)
        can_regen = self.regen_cooldown <= 0
        self.resource_cap = np.where(
            can_regen,
            np.minimum(self.resource_cap + self.resource_regen * self.dt, self.resource_max_cap),
            self.resource_cap,
        )
        self.regen_cooldown = np.maximum(self.regen_cooldown - 1, 0)

        # 5. Internal state dynamics: dH = -decay*dt + sum of delivered (by type)
        gain = np.zeros(self.n_internal)
        np.add.at(gain, self.resource_type, delivered)
        self.internal_state = self.internal_state - self.decay_rates * self.dt + gain
        self.internal_state = np.clip(self.internal_state, 0.0, 2.0)

        # 6. Reward: per-step drive reduction (D_t - D_{t+1}), minus effort, plus a
        #    STATE-DEPENDENT survival bonus. The bonus shrinks to 0 as the agent
        #    starves, so "minimise effort and die" is no longer a positive-return
        #    policy -- the agent must actually maintain homeostasis to keep earning it.
        D_next = self.drive(self.internal_state)
        drive_reduction = D_current - D_next
        effort = self.effort_penalty * np.sum(action ** 2)

        self.time_step += 1

        # Agent "dies" (terminated) if any essential internal state depletes to 0
        terminated = bool(np.any(self.internal_state <= 0.0))
        # Episode ends (truncated) on the time limit
        truncated = bool(self.time_step >= self.max_steps)

        survival = self._survival_bonus(D_next, terminated)
        reward = self.reward_scale * (drive_reduction - effort + survival)

        info = {
            "drive": D_next,
            "internal_state": self.internal_state.copy(),
            "agent_pos": self.agent_pos.copy(),
            "dists": dists.copy(),
            "delivered": delivered.copy(),          # per-resource amount consumed this step
            "consuming": bool(np.any(delivered > 0.0)),
            "within_radius": within_radius.copy(),  # per-resource spatial presence
            "resource_cap": self.resource_cap.copy(),
        }
        return self._get_obs(), float(reward), terminated, truncated, info
