import numpy as np
import torch

from config import Config


# ==========================================
# 3. Experience Replay Buffer (with simple prioritized sampling)
# ==========================================
class ReplayBuffer:
    """Experience replay with an optional, lightweight prioritized sampling.

    Priority ~ (|reward| + eps); transitions from successful episodes are boosted
    so good rollouts are replayed more often. This is a cheap approximation of
    PER (no sum-tree, no importance-sampling correction) — enough to stop a single
    rare successful episode from being drowned out by many "walk into death" ones.
    """

    def __init__(self, state_dim, action_dim, max_size=Config.BUFFER_SIZE, config=Config):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.config = config
        self.prioritized = config.PRIORITIZED_REPLAY
        self.alpha = config.PER_ALPHA
        self.epsilon = config.PER_EPSILON

        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)
        self.priorities = np.ones(max_size, dtype=np.float64)
        self.success = np.zeros(max_size, dtype=np.float32)  # 1 if from a successful episode
        self.step_ids = np.zeros(max_size, dtype=np.int64)
        self.stage_ids = np.zeros(max_size, dtype=np.int64)

    def add(self, state, action, reward, next_state, done, step_id=0, stage_id=0):
        idx = self.ptr
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.priorities[idx] = abs(float(reward)) + self.epsilon
        self.success[idx] = 0.0
        self.step_ids[idx] = int(step_id)
        self.stage_ids[idx] = int(stage_id)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        return idx

    def update_priorities(self, indices, priorities):
        """Refresh replay priorities from TD error or another proxy signal."""
        if len(indices) == 0:
            return
        indices = np.asarray(indices, dtype=np.int64)
        priorities = np.asarray(priorities, dtype=np.float64)
        self.priorities[indices] = np.maximum(priorities, self.epsilon)

    def mark_success(self, indices):
        """Flag transitions from a successful episode and boost their priority."""
        if len(indices) == 0:
            return
        indices = np.asarray(indices)
        self.success[indices] = 1.0
        self.priorities[indices] *= self.config.SUCCESS_PRIORITY_BOOST

    def success_ratio(self):
        """Fraction of stored transitions that came from successful episodes."""
        return float(self.success[:self.size].mean()) if self.size else 0.0

    def stage_fraction(self, stage_id):
        if self.size == 0:
            return 0.0
        return float(np.mean(self.stage_ids[:self.size] == int(stage_id)))

    def average_age(self, current_step):
        """Average transition age in environment steps."""
        if self.size == 0:
            return 0.0
        ages = max(int(current_step), 0) - self.step_ids[:self.size]
        return float(np.mean(np.maximum(ages, 0)))

    def _sample_indices_from_pool(self, pool, count):
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        pool = np.asarray(pool, dtype=np.int64)
        if pool.size == 0:
            return np.empty(0, dtype=np.int64)
        replace = pool.size < count
        if self.prioritized:
            weights = self.priorities[pool] ** self.alpha
            weights = weights / weights.sum()
            return np.random.choice(pool, size=count, replace=replace, p=weights)
        return np.random.choice(pool, size=count, replace=replace)

    def sample(self, batch_size, preferred_indices=None, preferred_fraction=0.0):
        preferred_count = int(round(batch_size * preferred_fraction))
        preferred_count = max(0, min(batch_size, preferred_count))
        preferred = self._sample_indices_from_pool(preferred_indices, preferred_count)
        remaining = batch_size - len(preferred)
        if self.prioritized:
            p = self.priorities[:self.size] ** self.alpha
            p /= p.sum()
            fallback = np.random.choice(self.size, size=remaining, p=p)
        else:
            fallback = np.random.randint(0, self.size, size=remaining)
        ind = np.concatenate([preferred, np.asarray(fallback, dtype=np.int64)])
        return (
            torch.FloatTensor(self.states[ind]),
            torch.FloatTensor(self.actions[ind]),
            torch.FloatTensor(self.rewards[ind]),
            torch.FloatTensor(self.next_states[ind]),
            torch.FloatTensor(self.dones[ind]),
            ind,
        )

    def save(self, path):
        np.savez(
            path, ptr=self.ptr, size=self.size,
            states=self.states, actions=self.actions, rewards=self.rewards,
            next_states=self.next_states, dones=self.dones,
            priorities=self.priorities, success=self.success, step_ids=self.step_ids,
            stage_ids=self.stage_ids,
        )

    def load(self, path):
        data = np.load(path)
        self.ptr = int(data["ptr"])
        self.size = int(data["size"])
        self.states = data["states"]
        self.actions = data["actions"]
        self.rewards = data["rewards"]
        self.next_states = data["next_states"]
        self.dones = data["dones"]
        self.priorities = data["priorities"]
        self.success = data["success"]
        if "step_ids" in data:
            self.step_ids = data["step_ids"]
        if "stage_ids" in data:
            self.stage_ids = data["stage_ids"]
