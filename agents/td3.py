import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import Actor, TwinCritic
from replay_buffer import ReplayBuffer
from noise import build_noise


# ==========================================
# TD3 Agent  (Twin Delayed DDPG)
# ==========================================
class TD3Agent:
    """TD3 = DDPG + three tricks (Fujimoto et al., 2018):

    1. Clipped double-Q: twin critics, take the min in the target to fight
       overestimation.
    2. Target policy smoothing: add clipped noise to the target action.
    3. Delayed policy updates: update the actor (and targets) every
       ``POLICY_DELAY`` critic steps.

    Drop-in compatible with ``DDPGAgent`` (same constructor + ``select_action``,
    ``update``, ``replay_buffer``, ``noise``, ``save``, ``load``, ``evaluate_q``),
    so the shared ``train.py`` loop drives either agent.
    """

    def __init__(self, state_dim, action_dim, device, config=Config,
                 action_low=-1.0, action_high=1.0):
        self.device = device
        self.config = config
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high
        # Symmetric action bound used for clamping the smoothed target action.
        self.max_action = float(np.max(np.abs([action_low, action_high])))

        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE
        self.policy_noise = config.POLICY_NOISE
        self.noise_clip = config.NOISE_CLIP
        self.policy_delay = config.POLICY_DELAY
        self.total_it = 0

        # --- Actor and its target ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        # --- Twin critics and their target ---
        self.critic = TwinCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # --- Replay buffer and exploration noise ---
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config.BUFFER_SIZE)
        self.noise = build_noise(action_dim, config)

    def select_action(self, state, add_noise=True, use_target=False):
        """Deterministic action + (optional) Gaussian exploration noise.

        ``use_target`` evaluates the slow-moving target actor (for diagnostics).
        """
        net = self.actor_target if use_target else self.actor
        state_t = torch.as_tensor(np.asarray(state), dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = net(state_t).cpu().numpy().flatten()

        if add_noise:
            action = action + self.noise.sample()

        return np.clip(action, self.action_low, self.action_high)

    def update(self):
        """Run one TD3 optimization step. Returns loss dict, or None if warming up."""
        if self.replay_buffer.size < self.batch_size:
            return None

        self.total_it += 1

        b_state, b_action, b_reward, b_next_state, b_done, batch_indices = \
            self.replay_buffer.sample(self.batch_size)

        b_state = b_state.to(self.device)
        b_action = b_action.to(self.device)
        b_reward = b_reward.to(self.device)
        b_next_state = b_next_state.to(self.device)
        b_done = b_done.to(self.device)

        # --- Critic update with target policy smoothing + clipped double-Q ---
        with torch.no_grad():
            # 1) noise -> 2) clip noise -> 3) add to target action -> 4) clip action
            noise = (torch.randn_like(b_action) * self.policy_noise
                     ).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(b_next_state) + noise
                           ).clamp(-self.max_action, self.max_action)

            target_Q1, target_Q2 = self.critic_target(b_next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = b_reward + (self.gamma * target_Q * (1 - b_done))

        current_Q1, current_Q2 = self.critic(b_state, b_action)
        critic_loss = nn.MSELoss()(current_Q1, target_Q) + nn.MSELoss()(current_Q2, target_Q)
        td_error = 0.5 * (
            (current_Q1 - target_Q).detach().abs() +
            (current_Q2 - target_Q).detach().abs()
        )
        self.replay_buffer.update_priorities(
            batch_indices, td_error.squeeze(-1).cpu().numpy() + self.config.PER_EPSILON
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        losses = {
            "critic_loss": critic_loss.item(),
            "q1": current_Q1.mean().item(),
            "q2": current_Q2.mean().item(),
            # Twin-critic disagreement: a large/growing gap signals critic divergence.
            "q_diff": (current_Q1 - current_Q2).abs().mean().item(),
        }

        # --- Delayed actor & target updates ---
        if self.total_it % self.policy_delay == 0:
            # Actor maximizes Q1 only (TD3).
            actor_loss = -self.critic.Q1(b_state, self.actor(b_state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.critic, self.critic_target)
            self._soft_update(self.actor, self.actor_target)

            losses["actor_loss"] = actor_loss.item()

        return losses

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @torch.no_grad()
    def evaluate_q(self, states, actions):
        """TD3 value estimate min(Q1, Q2) for batches of (state, action)."""
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
        q1, q2 = self.critic(states, actions)
        return torch.min(q1, q2).cpu().numpy().flatten()

    def actor_state_dict(self, use_target=False):
        net = self.actor_target if use_target else self.actor
        return net.state_dict()

    def save(self, actor_path=None, critic_path=None):
        actor_path = actor_path or self.config.ACTOR_PATH
        critic_path = critic_path or self.config.CRITIC_PATH
        torch.save(self.actor_state_dict(use_target=False), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def load(self, actor_path=None, critic_path=None):
        actor_path = actor_path or self.config.ACTOR_PATH
        critic_path = critic_path or self.config.CRITIC_PATH
        self.actor.load_state_dict(torch.load(actor_path, map_location=self.device))
        self.critic.load_state_dict(torch.load(critic_path, map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
