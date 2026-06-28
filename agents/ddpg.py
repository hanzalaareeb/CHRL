import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import Config
from network import Actor, Critic
from replay_buffer import ReplayBuffer
from noise import build_noise


# ==========================================
# DDPG Agent
# ==========================================
class DDPGAgent:
    """Deep Deterministic Policy Gradient agent for continuous control.

    Encapsulates the actor/critic networks, their target copies, optimizers,
    the replay buffer and the exploration noise process. The episode loop that
    drives this agent lives in ``train.py``.
    """

    def __init__(self, state_dim, action_dim, device, config=Config,
                 action_low=-1.0, action_high=1.0):
        self.device = device
        self.config = config
        self.action_dim = action_dim
        self.action_low = action_low
        self.action_high = action_high

        self.gamma = config.GAMMA
        self.tau = config.TAU
        self.batch_size = config.BATCH_SIZE

        # --- Actor and its target ---
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.ACTOR_LR)

        # --- Critic and its target ---
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.CRITIC_LR)

        # --- Replay buffer and exploration noise ---
        self.replay_buffer = ReplayBuffer(state_dim, action_dim, config.BUFFER_SIZE)
        self.noise = build_noise(action_dim, config)

    def select_action(self, state, add_noise=True, use_target=False):
        """Return a (clipped) action for a single environment state.

        No actor.eval()/train() toggling: the actor has no dropout/batchnorm,
        so torch.no_grad() is all that's needed. ``use_target`` evaluates the
        slow-moving target actor instead (for diagnostics).
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
        """Run one DDPG optimization step. Returns loss dict, or None if warming up."""
        if self.replay_buffer.size < self.batch_size:
            return None

        b_state, b_action, b_reward, b_next_state, b_done, batch_indices = \
            self.replay_buffer.sample(self.batch_size)

        b_state = b_state.to(self.device)
        b_action = b_action.to(self.device)
        b_reward = b_reward.to(self.device)
        b_next_state = b_next_state.to(self.device)
        b_done = b_done.to(self.device)

        # --- Update Critic ---
        with torch.no_grad():
            next_action = self.actor_target(b_next_state)
            target_Q = self.critic_target(b_next_state, next_action)
            target_Q = b_reward + (self.gamma * target_Q * (1 - b_done))

        current_Q = self.critic(b_state, b_action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)
        td_error = (current_Q - target_Q).detach().abs().squeeze(-1).cpu().numpy()
        self.replay_buffer.update_priorities(batch_indices, td_error + self.config.PER_EPSILON)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --- Update Actor ---
        actor_loss = -self.critic(b_state, self.actor(b_state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- Soft Update Target Networks ---
        self._soft_update(self.critic, self.critic_target)
        self._soft_update(self.actor, self.actor_target)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "q_value": current_Q.mean().item(),
        }

    def _soft_update(self, source, target):
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    @torch.no_grad()
    def evaluate_q(self, states, actions):
        """Critic Q-estimates for batches of (state, action). Used by diagnostics."""
        states = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
        if states.ndim == 1:
            states = states.unsqueeze(0)
            actions = actions.unsqueeze(0)
        return self.critic(states, actions).cpu().numpy().flatten()

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
