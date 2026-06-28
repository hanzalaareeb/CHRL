import torch
import torch.nn as nn

from config import Config

# ============================================
# 2. Continuous Actor-Critic Network (PyTorch)
# ============================================

class Actor(nn.Module):
    """Maps (External State + Internal State) to Continuous Spatial Velocity."""

    def __init__(self, state_dim, action_dim, hidden_dim=Config.HIDDEN_DIM):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # Bounds spatial movement to [-1, 1]
        )

    def forward(self, state):
        return self.net(state)


class Critic(nn.Module):
    """Evaluates the Q-value of the homeostatic trajectory (single critic, DDPG)."""

    def __init__(self, state_dim, action_dim, hidden_dim=Config.HIDDEN_DIM):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=1))


def _q_network(state_dim, action_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(state_dim + action_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )


class TwinCritic(nn.Module):
    """Twin Q-networks for TD3 (clipped double-Q learning).

    ``forward`` returns both Q-estimates; ``Q1`` returns only the first head,
    which is what the actor is optimized against.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=Config.HIDDEN_DIM):
        super(TwinCritic, self).__init__()
        self.q1 = _q_network(state_dim, action_dim, hidden_dim)
        self.q2 = _q_network(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        return self.q1(torch.cat([state, action], dim=1))