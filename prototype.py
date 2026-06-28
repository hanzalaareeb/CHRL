import numpy as np
import gym
from gym import spaces
import torch
import torch.nn as nn

# ======================================
# 1. Continuse Time/space Homestatic Env
# ======================================

class ContinuousHomeostaticEnv(gym.Env):
    """CTCS-HRRL Environment based on laurencon et al. (link)
    Model internal state dynamics (dH/dt) and drive reduction as reward

    Args:
        gym (_type_): _description_
    """
    
    def __init__(self, dt=0.1, n_resources=2):
        super(ContinuousHomeostaticEnv, self).__init__()
        self.dt = dt
        self.n_resources = n_resources
        
        # Optimal internal state (H*) and biological decay rates
        self.H_star = np.ones(n_resources) * 1.0
        self.decay_rates = np.ones(n_resources) * 0.05
        
        # Observation: [Agent X, Y] + [Internal states] + [Distances to resources]
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(2 + n_resources + n_resources,), dtype=np.float32
        )
        
        # Action: [Velocity X, Velocity Y] - Continuous spatial movement
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(2,), dtype=np.float32
        )
        
        # Fixed locations of resources (e.g., Food and Water)
        self.resource_locations = np.array([[3.0, 3.0], [-3.0, -3.0]])
        self.reset()
        
    def reset(self):
        self.agent_pos = np.zeros(2)
        # Agent starts at perfect homestasis
        self.internal_state = np.copy(self.H_star)
        self.time_step = 0
        return self._get_obs()
    
    def _get_obs(self):
        # Calculate distances to all biological resources
        dists = np.linalg.norm(self.resource_locations - self.agent_pos, axis=1)
        return np.concatenate([self.agent_pos, self.internal_state, dists]).astype(np.float32)
    
    def drive(self, H):
        # Drive function D(H) = \\H - H*\\^2 (Eculidean distance to optimal state)
        return np.sum((H - self.H_star)**2)
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        
        # 1. Record current physiolo9gical drive
        D_current = self.drive(self.internal_state)
        
        # 2. Continuous Spatial Kinematics
        self.agent_pos += action * self.dt
        
        # 3. Continuous Internal State Dynamic (dH/dt)
        dists = np.linalg.norm(self.resource_locations - self.agent_pos, axis=1)
        
        # Consumption triggerts automatically if agent is within resource radius
        consumption_rates = np.where(dists < 1.0, 0.5, 0.0)
        
        # dH/dt = -decay + consumption
        dH_dt = -self.decay_rates + consumption_rates
        self.internal_state += dH_dt * self.dt
        
        # Biological bounds
        self.internal_state = np.clip(self.internal_state, 0.0, 2.0)
        
        # 4. fomulation: Drive Reduction over continuous time
        D_next = self.drive(self.internal_state)
        
        # r(t) = d/dt D(H(t))
        drive_reduction = (D_current - D_next) / self.dt
        
        # Optional: Add HJB effort penalty for taking actions
        effort_penalty = 0.01 * np.sum(action**2) # 0.01 * np.linalg.norm(action)
        reward = drive_reduction - effort_penalty
        
        self.time_step += 1
        
        # Agent "dies" if any essential internal stte is depletes to 0
        done = self.time_step >= 1000 or np.any(self.internal_state <= 0.0)
        
        info = {'drive': D_next, 'internal_state': self.internal_state}
        return self._get_obs(), float(reward), done, info


# ============================================
# 2. Continuous Actor-Critic Network (PyTorhc)
# ============================================

class Actor(nn.Module):
    """Maps (External State + Internal state) to Continuous Spatial Velocity

    Args:
        nn (_type_): _description_
    """
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # Bounds spatial movement tp [-1, 1]
        )
            
    def forward(self, state):
        return self.net(state)
    
    
class Critic(nn.Module):
    """Evaluates the Q-value f the homestatic trajactory

    Args:
        nn (_type_): _description_
    """
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
            
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim = 1))