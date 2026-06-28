import numpy as np

from config import Config


# ==========================================
# Exploration Noise Processes
# ==========================================
# DDPG is a deterministic policy, so exploration must be injected by adding
# noise to the actor's output. Two standard choices are provided:
#   - GaussianNoise: simple, uncorrelated noise (TD3 / modern DDPG default).
#   - OrnsteinUhlenbeckNoise: temporally correlated noise (original DDPG paper),
#     useful for environments with momentum/inertia like our spatial agent.


class GaussianNoise:
    """Uncorrelated Gaussian exploration noise."""

    def __init__(self, action_dim, sigma=Config.EXPLORATION_NOISE):
        self.action_dim = action_dim
        self.sigma = sigma

    def sample(self):
        return np.random.normal(0.0, self.sigma, size=self.action_dim)

    def reset(self):
        # Stateless; nothing to reset between episodes.
        pass


class OrnsteinUhlenbeckNoise:
    """Temporally correlated noise from an Ornstein-Uhlenbeck process."""

    def __init__(self, action_dim, mu=Config.OU_MU, theta=Config.OU_THETA,
                 sigma=Config.OU_SIGMA, dt=Config.DT):
        self.action_dim = action_dim
        self.mu = mu * np.ones(action_dim)
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = (
            self.theta * (self.mu - self.state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.action_dim)
        )
        self.state = self.state + dx
        return self.state


def build_noise(action_dim, config=Config):
    """Factory that returns the exploration process selected in the config."""
    noise_type = config.NOISE_TYPE.lower()
    if noise_type == "gaussian":
        return GaussianNoise(action_dim, sigma=config.EXPLORATION_NOISE)
    if noise_type == "ou":
        return OrnsteinUhlenbeckNoise(
            action_dim, mu=config.OU_MU, theta=config.OU_THETA,
            sigma=config.OU_SIGMA, dt=config.DT,
        )
    raise ValueError(f"Unknown NOISE_TYPE '{config.NOISE_TYPE}'. Use 'gaussian' or 'ou'.")
