from config import Config
from .ddpg import DDPGAgent
from .td3 import TD3Agent

AGENTS = {"ddpg": DDPGAgent, "td3": TD3Agent}


def make_agent(env, device, config=Config, name=None):
    """Build the agent selected by ``name`` (or ``config.AGENT``) for ``env``."""
    name = (name or config.AGENT).lower()
    if name not in AGENTS:
        raise ValueError(f"Unknown agent '{name}'. Choose from {list(AGENTS)}.")
    return AGENTS[name](
        env.observation_space.shape[0],
        env.action_space.shape[0],
        device,
        config=config,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )


__all__ = ["DDPGAgent", "TD3Agent", "make_agent", "AGENTS"]
