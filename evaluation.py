import os
from dataclasses import dataclass

import numpy as np

from config import Config
from env import ContinuousHomeostaticEnv


@dataclass
class EpisodeDiagnostics:
    steps: int
    survived: float
    consumption_events: int
    final_drive: float
    nearest_resource_distance: float
    minimum_distance_reached: float
    resource_entered: float
    consumed_any: float
    time_to_first_consumption: float


def _rollout_checkpoint_episode(agent, env, seed, use_target=False):
    state, _ = env.reset(seed=seed)
    terminated = False
    done = False
    steps = 0
    consumption_events = 0
    first_consumption_step = env.max_steps
    nearest_resource_distance = float("inf")
    minimum_distance_reached = float("inf")
    resource_entered = False
    info = {}

    while not done:
        action = agent.select_action(state, add_noise=False, use_target=use_target)
        state, _, terminated, truncated, info = env.step(action)
        steps += 1

        dists = np.asarray(info["dists"], dtype=np.float64)
        within = np.asarray(info["within_radius"], dtype=bool)
        delivered = np.asarray(info["delivered"], dtype=np.float64)

        nearest_resource_distance = min(nearest_resource_distance, float(dists.min()))
        minimum_distance_reached = min(minimum_distance_reached, float(dists.min()))
        resource_entered = resource_entered or bool(np.any(within))

        if np.any(delivered > 0.0):
            consumption_events += 1
            if first_consumption_step == env.max_steps:
                first_consumption_step = steps

        done = terminated or truncated

    return EpisodeDiagnostics(
        steps=steps,
        survived=float(not terminated),
        consumption_events=consumption_events,
        final_drive=float(info.get("drive", float("nan"))),
        nearest_resource_distance=nearest_resource_distance,
        minimum_distance_reached=minimum_distance_reached,
        resource_entered=float(resource_entered),
        consumed_any=float(consumption_events > 0),
        time_to_first_consumption=float(first_consumption_step),
    )


def evaluate_checkpoint(agent, actor_path, critic_path, seeds, config=Config, use_target=False):
    """Evaluate one checkpoint on the full environment over fixed seeds."""
    env = ContinuousHomeostaticEnv(
        config, resources=config.RESOURCES, regen_delay=config.REGEN_DELAY, survival_bonus=0.0
    )
    agent.load(actor_path, critic_path)

    episodes = [_rollout_checkpoint_episode(agent, env, seed, use_target=use_target) for seed in seeds]
    return {
        "checkpoint": actor_path,
        "avg_survival": float(np.mean([ep.steps for ep in episodes])),
        "avg_consumption": float(np.mean([ep.consumption_events for ep in episodes])),
        "avg_final_drive": float(np.mean([ep.final_drive for ep in episodes])),
        "nearest_resource_distance": float(np.mean([ep.nearest_resource_distance for ep in episodes])),
        "minimum_distance_reached": float(np.mean([ep.minimum_distance_reached for ep in episodes])),
        "resource_entered": float(np.mean([ep.resource_entered for ep in episodes])),
        "consumed_any": float(np.mean([ep.consumed_any for ep in episodes])),
        "time_to_first_consumption": float(np.mean([ep.time_to_first_consumption for ep in episodes])),
        "episodes": episodes,
    }


def checkpoint_path_map(run_dir):
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    return {
        "latest": (
            os.path.join(checkpoint_dir, "homeostatic_actor.pth"),
            os.path.join(checkpoint_dir, "homeostatic_critic.pth"),
        ),
        "best": (
            os.path.join(checkpoint_dir, "best_actor.pth"),
            os.path.join(checkpoint_dir, "best_critic.pth"),
        ),
        "stage1_best": (
            os.path.join(checkpoint_dir, "stage1_best_actor.pth"),
            os.path.join(checkpoint_dir, "stage1_best_critic.pth"),
        ),
        "stage2_best": (
            os.path.join(checkpoint_dir, "stage2_best_actor.pth"),
            os.path.join(checkpoint_dir, "stage2_best_critic.pth"),
        ),
        "stage3_best": (
            os.path.join(checkpoint_dir, "stage3_best_actor.pth"),
            os.path.join(checkpoint_dir, "stage3_best_critic.pth"),
        ),
    }


def available_checkpoints(run_dir):
    available = {}
    for name, (actor_path, critic_path) in checkpoint_path_map(run_dir).items():
        if os.path.exists(actor_path) and os.path.exists(critic_path):
            available[name] = (actor_path, critic_path)
    return available
