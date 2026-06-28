from config import Config
from training.metrics import EpisodeMetrics, EvaluationAccumulator


def evaluate(agent, env, use_target=False, n_episodes=1, seed_base=Config.SEED):
    accumulator = EvaluationAccumulator()
    resource_type = env.resource_type

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_base + ep)
        metrics = EpisodeMetrics(
            action_dim=agent.action_dim,
            n_internal=env.n_internal,
            n_resources=env.n_resources,
            max_steps=env.max_steps,
            initial_internal_state=env.internal_state.copy(),
        )
        terminated = False
        done = False
        nearest_resource_distance = float("inf")
        prev_pos = env.agent_pos.copy()

        while not done:
            action = agent.select_action(state, add_noise=False, use_target=use_target)
            state, reward, terminated, truncated, info = env.step(action)
            prev_pos = metrics.update_from_step(reward, info, prev_pos, resource_type)
            nearest_resource_distance = min(nearest_resource_distance, metrics.minimum_distance_reached)
            done = terminated or truncated

        accumulator.add_episode(metrics, terminated, nearest_resource_distance)

    return accumulator.summary()
