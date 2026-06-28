import argparse
import os
import re

from agents import make_agent
from config import Config
from env import ContinuousHomeostaticEnv
from evaluation import available_checkpoints, checkpoint_path_map, evaluate_checkpoint
from main import latest_run_dir


def resolve_run_dir(explicit_run_dir=None):
    if explicit_run_dir is not None:
        return explicit_run_dir
    latest_training = latest_run_dir("analysis", "training")
    if latest_training is None:
        raise FileNotFoundError("No training runs found under analysis/.")
    return latest_training


def infer_stage_aliases(run_dir, available):
    """Backfill stage-best aliases for older runs that only saved global best."""
    training_log = os.path.join(run_dir, "training_result.txt")
    if not os.path.exists(training_log) or "best" not in available:
        return available

    text = open(training_log, "r", encoding="utf-8").read()
    match = re.search(r"\[best checkpoint\].*during stage '([^']+)'", text)
    if not match:
        return available

    stage_name = match.group(1)
    if stage_name.startswith("1-") and "stage1_best" not in available:
        available["stage1_best"] = available["best"]
    if stage_name.startswith("2-") and "stage2_best" not in available:
        available["stage2_best"] = available["best"]
    if stage_name.startswith("3-") and "stage3_best" not in available:
        available["stage3_best"] = available["best"]
    return available


def print_summary_table(results):
    print("\nCheckpoint\tAvg survival\tAvg consumption\tAvg final drive")
    for name, metrics in results:
        print(
            f"{name}\t{metrics['avg_survival']:.1f}\t"
            f"{metrics['avg_consumption']:.1f}\t{metrics['avg_final_drive']:.2f}"
        )


def print_diagnostic_block(name, metrics):
    print(
        f"\n[{name}] "
        f"nearest={metrics['nearest_resource_distance']:.2f} | "
        f"min_reached={metrics['minimum_distance_reached']:.2f} | "
        f"entered={metrics['resource_entered']:.1%} | "
        f"consumed={metrics['consumed_any']:.1%} | "
        f"time_to_first_consumption={metrics['time_to_first_consumption']:.1f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate latest/best/stage-best checkpoints on fixed seeds.")
    parser.add_argument("--run-dir", default=None, help="training run dir to inspect, defaults to latest analysis/training_*")
    parser.add_argument("--agent", choices=["ddpg", "td3"], default=None, help="agent to use (defaults to Config.AGENT)")
    parser.add_argument("--episodes", type=int, default=20, help="number of fixed-seed evaluation episodes")
    args = parser.parse_args()

    run_dir = resolve_run_dir(args.run_dir)
    print(f"[evaluate] run_dir={run_dir}")

    env = ContinuousHomeostaticEnv(Config)
    agent = make_agent(env, Config.get_device(), Config, name=args.agent)
    seeds = [Config.SEED + i for i in range(args.episodes)]

    requested = checkpoint_path_map(run_dir)
    available = infer_stage_aliases(run_dir, available_checkpoints(run_dir))
    missing = [name for name in requested if name not in available]
    if missing:
        print(f"[evaluate] missing checkpoints in this run: {', '.join(missing)}")

    results = []
    for name in ["latest", "best", "stage1_best", "stage2_best", "stage3_best"]:
        if name not in available:
            continue
        actor_path, critic_path = available[name]
        metrics = evaluate_checkpoint(agent, actor_path, critic_path, seeds, config=Config)
        results.append((name, metrics))

    if not results:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")

    print_summary_table(results)
    for name, metrics in results:
        print_diagnostic_block(name, metrics)


if __name__ == "__main__":
    main()
