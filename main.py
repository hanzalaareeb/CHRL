import argparse
import contextlib
import os
import sys

import numpy as np
import torch

from config import Config
from env import ContinuousHomeostaticEnv
from agents import make_agent
from training import train, train_curriculum, stage_checkpoint_paths
from visualization import visualize
from diagnostics import run as run_diagnostics


class TeeStream:
    """Mirror stdout/stderr to both terminal and a log file."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def next_run_number(root_dir, run_kind):
    prefix = f"{run_kind}_"
    max_id = 0
    for name in os.listdir(root_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            max_id = max(max_id, int(suffix))
    return max_id + 1


def latest_run_dir(root_dir, run_kind):
    prefix = f"{run_kind}_"
    best_id = None
    best_name = None
    for name in os.listdir(root_dir):
        if not name.startswith(prefix):
            continue
        suffix = name[len(prefix):]
        if suffix.isdigit():
            run_id = int(suffix)
            if best_id is None or run_id > best_id:
                best_id = run_id
                best_name = name
    return os.path.join(root_dir, best_name) if best_name is not None else None


def build_run_config(base_config=Config, do_train=True):
    """Create a numbered artifact bundle for this invocation."""
    os.makedirs("analysis", exist_ok=True)
    run_kind = "training" if do_train else "test"
    run_number = next_run_number("analysis", run_kind)
    run_label = f"{run_kind}_{run_number:03d}"
    run_dir = os.path.join("analysis", run_label)
    plot_dir = os.path.join(run_dir, "plots")
    tensorboard_dir = os.path.join(run_dir, "tensorboard")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    overrides = {
        "RUN_LABEL": run_label,
        "RUN_DIR": run_dir,
        "PLOT_DIR": plot_dir,
        "LOG_DIR": tensorboard_dir,
        "ACTOR_PATH": os.path.join(checkpoint_dir, "homeostatic_actor.pth"),
        "CRITIC_PATH": os.path.join(checkpoint_dir, "homeostatic_critic.pth"),
        "BEST_ACTOR_PATH": os.path.join(checkpoint_dir, "best_actor.pth"),
        "BEST_CRITIC_PATH": os.path.join(checkpoint_dir, "best_critic.pth"),
        "BUFFER_PATH": os.path.join(checkpoint_dir, "replay_buffer.npz"),
        "RESULT_LOG_PATH": os.path.join(run_dir, f"{run_kind}_result.txt"),
        "TENSORBOARD_TXT_PATH": os.path.join(run_dir, "tensorboard_scalars.txt"),
    }

    if not do_train:
        latest_training = latest_run_dir("analysis", "training")
        if latest_training is not None:
            checkpoint_dir = os.path.join(latest_training, "checkpoints")
            overrides["ACTOR_PATH"] = os.path.join(checkpoint_dir, "homeostatic_actor.pth")
            overrides["CRITIC_PATH"] = os.path.join(checkpoint_dir, "homeostatic_critic.pth")
            overrides["BEST_ACTOR_PATH"] = os.path.join(checkpoint_dir, "best_actor.pth")
            overrides["BEST_CRITIC_PATH"] = os.path.join(checkpoint_dir, "best_critic.pth")
            overrides["SOURCE_RUN_DIR"] = latest_training
        else:
            overrides["SOURCE_RUN_DIR"] = None

    return type("RunConfig", (base_config,), overrides)


def export_tensorboard_scalars(log_dir, save_path):
    """Flatten scalar TensorBoard logs into a readable text file."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        with open(save_path, "w", encoding="utf-8") as handle:
            handle.write("TensorBoard scalar export unavailable: tensorboard package not installed.\n")
        return

    accumulator = EventAccumulator(log_dir)
    accumulator.Reload()
    scalar_tags = sorted(accumulator.Tags().get("scalars", []))
    with open(save_path, "w", encoding="utf-8") as handle:
        if not scalar_tags:
            handle.write("No scalar TensorBoard data found.\n")
            return
        for tag in scalar_tags:
            handle.write(f"[{tag}]\n")
            for event in accumulator.Scalars(tag):
                handle.write(f"step={event.step} value={event.value:.6f}\n")
            handle.write("\n")


def build(config=Config, agent_name=None):
    """Construct a seeded env + agent on the best available device."""
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    device = config.get_device()
    env = ContinuousHomeostaticEnv(config)
    agent = make_agent(env, device, config, name=agent_name)
    print(f"Agent: {type(agent).__name__} | Device: {device}")
    return env, agent


def execute_run(config, agent_name=None, curriculum=None,
                do_train=True, do_viz=True, do_diag=True):
    env, agent = build(config, agent_name)
    use_curriculum = config.CURRICULUM if curriculum is None else curriculum

    if do_train:
        if use_curriculum:
            train_curriculum(agent, config)
        else:
            train(agent, env, config)
    else:
        if getattr(config, "SOURCE_RUN_DIR", None):
            print(f"[load] source training run: {config.SOURCE_RUN_DIR}")
        agent.load(config.ACTOR_PATH, config.CRITIC_PATH)

    if do_diag:
        diag_path = os.path.join(config.PLOT_DIR, f"{config.RUN_LABEL}_diagnostics.png")
        run_diagnostics(agent, env, config, save_path=diag_path)
    if do_viz:
        visualize(agent, env, config, prefix=f"{config.RUN_LABEL}_trained")
        if config.CURRICULUM and len(config.STAGES) >= 2:
            stage2 = config.STAGES[1]
            stage2_actor_path, stage2_critic_path = stage_checkpoint_paths(config, stage2["name"])
            if os.path.exists(stage2_actor_path) and os.path.exists(stage2_critic_path):
                stage2_env = ContinuousHomeostaticEnv(
                    config,
                    resources=stage2["resources"],
                    regen_delay=stage2.get("regen_delay", 0),
                    survival_bonus=stage2.get("survival_bonus", config.SURVIVAL_BONUS),
                    resource_jitter=stage2.get("resource_jitter", 0.0),
                )
                agent.load(stage2_actor_path, stage2_critic_path)
                visualize(agent, stage2_env, config, prefix=f"{config.RUN_LABEL}_stage2_best")
            else:
                print(f"[visualize] stage2 checkpoint missing; skipped stage-2 trajectory plot.")

    if do_train:
        export_tensorboard_scalars(config.LOG_DIR, config.TENSORBOARD_TXT_PATH)
        print(f"[tensorboard-export] saved scalar dump: {config.TENSORBOARD_TXT_PATH}")


def main(config=Config, agent_name=None, curriculum=None,
         do_train=True, do_viz=True, do_diag=True):
    run_config = build_run_config(config, do_train=do_train)

    with open(run_config.RESULT_LOG_PATH, "w", encoding="utf-8") as log_handle:
        tee = TeeStream(sys.stdout, log_handle)
        with contextlib.redirect_stdout(tee), contextlib.redirect_stderr(tee):
            print(f"[run] {run_config.RUN_LABEL}")
            print(f"[run] artifacts: {run_config.RUN_DIR}")
            execute_run(
                run_config,
                agent_name=agent_name,
                curriculum=curriculum,
                do_train=do_train,
                do_viz=do_viz,
                do_diag=do_diag,
            )

    print(f"[run] saved log: {run_config.RESULT_LOG_PATH}")
    return run_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CHRL DDPG/TD3 runner")
    parser.add_argument("--agent", choices=["ddpg", "td3"], default=None,
                        help="agent to use (defaults to Config.AGENT)")
    curr = parser.add_mutually_exclusive_group()
    curr.add_argument("--curriculum", dest="curriculum", action="store_true",
                      help="train through the staged curriculum")
    curr.add_argument("--no-curriculum", dest="curriculum", action="store_false",
                      help="single-stage training on the full layout")
    parser.set_defaults(curriculum=None)
    parser.add_argument("--no-train", action="store_true", help="skip training (load saved model)")
    parser.add_argument("--no-viz", action="store_true", help="skip trajectory/state plots")
    parser.add_argument("--no-diag", action="store_true", help="skip critic/reward diagnostics")
    args = parser.parse_args()

    main(
        agent_name=args.agent,
        curriculum=args.curriculum,
        do_train=not args.no_train,
        do_viz=not args.no_viz,
        do_diag=not args.no_diag,
    )
