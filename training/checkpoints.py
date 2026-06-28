import os


def stage_checkpoint_paths(config, stage_name):
    digits = "".join(ch for ch in stage_name if ch.isdigit())
    stage_id = digits or stage_name.lower().replace(" ", "_")
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    return (
        os.path.join(checkpoint_dir, f"stage{stage_id}_best_actor.pth"),
        os.path.join(checkpoint_dir, f"stage{stage_id}_best_critic.pth"),
    )


def stage_periodic_checkpoint_paths(config, stage_name, episode_number):
    digits = "".join(ch for ch in stage_name if ch.isdigit())
    stage_id = digits or stage_name.lower().replace(" ", "_")
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    return (
        os.path.join(checkpoint_dir, f"stage{stage_id}_ep{episode_number:04d}_actor.pth"),
        os.path.join(checkpoint_dir, f"stage{stage_id}_ep{episode_number:04d}_critic.pth"),
    )


def transition_checkpoint_paths(config, label):
    checkpoint_dir = os.path.dirname(config.ACTOR_PATH) or "."
    safe = label.lower().replace(" ", "_").replace("-", "_")
    return (
        os.path.join(checkpoint_dir, f"{safe}_actor.pth"),
        os.path.join(checkpoint_dir, f"{safe}_critic.pth"),
    )
