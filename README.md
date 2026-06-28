# CHRL

Continuous homeostatic reinforcement learning in a 2D world with food and water regulation.

## Run

```bash
uv run python main.py
uv run python main.py --no-curriculum
uv run python main.py --agent ddpg
uv run python main.py --no-train
uv run python evaluate_all_checkpoints.py
```

## Files

- `config.py`: experiment and curriculum settings
- `env.py`: environment
- `agents/`: DDPG and TD3 agents
- `train.py`: training loop and curriculum logic
- `diagnostics.py`: reward / critic diagnostics
- `visualize.py`: trajectory and resource plots
- `evaluate_all_checkpoints.py`: fixed-seed checkpoint comparison
- `analysis/`: numbered training and test outputs

## Outputs

Each run auto-saves into `analysis/training_XXX/` or `analysis/test_XXX/`:

- `training_result.txt` or `test_result.txt`
- `tensorboard_scalars.txt`
- `plots/`
- `checkpoints/`

## Logs

- Training comparisons: [TRAINING_LOG.md](/Users/khanhanzalaareeb/Documents/project/CHRL/TRAINING_LOG.md)
