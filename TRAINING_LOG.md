# Training Log

## Summary Table

| Run | Command | Checkpoint | Avg survival | Avg consumption | Avg final drive | Notes |
|---|---|---:|---:|---:|---:|---|
| `training_001` | `uv run python evaluate_all_checkpoints.py` | latest | 136.6 | 0.0 | 1.72 | Never entered a resource radius in the comparison eval |
| `training_001` | `uv run python evaluate_all_checkpoints.py` | best | 150.9 | 14.1 | 1.02 | Strongly better than final checkpoint |
| `training_001` | `uv run python evaluate_all_checkpoints.py` | stage3_best | 150.9 | 14.1 | 1.02 | Same as `best` for this run |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | latest | 136.6 | 0.0 | 1.72 | Final checkpoint regressed, no resource entry |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | best | 150.9 | 23.0 | 1.10 | Best overall checkpoint for this run |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage1_best | 147.4 | 15.8 | 1.15 | Usable but weaker and much slower first consumption |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage2_best | 150.9 | 23.0 | 1.10 | Matches global best; stage 2 peak |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage3_best | 136.6 | 0.0 | 1.72 | Stage 3 transition/regression failure |

## Detailed Comparison

### `training_001`

Source run: [analysis/training_001](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_001/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_001
[evaluate] missing checkpoints in this run: stage1_best, stage2_best

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          136.6           0.0             1.72
best            150.9           14.1            1.02
stage3_best     150.9           14.1            1.02
```

Resource-interaction diagnostics:

| Checkpoint | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
|---|---:|---:|---:|---:|---:|
| latest | 1.91 | 1.91 | 0.0% | 0.0% | 1000.0 |
| best | 0.89 | 0.89 | 100.0% | 100.0% | 27.0 |
| stage3_best | 0.89 | 0.89 | 100.0% | 100.0% | 27.0 |

Verdict:

- `best` and `stage3_best` are clearly better than `latest`.
- The final checkpoint appears to overshoot after stage 3.
- `training_001` predates per-stage best saves, so `stage1_best` and `stage2_best` were not available.

### `training_002`

Source run: [analysis/training_002](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_002/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_002

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          136.6           0.0             1.72
best            150.9           23.0            1.10
stage1_best     147.4           15.8            1.15
stage2_best     150.9           23.0            1.10
stage3_best     136.6           0.0             1.72
```

Resource-interaction diagnostics:

| Checkpoint | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
|---|---:|---:|---:|---:|---:|
| latest | 1.79 | 1.79 | 0.0% | 0.0% | 1000.0 |
| best | 0.67 | 0.67 | 100.0% | 100.0% | 74.5 |
| stage1_best | 0.92 | 0.92 | 85.0% | 85.0% | 243.3 |
| stage2_best | 0.67 | 0.67 | 100.0% | 100.0% | 74.5 |
| stage3_best | 2.46 | 2.46 | 0.0% | 0.0% | 1000.0 |

Verdict:

- `best` and `stage2_best` are the strongest checkpoints in this run.
- `stage1_best` is viable but clearly slower to achieve consumption.
- `stage3_best` collapses to the same failure mode as `latest`.
- This supports the conclusion that the stage-3 transition or subsequent updates caused regression.

## Next Training Changes

- Reset `consumption_plateau_evals` at each new stage.
- Reset or stage-scope the best-consumption tracker for early-stop decisions.
- Compare stage-3 progress only against stage-3 history, not against stage-2 peak forever.
- Keep global best checkpoint saving as-is.
- Keep `stage2_best` and `stage3_best` saving as-is.

## Next Training Checks

- Evaluate before stage 3 training starts.
- Save and log stage-2 best with survival, consumption, and nearest-resource distance before stage 3 begins.
- During stage 3, save checkpoints every `n` episodes, not only `stage3_best`.

## Logging Requests

- After every 20 episodes, log replay size, successful transition percentage, episodes with consumption, episodes without consumption, and average reward.
- Log actor consumption and target consumption separately.
- Track current evaluation vs best evaluation; if current is less than `best * 0.5`, print a performance-collapse warning.
- Print exploration noise once per episode: episode, stage, sigma, and noise floor.
- Log replay age and average transition age.
