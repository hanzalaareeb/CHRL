# Training Log

## Summary Table

| Run | Command | Checkpoint | Avg survival | Avg consumption | Avg final drive | Notes |
|---|---|---:|---:|---:|---:|---|
| `training_001` | `uv run python evaluate_all_checkpoints.py` | latest | 136.6 | 0.0 | 1.72 | Never entered a resource radius in the comparison eval |
| `training_001` | `uv run python evaluate_all_checkpoints.py` | best | 150.9 | 14.1 | 1.02 | Strongly better than final checkpoint |
| `training_001` | `uv run python evaluate_all_checkpoints.py` | stage3_best | 150.9 | 14.1 | 1.02 | Same as `best` for this run |

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
