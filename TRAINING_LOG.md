# Training Log

## Summary Table

| Run            | Command                                     |  Checkpoint | Avg survival | Avg consumption | Avg final drive | Notes                                                     |
| -------------- | ------------------------------------------- | ----------: | -----------: | --------------: | --------------: | --------------------------------------------------------- |
| `training_001` | `uv run python evaluate_all_checkpoints.py` |      latest |        136.6 |             0.0 |            1.72 | Never entered a resource radius in the comparison eval    |
| `training_001` | `uv run python evaluate_all_checkpoints.py` |        best |        150.9 |            14.1 |            1.02 | Strongly better than final checkpoint                     |
| `training_001` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        150.9 |            14.1 |            1.02 | Same as `best` for this run                               |
| `training_002` | `uv run python evaluate_all_checkpoints.py` |      latest |        136.6 |             0.0 |            1.72 | Final checkpoint regressed, no resource entry             |
| `training_002` | `uv run python evaluate_all_checkpoints.py` |        best |        150.9 |            23.0 |            1.10 | Best overall checkpoint for this run                      |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        147.4 |            15.8 |            1.15 | Usable but weaker and much slower first consumption       |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        150.9 |            23.0 |            1.10 | Matches global best; stage 2 peak                         |
| `training_002` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        136.6 |             0.0 |            1.72 | Stage 3 transition/regression failure                     |
| `training_004` | `uv run python evaluate_all_checkpoints.py` |      latest |        136.6 |             2.6 |            1.62 | User provided block says `run_dir=analysis/training_004`  |
| `training_004` | `uv run python evaluate_all_checkpoints.py` |        best |        150.9 |            22.2 |            1.10 | Strong checkpoint; same as `stage1_best` in provided eval |
| `training_004` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        150.9 |            22.2 |            1.10 | Matches global best in provided eval                      |
| `training_004` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        150.9 |            24.9 |            1.24 | Best consumption in provided eval                         |
| `training_004` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        137.2 |             1.2 |            1.62 | Partial stage-3 competence, but still weak                |
| `training_007` | `uv run python evaluate_all_checkpoints.py` |      latest |        145.3 |            15.6 |            1.13 | Final checkpoint stays usable on full env                 |
| `training_007` | `uv run python evaluate_all_checkpoints.py` |        best |        252.6 |            35.5 |            1.20 | Strongest run so far; global best found during stage 4    |
| `training_007` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        138.6 |             0.6 |            1.71 | Stage 1 does not transfer well to full env                |
| `training_007` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        153.8 |            14.3 |            1.30 | Useful intermediate competence                            |
| `training_007` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        150.9 |             6.2 |            1.37 | Stage 3 no longer fully collapses, but is still modest    |

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

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             1.91 |                     1.91 |             0.0% |        0.0% |                    1000.0 |
| best        |             0.89 |                     0.89 |           100.0% |      100.0% |                      27.0 |
| stage3_best |             0.89 |                     0.89 |           100.0% |      100.0% |                      27.0 |

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

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             1.79 |                     1.79 |             0.0% |        0.0% |                    1000.0 |
| best        |             0.67 |                     0.67 |           100.0% |      100.0% |                      74.5 |
| stage1_best |             0.92 |                     0.92 |            85.0% |       85.0% |                     243.3 |
| stage2_best |             0.67 |                     0.67 |           100.0% |      100.0% |                      74.5 |
| stage3_best |             2.46 |                     2.46 |             0.0% |        0.0% |                    1000.0 |

Verdict:

- `best` and `stage2_best` are the strongest checkpoints in this run.
- `stage1_best` is viable but clearly slower to achieve consumption.
- `stage3_best` collapses to the same failure mode as `latest`.
- This supports the conclusion that the stage-3 transition or subsequent updates caused regression.

### `training_004`

Source run: [analysis/training_004](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_004/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_004

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          136.6           2.6             1.62
best            150.9           22.2            1.10
stage1_best     150.9           22.2            1.10
stage2_best     150.9           24.9            1.24
stage3_best     137.2           1.2             1.62
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             1.55 |                     1.55 |            10.0% |       10.0% |                     904.8 |
| best        |             0.34 |                     0.34 |           100.0% |      100.0% |                      81.8 |
| stage1_best |             0.34 |                     0.34 |           100.0% |      100.0% |                      81.8 |
| stage2_best |             0.60 |                     0.60 |           100.0% |      100.0% |                      59.2 |
| stage3_best |             1.09 |                     1.09 |            45.0% |       45.0% |                     597.6 |

Verdict:

- `stage2_best` is the strongest checkpoint in the provided evaluation.
- `best` and `stage1_best` are also strong and fully consumption-capable.
- `latest` retains only weak partial competence.
- `stage3_best` is better than a total collapse, but still far behind the earlier checkpoints.

### `training_007`

Source run: [analysis/training_007](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_007/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_007

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          145.3           15.6            1.13
best            252.6           35.5            1.20
stage1_best     138.6           0.6             1.71
stage2_best     153.8           14.3            1.30
stage3_best     150.9           6.2             1.37
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             0.44 |                     0.44 |            85.0% |       85.0% |                     176.2 |
| best        |             0.35 |                     0.35 |           100.0% |      100.0% |                      30.7 |
| stage1_best |             1.09 |                     1.09 |            10.0% |       10.0% |                     912.0 |
| stage2_best |             0.92 |                     0.92 |            70.0% |       70.0% |                     338.8 |
| stage3_best |             1.03 |                     1.03 |            80.0% |       80.0% |                     272.0 |

Verdict:

- `training_007` is the strongest run so far.
- `best` is dramatically stronger than `latest`, so checkpoint selection still matters a lot.
- `latest` is no longer a total failure on the full environment, which is a meaningful improvement over earlier runs.
- `stage3_best` shows partial transfer instead of catastrophic collapse, but the major breakthrough appears later in stage 4.
