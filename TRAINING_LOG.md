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
| `training_009` | `uv run python evaluate_all_checkpoints.py` |      latest |        155.2 |            19.2 |            1.17 | Final checkpoint is usable and fast to first consumption  |
| `training_009` | `uv run python evaluate_all_checkpoints.py` |        best |        150.9 |            21.0 |            1.06 | Best checkpoint found in stage 2; strong single-resource policy |
| `training_009` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        150.9 |            18.5 |            1.04 | Strong transfer, but still specialized                    |
| `training_009` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        150.9 |            21.0 |            1.06 | Matches global best; stage 2 peak                         |
| `training_009` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        150.9 |            20.9 |            1.10 | Stage 3 remains strong, no catastrophic collapse          |
| `training_009` | `uv run python evaluate_all_checkpoints.py` | stage4_best |        150.9 |            18.1 |            1.06 | Stage 4 usable, but not better than stage 2               |
| `training_011` | `uv run python evaluate_all_checkpoints.py` |      latest |        136.6 |             0.0 |            1.72 | Final checkpoint regressed back to no-resource failure    |
| `training_011` | `uv run python evaluate_all_checkpoints.py` |        best |        155.2 |            22.7 |            1.10 | Best checkpoint came from stage 3; usable but specialized |
| `training_011` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        147.4 |            18.2 |            1.08 | Early stage remains viable                                |
| `training_011` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        144.6 |            11.8 |            1.27 | Weaker than best; some usable competence                  |
| `training_011` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        155.2 |            22.7 |            1.10 | Matches global best; stage 3 peak                         |
| `training_011` | `uv run python evaluate_all_checkpoints.py` | stage4_best |        136.6 |             0.0 |            1.72 | Stage 4 collapsed completely                              |
| `training_015` | `uv run python evaluate_all_checkpoints.py` |      latest |        138.4 |            13.9 |            1.36 | Final checkpoint remains usable but weak on dual-resource behavior |
| `training_015` | `uv run python evaluate_all_checkpoints.py` |        best |        335.2 |            37.9 |            1.50 | Global best came from stage 1; strongest dual-resource checkpoint in this run |
| `training_015` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        335.2 |            37.9 |            1.50 | Matches global best; stage 1 unexpectedly produced the strongest policy |
| `training_015` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        136.6 |             1.7 |            1.64 | Stage 2 collapsed badly and failed to preserve stage-1 competence |
| `training_015` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        203.6 |            24.4 |            1.12 | Stage 3 partially recovered competence after stage-2 failure |
| `training_015` | `uv run python evaluate_all_checkpoints.py` | stage4_best |        231.2 |            29.0 |            1.46 | Stage 4 improved alternation somewhat, but still below stage-1 best |
| `training_016` | `uv run python evaluate_all_checkpoints.py` |      latest |        273.1 |            24.6 |            1.67 | Current benchmark for dual-resource alternation; strongest balanced final policy so far |
| `training_016` | `uv run python evaluate_all_checkpoints.py` |        best |        281.6 |            36.4 |            1.72 | Highest-consumption checkpoint, but not the most balanced homeostatic policy |
| `training_016` | `uv run python evaluate_all_checkpoints.py` | stage1_best |        150.9 |            22.0 |            1.10 | Stage 1 remains strong, but still single-resource specialized |
| `training_016` | `uv run python evaluate_all_checkpoints.py` | stage2_best |        150.9 |            22.5 |            1.10 | Stage-1 → stage-2 bridge preserved competence much better than training_015 |
| `training_016` | `uv run python evaluate_all_checkpoints.py` | stage3_best |        215.2 |            31.9 |            1.36 | Stage 3 regained strong resource acquisition, but alternation stayed limited |
| `training_016` | `uv run python evaluate_all_checkpoints.py` | stage4_best |        136.6 |             0.7 |            1.67 | Stage 4 collapsed again, but revert protection prevented long-term damage |

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

### `training_009`

Source run: [analysis/training_009](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_009/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_009

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          155.2           19.2            1.17
best            150.9           21.0            1.06
stage1_best     150.9           18.5            1.04
stage2_best     150.9           21.0            1.06
stage3_best     150.9           20.9            1.10
stage4_best     150.9           18.1            1.06
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             0.26 |                     0.26 |           100.0% |      100.0% |                      26.0 |
| best        |             0.39 |                     0.39 |           100.0% |      100.0% |                      53.5 |
| stage1_best |             0.71 |                     0.71 |           100.0% |      100.0% |                      95.0 |
| stage2_best |             0.39 |                     0.39 |           100.0% |      100.0% |                      53.5 |
| stage3_best |             0.37 |                     0.37 |           100.0% |      100.0% |                      36.5 |
| stage4_best |             0.37 |                     0.37 |           100.0% |      100.0% |                      28.9 |

Long-horizon diagnostics:

| Checkpoint  | First food | First water | Food→Water | Water→Food | Alternating visits |
| ----------- | ---------: | ----------: | ---------: | ---------: | -----------------: |
| latest      |     1000.0 |        26.0 |       0.0% |       0.0% |                0.0 |
| best        |       53.5 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage1_best |       95.0 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage2_best |       53.5 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage3_best |       36.5 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage4_best |       28.9 |      1000.0 |       0.0% |       0.0% |                0.0 |

Verdict:

- `training_009` is strong at short-horizon resource acquisition: every listed checkpoint reaches and consumes a resource reliably.
- `latest` is a genuinely usable final checkpoint and is the fastest to first consumption in the provided evaluation.
- `stage3_best` and `stage4_best` remain competent, so the earlier stage-3 collapse problem is much reduced.
- However, the long-horizon metrics show complete single-resource specialization: no checkpoint alternates between food and water in evaluation.
- The global best still comes from stage 2, which matches the run log and suggests later stages are not yet producing a better dual-resource policy.

### `training_011`

Source run: [analysis/training_011](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_011/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_011

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          136.6           0.0             1.72
best            155.2           22.7            1.10
stage1_best     147.4           18.2            1.08
stage2_best     144.6           11.8            1.27
stage3_best     155.2           22.7            1.10
stage4_best     136.6           0.0             1.72
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             1.52 |                     1.52 |             0.0% |        0.0% |                    1000.0 |
| best        |             0.70 |                     0.70 |           100.0% |      100.0% |                     110.5 |
| stage1_best |             0.80 |                     0.80 |            95.0% |       95.0% |                     144.7 |
| stage2_best |             0.93 |                     0.93 |            90.0% |       90.0% |                     141.4 |
| stage3_best |             0.70 |                     0.70 |           100.0% |      100.0% |                     110.5 |
| stage4_best |             1.38 |                     1.38 |             0.0% |        0.0% |                    1000.0 |

Long-horizon diagnostics:

| Checkpoint  | First food | First water | Food→Water | Water→Food | Alternating visits |
| ----------- | ---------: | ----------: | ---------: | ---------: | -----------------: |
| latest      |     1000.0 |      1000.0 |       0.0% |       0.0% |                0.0 |
| best        |     1000.0 |       110.5 |       0.0% |       0.0% |                0.0 |
| stage1_best |      144.7 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage2_best |      141.4 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage3_best |     1000.0 |       110.5 |       0.0% |       0.0% |                0.0 |
| stage4_best |     1000.0 |      1000.0 |       0.0% |       0.0% |                0.0 |

Verdict:

- `training_011` regresses back toward the older failure mode: the final checkpoint and `stage4_best` fail to reach any resource in evaluation.
- The strongest checkpoint comes from stage 3, not from the later stages; the run log also reports the global best at episode 220 during `3-six-resources`.
- `best` and `stage3_best` are still usable single-resource policies, but the later curriculum stages do not preserve that competence.
- Long-horizon diagnostics remain completely flat: no checkpoint alternates between food and water, so dual-resource behavior is still not learned.

### `training_015`

Source run: [analysis/training_015](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_015/)

Command:

```bash
uv run python evaluate_all_checkpoints.py
```

Output:

```text
[evaluate] run_dir=analysis/training_015

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          138.4           13.9            1.36
best            335.2           37.9            1.50
stage1_best     335.2           37.9            1.50
stage2_best     136.6           1.7             1.64
stage3_best     203.6           24.4            1.12
stage4_best     231.2           29.0            1.46
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             0.66 |                     0.66 |            55.0% |       55.0% |                     466.1 |
| best        |             0.34 |                     0.34 |           100.0% |      100.0% |                      49.8 |
| stage1_best |             0.34 |                     0.34 |           100.0% |      100.0% |                      49.8 |
| stage2_best |             1.21 |                     1.21 |            10.0% |       10.0% |                     905.5 |
| stage3_best |             0.58 |                     0.58 |           100.0% |      100.0% |                      21.9 |
| stage4_best |             0.84 |                     0.84 |            75.0% |       75.0% |                     330.0 |

Long-horizon diagnostics:

| Checkpoint  | First food | First water | Food→Water | Water→Food | Alternating visits |
| ----------- | ---------: | ----------: | ---------: | ---------: | -----------------: |
| latest      |      466.1 |       907.1 |      10.0% |      10.0% |                0.2 |
| best        |       49.8 |        62.6 |     100.0% |     100.0% |               15.3 |
| stage1_best |       49.8 |        62.6 |     100.0% |     100.0% |               15.3 |
| stage2_best |     1000.0 |       905.5 |       0.0% |       0.0% |                0.0 |
| stage3_best |       29.8 |       674.1 |      35.0% |      25.0% |                1.6 |
| stage4_best |      604.1 |       330.0 |      45.0% |      45.0% |                3.5 |

Verdict:

- `training_015` shows that stage 1 can now produce a genuinely strong policy: `best` and `stage1_best` match exactly and are the strongest checkpoints in the run.
- Stage 2 is the main failure point. `stage2_best` collapses to near-total failure, with almost no resource entry or useful dual-resource behavior.
- Stage 3 partially recovers after the stage-2 collapse, and stage 4 improves alternation somewhat further, but neither surpasses the best stage-1 checkpoint.
- The final checkpoint remains usable, but it is much weaker than the best checkpoint and only shows limited food-water alternation.
- Overall, this run supports the conclusion that the critical bottleneck is no longer basic stage-1 learning, but preserving stage-1 competence across the stage-1 → stage-2 transition.

### `training_016`

Source run: [analysis/training_016](/Users/khanhanzalaareeb/Documents/project/CHRL/analysis/training_016/)

Command:

```bash
uv run python evaluate_all_checkpoints.py --run-dir analysis/training_016
```

Output:

```text
[evaluate] run_dir=analysis/training_016

Checkpoint      Avg survival    Avg consumption Avg final drive
latest          273.1           24.6            1.67
best            281.6           36.4            1.72
stage1_best     150.9           22.0            1.10
stage2_best     150.9           22.5            1.10
stage3_best     215.2           31.9            1.36
stage4_best     136.6           0.7             1.67
```

Resource-interaction diagnostics:

| Checkpoint  | Nearest resource | Minimum distance reached | Resource entered | Consumption | Time to first consumption |
| ----------- | ---------------: | -----------------------: | ---------------: | ----------: | ------------------------: |
| latest      |             0.09 |                     0.09 |           100.0% |      100.0% |                      23.9 |
| best        |             0.07 |                     0.07 |           100.0% |      100.0% |                      27.6 |
| stage1_best |             0.62 |                     0.62 |           100.0% |      100.0% |                      48.5 |
| stage2_best |             0.78 |                     0.78 |           100.0% |      100.0% |                      41.3 |
| stage3_best |             0.48 |                     0.48 |           100.0% |      100.0% |                      20.2 |
| stage4_best |             1.01 |                     1.01 |            15.0% |       15.0% |                     855.0 |

Long-horizon diagnostics:

| Checkpoint  | First food | First water | Food→Water | Water→Food | Alternating visits |
| ----------- | ---------: | ----------: | ---------: | ---------: | -----------------: |
| latest      |      131.2 |        23.9 |      90.0% |      90.0% |               21.5 |
| best        |      240.6 |        27.6 |      80.0% |      80.0% |                4.3 |
| stage1_best |       48.5 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage2_best |       41.3 |      1000.0 |       0.0% |       0.0% |                0.0 |
| stage3_best |       20.2 |       694.8 |      35.0% |      20.0% |                1.1 |
| stage4_best |      855.0 |      1000.0 |       0.0% |       0.0% |                0.0 |

Verdict:

- `training_016` is the current benchmark run because `latest` is the strongest final checkpoint so far for genuine dual-resource alternation.
- The new stage-1 → stage-2 bridge helped: stage 2 no longer catastrophically erased the policy the way it did in `training_015`.
- Stage 2 still does not reliably teach switching itself. The stage-2 checkpoint remains mostly single-resource, even though it preserves competence better than before.
- Stage 4 collapsed again, but the revert protection worked and prevented that collapse from defining the final outcome.
- The checkpoint-selection rule is now visibly misaligned with the scientific objective: `best` has higher consumption, but `latest` is clearly the more balanced homeostatic policy.

Benchmark findings:

- Use `training_016 latest` as the practical benchmark for now when comparing future runs on balanced food-water behavior.
- Keep `training_016 best` as a high-consumption reference, but not as the main homeostatic benchmark.
- Stage 1 anchor inheritance and stage-2 revert logic should stay; both changes improved stability.

Proposed optimization priorities for future testing:

1. Change best-checkpoint selection to include dual-resource behavior directly.
   - Rank using food→water, water→food, alternating visits, and then average consumption.

2. Tighten stage-2 objective around switching rather than simple consumption.
   - Reward or select for reaching the opposite resource after first consumption.
   - Track time-to-second-resource explicitly.

3. Keep stage-1-best inheritance into stage 2.
   - This reduced catastrophic forgetting and should remain part of the curriculum.

4. Keep stage-4 collapse protection and revert behavior.
   - This successfully prevented destructive late-stage drift in `training_016`.

5. Continue using the stage-2 diagnostics added in this run.
   - Especially: nearby opposite resource ignored, heading entropy, action variance, and cluster visits.
