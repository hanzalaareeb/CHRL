# CHRL — Continuous Homeostatic Reinforcement Learning

A continuous-time / continuous-space homeostatic RL environment (after Laurençon et al.) with
DDPG and TD3 agents. An agent moves in 2D and must keep its internal physiological variables
(**food**, **water**) near their set-point `H*` by foraging from depletable, regenerating resource
sites. The reward follows the drive-reduction formulation `r = D(H_t) − D(H_{t+1})`.

## Quick start

```bash
uv run python main.py                  # curriculum TD3 (default), then diagnostics + plots
uv run python main.py --no-curriculum  # single-stage training on the full 6-resource layout
uv run python main.py --agent ddpg     # use DDPG instead of TD3
uv run python main.py --no-train       # load saved model, just diagnose + visualize
uv run tensorboard --logdir runs       # watch training curves (incl. Eval/* and Q/Q1_minus_Q2)
```

Standalone tools (load the saved model, or fall back to a random policy):

```bash
uv run python visualize.py     # trajectory, internal-state, and per-resource plots -> plots/
uv run python diagnostics.py   # critic-bias + reward-scale analysis -> plots/diagnostics.png
```

## Layout

| File | Role |
|------|------|
| `config.py` | All hyperparameters (one place to tune). |
| `env.py` | `ContinuousHomeostaticEnv` (Gymnasium API). |
| `network.py` | `Actor`, `Critic`, `TwinCritic`. |
| `agents/` | `DDPGAgent`, `TD3Agent`, `make_agent` factory. |
| `noise.py` | Gaussian / Ornstein-Uhlenbeck exploration noise. |
| `replay_buffer.py` | Experience replay. |
| `train.py` | `train`, `train_curriculum`, periodic noise-free evaluation. |
| `diagnostics.py` | Critic value-bias + reward-scale diagnostics. |
| `visualize.py` | Trajectory / internal-state / per-resource plots. |
| `main.py` | Entry point wiring it together. |

## Experiment log

### `analysis/test_result_td3_01.txt` — "minimise effort and die"

A full 200-episode TD3 run looked healthy on paper (episode reward ~3–7, critic `Q` rising to 1.18)
but the behaviour was a **total failure**: `Consumed: 0` on ~196/200 episodes, every eval
`survived=no`, `Final Drive` pinned at ~1.3–2.0, and the agent died around step ~140 every time.

The diagnostics found the cause. The per-step reward histogram (`plots/diagnostics.png`) was a narrow
spike at **+0.03–0.05** — i.e. almost entirely the **constant `SURVIVAL_BONUS = 0.05`**. The real
homeostatic signal (drive reduction, ~0.005) was drowned out ~10×, so the reward was **nearly
constant regardless of behaviour**. With no gradient to distinguish "go eat" from "sit still," the
actor collapsed to the minimum-effort policy: barely move, collect the flat bonus, die. The rising
`Q` had simply learned `Q ≈ survival_bonus × steps-until-death`.

> **On exploration.** Interestingly, this failure connects to the earlier idea that exploration
> shouldn't be *directly rewarded*. This experiment supports that intuition: if exploration has no
> chance of producing a meaningful physiological improvement, the optimal policy is exactly what TD3
> found — minimise effort, collect the constant reward, die. **That isn't a bug in TD3; it's a
> rational solution to the reward we defined.** Fix the reward and the curriculum, and the optimum
> moves with it.

The eval / final-drive / consumption metrics did their job: they exposed a failure the reward curve
completely masked.

## The fix

### 1. State-dependent survival bonus

The constant bonus is replaced by one that **vanishes as the agent starves**, so "just exist and die"
no longer pays. Configurable via `SURVIVAL_BONUS` and `SURVIVAL_BONUS_MODE`:

- `"exp_drive"`: `bonus = SURVIVAL_BONUS · exp(−drive)` — maximal at the set-point, → 0 as drive grows.
- `"min_state"`: `bonus = SURVIVAL_BONUS · min(internal_state)` — gated by the *worst* reserve, so the
  agent must balance **both** food and water.

Combined with the per-step drive-reduction reward (and the earlier removal of the `1/dt` cost
amplification), drift-and-die now telescopes to a clearly **negative** return while a homeostatic
trajectory earns a positive one. Setting `SURVIVAL_BONUS = 0` recovers the pure, paper-faithful
drive-reduction reward.

### 2. Direction-aware, normalized observation

Each resource is now encoded as `[dx, dy, distance, capacity_fraction, active]`. `dx/dy` give the
agent *direction* ("move left"), not just a scalar distance. Position, distance and internal state are
normalized to ~[−1, 1] (`NORMALIZE_OBS`).

### 3. Staged curriculum (`CURRICULUM = True`)

One agent is trained through progressively harder environments so navigation skill transfers. The
observation is **padded to `MAX_RESOURCES`** (inactive slots flagged `active=0`), keeping the network
input fixed across stages:

| Stage | Resources | Goal |
|-------|-----------|------|
| 1 — navigation | 1 food + 1 water, ~1.5 units away, no depletion | learn to reach a resource |
| 2 — selection | 3 resources (mixed types) | learn to pick the type it needs |
| 3 — full | 6 resources + depletion + regeneration + **regen delay** | the full task |

A depleted store in stage 3 stays empty for `REGEN_DELAY` steps before regenerating, discouraging
camping. Edit `STAGES` in `config.py` to retune.

## Notes

- The observation changed shape (now `2 + N_INTERNAL + 5·MAX_RESOURCES`), so **models saved before this
  change are incompatible** — retrain from scratch.
- DDPG vs TD3 is selected by `Config.AGENT` or `--agent`; both run through the same training loop.
  TD3 adds clipped double-Q, target-policy smoothing, and delayed updates (`POLICY_DELAY = 2`), and
  logs `Q/Q1`, `Q/Q2`, and `Q/Q1_minus_Q2` (a divergence early-warning).
