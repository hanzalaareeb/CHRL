import torch


class Config:
    """Central configuration for the Continuous Homeostatic RL (DDPG) project.

    Every module reads its hyperparameters from here so experiments can be
    tuned in a single place. Import with ``from config import Config``.
    """

    # ----- Reproducibility -----
    SEED = 42

    # ----- Time -----
    DT = 0.1                      # Continuous-time integration step
    MAX_STEPS = 1000              # Max steps per episode before truncation

    # ----- Internal physiology -----
    # Two homeostatic variables: index 0 = "food", index 1 = "water".
    N_INTERNAL = 2
    H_STAR = 1.0                  # Target (set-point) level for each internal variable
    DECAY_RATES = [0.05, 0.05]    # Per-variable decay rate (food, water)
    INIT_STATE_LOW = 0.5          # Internal state starts randomly depleted in
    INIT_STATE_HIGH = 1.0         # [LOW, HIGH] so there is an immediate drive to reduce.

    # ----- Observation -----
    # Per resource the agent sees [dx, dy, distance, capacity_fraction, active]:
    # dx/dy give direction ("move left"), distance gives magnitude, capacity tells
    # if depleted, active marks whether the slot is a real resource (curriculum
    # stages use fewer than MAX_RESOURCES, so the obs is padded to a fixed size).
    NORMALIZE_OBS = True          # Scale position/distance/internal-state to ~[-1, 1]
    MAX_RESOURCES = 6             # Fixed number of resource slots in the observation

    # ----- World / resources -----
    WORLD_SIZE = 6.0              # Agent position is clipped to [-WORLD_SIZE, WORLD_SIZE]
    CONSUME_RADIUS = 1.0          # Agent consumes a resource when within this radius
    EFFORT_PENALTY = 0.01         # HJB effort cost weighting on actions
    REWARD_SCALE = 1.0            # Global multiplier on the per-step reward
    REGEN_DELAY = 20              # Steps a depleted store stays empty before regenerating (stage 3)

    # Survival bonus is now STATE-DEPENDENT so "just exist and die" no longer pays:
    # the bonus shrinks to 0 as the agent starves, restoring a discriminative reward.
    #   "exp_drive": bonus = SURVIVAL_BONUS * exp(-drive)      (max at the set-point)
    #   "min_state": bonus = SURVIVAL_BONUS * min(internal)    (gated by the worst reserve)
    SURVIVAL_BONUS = 0.0
    SURVIVAL_BONUS_MODE = "exp_drive"   # "exp_drive" or "min_state"

    # Resource sites scattered at variable distances/locations.
    # Each: (x, y, type, consume_rate, max_capacity, regen_rate)
    #   type:          "food" -> internal[0],  "water" -> internal[1]
    #   consume_rate:  internal-state gain rate while consuming (food > water, per request)
    #   max_capacity:  finite store; depletes ~2-3 refills then must regenerate (anti-camping)
    #   regen_rate:    slow capacity recovery per unit time
    RESOURCES = [
        ( 3.5,  0.5, "food",  0.60, 1.20, 0.010),
        (-3.0,  2.5, "water", 0.40, 0.90, 0.010),
        ( 1.0,  3.5, "food",  0.50, 1.00, 0.008),
        (-2.5, -3.0, "water", 0.35, 0.80, 0.010),
        ( 4.0, -2.0, "food",  0.55, 1.10, 0.009),
        (-4.5, -0.5, "water", 0.45, 0.90, 0.010),
    ]

    # ----- Curriculum -----
    # Train one agent through progressively harder stages, changing ONE variable at
    # a time: resource count (2 -> 4 -> 6), then depletion, then regeneration delay.
    # Observation is padded to MAX_RESOURCES so the SAME network transfers across
    # stages. A tiny survival bonus seeds stage 1 and is annealed to 0 by stage 3.
    # Each stage runs up to `episodes`; it advances early once the recent success
    # rate exceeds STAGE_ADVANCE_THRESHOLD (see below). Harder stages get more budget.
    CURRICULUM = True
    # Non-depleting variants of the 6 sites (huge capacity, fast regen) for stages 1-3.
    _BIG_6 = [(x, y, t, c, 1000.0, 1.0) for (x, y, t, c, _m, _r) in RESOURCES]
    STAGES = [
        {   # change: 2 resources (navigation only)
            "name": "1-navigation",
            "resources": [
                ( 1.5,  0.0, "food",  0.60, 1000.0, 1.0),
                (-1.5,  0.0, "water", 0.60, 1000.0, 1.0),
            ],
            "episodes": 50, "regen_delay": 0, "survival_bonus": 0.01, "noise_floor": 0.10,
        },
        {   # change: 4 resources (selection among types)
            "name": "2-four-resources",
            "resources": [
                ( 2.0,  0.0, "food",  0.60, 1000.0, 1.0),
                (-2.0,  0.0, "water", 0.50, 1000.0, 1.0),
                ( 0.0,  2.5, "food",  0.55, 1000.0, 1.0),
                ( 0.0, -2.5, "water", 0.50, 1000.0, 1.0),
            ],
            "episodes": 100, "regen_delay": 0, "survival_bonus": 0.005, "noise_floor": 0.08,
        },
        {   # change: 6 resources (still non-depleting)
            "name": "3-six-resources",
            "resources": _BIG_6,
            "episodes": 500, "regen_delay": 0, "survival_bonus": 0.0, "noise_floor": 0.08,
        },
        {   # change: depletion (finite capacities)
            "name": "4-depletion",
            "resources": RESOURCES,
            "episodes": 120, "regen_delay": 0, "survival_bonus": 0.0, "noise_floor": 0.06,
        },
        {   # change: regeneration delay
            "name": "5-regen-delay",
            "resources": RESOURCES,
            "episodes": 120, "regen_delay": REGEN_DELAY, "survival_bonus": 0.0, "noise_floor": 0.05,
        },
    ]
    # Advance to the next stage early once success rate over the last
    # STAGE_ADVANCE_WINDOW episodes exceeds the threshold (success = survived the
    # whole episode). Otherwise the stage runs out its `episodes` budget.
    STAGE_ADVANCE_WINDOW = 20
    STAGE_ADVANCE_THRESHOLD = 0.80
    SUCCESS_RATE_WINDOW = 100      # window for the rolling "Success/Rate" metric

    # ----- Prioritized replay (simple approximation) -----
    # Sample transitions with probability ~ (|reward| + eps)^alpha, and boost the
    # priority of transitions from successful episodes so good rollouts replay more.
    PRIORITIZED_REPLAY = True
    PER_ALPHA = 0.6
    PER_EPSILON = 0.01
    SUCCESS_PRIORITY_BOOST = 5.0
    MEANINGFUL_CONSUMPTION_STEP = 200
    STAGE3_MIX_EPISODES = 50
    STAGE3_MIX_FRACTION = 0.50

    # ----- Checkpoints (best-eval) + buffer snapshot -----
    BEST_ACTOR_PATH = "best_actor.pth"
    BEST_CRITIC_PATH = "best_critic.pth"
    BUFFER_PATH = "replay_buffer.npz"   # saved whenever a successful episode occurs

    # ----- Network -----
    HIDDEN_DIM = 128              # Hidden layer width for actor & critic

    # ----- DDPG hyperparameters -----
    GAMMA = 0.99                  # Discount factor
    TAU = 0.005                   # Soft target update coefficient
    ACTOR_LR = 3e-5
    CRITIC_LR = 1e-3

    # ----- Replay buffer -----
    BUFFER_SIZE = 100_000
    BATCH_SIZE = 128

    # ----- Agent -----
    AGENT = "td3"                 # "ddpg" or "td3"

    # ----- Exploration noise -----
    NOISE_TYPE = "gaussian"       # "gaussian" or "ou"
    EXPLORATION_NOISE = 0.1       # Initial exploration std (decayed during training)
    EXPLORATION_NOISE_FINAL = 0.02   # Floor the exploration std decays to
    EXPLORATION_DECAY_FRAC = 0.8  # Decay linearly over this fraction of MAX_EPISODES
    STAGE3_EXPLORATION_RESET = 0.12
    STAGE3_EXPLORATION_RESET_EPISODES = 50
    OU_MU = 0.0
    OU_THETA = 0.15
    OU_SIGMA = 0.2

    # ----- TD3-specific -----
    POLICY_NOISE = 0.2            # Std of target-policy smoothing noise
    NOISE_CLIP = 0.5             # Target smoothing noise clipped to [-NOISE_CLIP, NOISE_CLIP]
    POLICY_DELAY = 4             # Actor & target nets updated every POLICY_DELAY critic steps

    # ----- Training -----
    MAX_EPISODES = 200
    START_STEPS = 1_000           # Warm-up steps of random actions before using the policy
    LEARNING_STARTS = 5_000       # Don't run gradient updates until the buffer holds this
                                  # many transitions (skips early low-quality data)
    EVAL_INTERVAL = 20            # Run a deterministic (noise=0) eval episode this often
    EVAL_EPISODES = 5             # Average each evaluation over fixed-seed episodes
    STAGE3_CHECKPOINT_INTERVAL = 50
    STAGE4_CHECKPOINT_INTERVAL = 50
    PRIMARY_METRIC = "avg_consumption"
    EARLY_STOPPING = True
    EARLY_STOPPING_PATIENCE = 6   # stop after this many non-improving evals
    EARLY_STOPPING_MIN_STAGE = 3  # only start early-stop checks from stage 3 onward
    EARLY_STOPPING_MIN_DELTA = 0.25
    STAGE3_MIN_EPISODES_BEFORE_EARLY_STOP = 100

    # ----- Logging / checkpoints -----
    LOG_DIR = "runs/homeostatic_ddpg"
    ACTOR_PATH = "homeostatic_actor.pth"
    CRITIC_PATH = "homeostatic_critic.pth"
    LOG_INTERVAL = 20             # Episodes between console summaries
    PLOT_DIR = "plots"            # Where visualize.py / diagnostics.py save figures

    @staticmethod
    def get_device():
        """Pick the best available compute device (MPS > CUDA > CPU)."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
