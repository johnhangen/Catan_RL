# Catan RL Autoresearch Program

## Goal
Maximize win rate against `WeightedRandomPlayer` opponents in 4-player Catan.
A perfectly uniform agent expects 0.25 win rate; a strong trained agent should
exceed 0.45. Target: beat 0.50 within the experiment budget.

## Search Space
You may change any of these fields in the config YAML:

**Training hyperparameters:**
- `training.learning_rate` — try 5e-5 to 1e-3 (log scale)
- `training.n_steps` — 512 / 1024 / 2048 / 4096
- `training.batch_size` — 64 / 128 / 256 / 512
- `training.n_epochs` — 5 to 20
- `training.gamma` — 0.95 to 0.999
- `training.gae_lambda` — 0.90 to 0.99
- `training.clip_range` — 0.1 to 0.3
- `training.ent_coef` — 0.001 to 0.05
- `training.vf_coef` — 0.25 to 1.0

**Reward weights** (shaped reward only):
- `rewards.win` — 5 to 20
- `rewards.loss` — -5 to -0.5
- `rewards.vp_delta_scale` — 0.1 to 3.0
- `rewards.longest_road` — 0.0 to 2.0
- `rewards.largest_army` — 0.0 to 2.0
- `rewards.turn_penalty` — -0.01 to 0.0

**Reward function** (env setting):
- `env.reward_function` — `sparse`, `vp_delta`, or `shaped`

## Constraints
Do NOT change:
- `env.map_type`, `env.representation`, `env.num_players`
- `training.algorithm`, `training.device`
- `training.total_timesteps`, `env.num_envs` (overridden by the harness)
- Anything under `curriculum` or `logging` (managed by harness)

## Strategy
- **Explore one variable at a time** when entering a new region of search space.
- **Follow gradients**: if increasing X helped, try increasing it further before
  pivoting.
- **Combine winners**: once several independent improvements are found, try
  combining them in a single experiment.
- **Reward shaping is high-leverage**: if hyperparameters plateau, switch to
  tuning reward weights or the reward function itself.
- Catan games are long and noisy; win rate variance on 50 games is high (~0.07).
  A difference < 0.03 is not significant — don't over-interpret small deltas.

## Stopping Criteria
Declare success and stop proposing changes if any of these hold:
- Win rate exceeds 0.60 for three consecutive kept experiments
- Win rate has not improved by more than 0.02 across the last 8 experiments
- All major dimensions (lr, n_steps, batch_size, reward function, key reward
  weights) have been explored at least once
