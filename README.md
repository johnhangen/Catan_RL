# Catan RL

A training framework for teaching an AI agent to play Settlers of Catan, built on top of [Catanatron](https://github.com/bcollazo/catanatron). Supports curriculum learning, self-play, and a real-life mode where a human executes the AI's moves on a physical board.

## Features

- **MaskablePPO** training — illegal actions are masked so the agent never wastes gradient on them
- **Curriculum learning** — three stages (Random → WeightedRandom → Self-play) with automatic win-rate graduation
- **Self-play** — the agent periodically plays against a frozen copy of itself, updating opponents from disk between games
- **Three reward functions** — sparse, VP-delta, and a shaped reward with road/army bonuses
- **Real-life mode** — CLI that announces AI moves in plain English and prompts an operator to relay human moves; dice are injected from the physical board so virtual and physical game states stay in sync
- **Evaluation script** — benchmark any checkpoint against random and weighted-random opponents

## Installation

Python 3.9+ required.

```bash
git clone https://github.com/johnhangen/catan_rl.git
cd catan_rl
pip install -r requirements.txt
```

## Quick Start

### Train with default config

```bash
python train.py
# or explicitly:
python train.py --config config/default.yaml
```

Checkpoints and TensorBoard logs are written to `runs/<timestamp>/`.

### Resume a run

```bash
python train.py --config config/default.yaml --resume runs/2025-05-16_12-00-00/best_model.zip
```

### Fast smoke test (CI / debugging)

```bash
python train.py --config config/fast_debug.yaml
```

### Evaluate a checkpoint

```bash
python evaluate.py --model runs/2025-05-16_12-00-00/best_model.zip
python evaluate.py --model runs/.../best_model.zip --opponents random weighted --games 200
```

### Real-life mode

One human operator sits at a computer. The AI announces its moves; the operator executes them on the physical board and enters what the human players do.

```bash
python play_reallife.py --model runs/.../best_model.zip
python play_reallife.py --model runs/.../best_model.zip --players 3   # 1 AI + 2 humans
```

The operator is prompted to enter dice rolls from the physical board so resource production stays in sync with the virtual state.

## Project Structure

```
Catan_RL/
├── train.py                    # Training entry point
├── evaluate.py                 # Benchmark a checkpoint
├── play_reallife.py            # Real-life mode entry point
│
├── config/
│   ├── default.yaml            # Full 3.5M-step curriculum run
│   ├── selfplay.yaml           # Start directly at self-play stage
│   └── fast_debug.yaml         # 500 steps, 1 env — for CI
│
├── envs/
│   ├── make_env.py             # Environment factory + action mask helper
│   ├── rewards.py              # sparse / vp_delta / shaped reward functions
│   └── wrappers.py             # EpisodeStatsWrapper (win/vp into info["episode"])
│
├── agents/
│   ├── policy_player.py        # Trained SB3 model wrapped as a Catanatron Player
│   ├── human_relay_player.py   # Terminal-prompt player for real-life opponents
│   └── checkpoint.py           # save_checkpoint / load helpers
│
├── training/
│   ├── curriculum.py           # Stage definitions + CurriculumManager
│   └── callbacks.py            # WinRateCallback, CheckpointCallback,
│                               # CurriculumCallback, SelfPlayCallback
│
├── reallife/
│   └── cli.py                  # run_reallife() loop with rich terminal UI
│
└── tests/
    ├── test_env.py
    ├── test_rewards.py
    ├── test_curriculum.py
    └── test_fixes.py           # Regression tests for reviewed bugs
```

## Configuration

All hyperparameters live in a YAML file. Key sections:

### `env`

| Key | Default | Description |
|-----|---------|-------------|
| `map_type` | `BASE` | Catanatron map type |
| `vps_to_win` | `10` | Victory points required to win |
| `representation` | `vector` | Observation encoding (`vector` or `mixed`) |
| `reward_function` | `shaped` | `sparse`, `vp_delta`, or `shaped` |
| `num_players` | `4` | Total players (1 agent + opponents) |
| `num_envs` | `8` | Parallel environments for training |

### `training`

Standard MaskablePPO hyperparameters — `learning_rate`, `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `ent_coef`, `vf_coef`, `max_grad_norm`, `total_timesteps`.

### `curriculum`

```yaml
curriculum:
  enabled: true
  stages:
    - name: random       # opponents: RandomPlayer
      timesteps: 500_000
      graduate_winrate: 0.60
    - name: weighted     # opponents: WeightedRandomPlayer
      timesteps: 1_000_000
      graduate_winrate: 0.55
    - name: selfplay     # opponents: PolicyPlayer (frozen copy of self)
      timesteps: 2_000_000
      graduate_winrate: null
  eval_freq: 50_000
```

The agent advances to the next stage when its rolling win rate crosses `graduate_winrate`. Graduation is checked every `eval_freq` steps.

### `rewards` (shaped reward weights)

```yaml
rewards:
  win: 10.0
  loss: -2.0
  vp_delta_scale: 1.0
  longest_road: 0.5
  largest_army: 0.5
  turn_penalty: -0.001
```

## Reward Functions

| Name | Description |
|------|-------------|
| `sparse` | +1 on win, −1 on loss, 0 otherwise |
| `vp_delta` | Change in victory points each turn, scaled; +10 win bonus |
| `shaped` | VP delta + longest road/army bonuses, turn penalty, configurable weights |

Select via `env.reward_function` in the config YAML.

## Self-Play

When the curriculum reaches the `selfplay` stage (or when `config/selfplay.yaml` is used directly):

1. The current model is saved to `runs/<run>/selfplay_opponent.zip`
2. `PolicyPlayer` opponents load their weights from this file at the start of each game (`reset_state()` clears the cached model so disk is re-read)
3. `SelfPlayCallback` overwrites `selfplay_opponent.zip` every 50k steps, gradually improving the opponent pool

## Running Tests

```bash
pytest tests/
```

All 28 tests should pass. The `test_fixes.py` suite covers regression cases found during code review (memory leak in reward state, dice injection, model reload on reset, etc.).

## Real-Life Mode Details

The AI plays as **Blue**. Human players are Red, Orange, and White (depending on `--players`).

- **AI turns**: the chosen action is printed in plain English, e.g. `Build a settlement at the intersection of 6-Hills, 8-Mountains, 4-Forest (node 23)`. Press Enter after executing it on the board.
- **Human turns**: Catanatron calls `HumanRelayPlayer.decide()` automatically. The operator picks from a numbered list of legal actions.
- **Dice**: both AI and human ROLL actions prompt for the actual dice values so virtual resource production matches the physical board.

## Dependencies

| Package | Purpose |
|---------|---------|
| `catanatron` | Full Catan rules engine |
| `catanatron-gym` | Gymnasium wrapper around Catanatron |
| `stable-baselines3` | PPO implementation and VecEnv utilities |
| `sb3-contrib` | MaskablePPO (action masking) |
| `torch` | Neural network backend |
| `rich` | Terminal UI for real-life mode |
| `tensorboard` | Training curve visualisation |
