#!/usr/bin/env python3
"""Karpathy-style autoresearch loop for Catan RL.

Claude proposes config changes, a short training probe runs, win rate is measured,
and the change is kept or discarded — overnight, without human input.

Usage:
    python autoresearch.py
    python autoresearch.py --experiments 30 --probe-steps 200000 --eval-games 50

Requirements:
    pip install anthropic
    ANTHROPIC_API_KEY must be set in the environment.
"""

import anthropic
import argparse
import copy
import json
import os
import re
import shutil
import subprocess
import sys
import time
import yaml
from pathlib import Path

MODEL = "claude-sonnet-4-6"

AUTORESEARCH_DIR = "runs/autoresearch"
LOG_FILE = os.path.join(AUTORESEARCH_DIR, "log.jsonl")
PROBE_CONFIG = "config/autoresearch_probe.yaml"
BEST_CONFIG = os.path.join(AUTORESEARCH_DIR, "best_config.yaml")
PROGRAM_FILE = "program.md"


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def patch_for_probe(config: dict, probe_steps: int, n_envs: int) -> dict:
    """Return a copy of config with probe-specific overrides that must not be tuned."""
    c = copy.deepcopy(config)
    c.setdefault("training", {})["total_timesteps"] = probe_steps
    c.setdefault("env", {})["num_envs"] = n_envs
    # Disable curriculum so the probe stays at the weighted stage the whole time.
    c["curriculum"] = {"enabled": False}
    c.setdefault("logging", {}).update({
        "tensorboard": False,
        "wandb": False,
        "checkpoint_freq": probe_steps,
        "keep_checkpoints": 1,
    })
    return c


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def run_training(config: dict, run_dir: str, probe_steps: int, n_envs: int) -> str:
    """Write patched config, run train.py, return path to saved model."""
    os.makedirs(run_dir, exist_ok=True)
    patched = patch_for_probe(config, probe_steps, n_envs)
    probe_cfg_path = os.path.join(run_dir, "probe_config.yaml")
    save_yaml(patched, probe_cfg_path)

    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--config", probe_cfg_path,
            "--run-dir", run_dir,
            "--stage", "weighted",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-4000:] or result.stdout[-4000:])

    for name in ("best_model.zip", "final_model.zip"):
        path = os.path.join(run_dir, name)
        if os.path.exists(path):
            return path
    raise RuntimeError("No model file found after training")


def run_eval(model_path: str, n_games: int) -> dict:
    """Run evaluate.py --json, return the stats dict for the weighted opponent."""
    result = subprocess.run(
        [
            sys.executable, "evaluate.py",
            "--model", model_path,
            "--opponents", "weighted",
            "--games", str(n_games),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])

    # The last line of stdout is the JSON array printed by --json
    for line in reversed(result.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("["):
            data = json.loads(line)
            return data[0]  # first (and only) opponent entry
    raise RuntimeError(f"No JSON in evaluate output:\n{result.stdout[:1000]}")


# ---------------------------------------------------------------------------
# Claude interaction
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an RL research assistant optimising a MaskablePPO agent to play "
    "Settlers of Catan. Your only tool is editing a YAML config file. "
    "Each experiment trains for a fixed number of steps; the metric is win rate "
    "against WeightedRandomPlayer opponents in 4-player Catan. "
    "Propose one clear, testable change per experiment. Reason from the history."
)


def build_user_message(program_md: str, current_config_str: str, history: list[dict]) -> str:
    if not history:
        history_block = "No experiments yet — this is the first one."
    else:
        rows = []
        for e in history[-15:]:
            tag = "KEPT ✓" if e["kept"] else "discarded"
            rows.append(
                f"Exp {e['exp']:>2} [{tag}]  win={e['win_rate']:.3f}  "
                f"mean_vp={e.get('mean_vp', '?'):.1f}  "
                f"— {e['reasoning'][:120]}"
            )
        history_block = "\n".join(rows)

    return f"""\
<program>
{program_md}
</program>

<current_config>
{current_config_str}
</current_config>

<history>
{history_block}
</history>

Propose the next experiment. Your reply must contain:
1. A <reasoning>…</reasoning> block — what you're changing, why, what you expect.
2. The complete updated YAML config in a ```yaml … ``` fenced block.

Do not truncate the config; include every field."""


def ask_claude(client: anthropic.Anthropic, program_md: str, current_config_str: str, history: list) -> str:
    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": build_user_message(program_md, current_config_str, history),
            }
        ],
    )
    return response.content[0].text


def parse_reply(text: str) -> tuple[str, dict]:
    reasoning_m = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
    reasoning = reasoning_m.group(1).strip() if reasoning_m else text[:300].strip()

    yaml_m = re.search(r"```yaml\s*(.*?)```", text, re.DOTALL)
    if not yaml_m:
        raise ValueError("No ```yaml block in response")
    config = yaml.safe_load(yaml_m.group(1))
    if not isinstance(config, dict):
        raise ValueError("Parsed YAML is not a dict")
    return reasoning, config


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Autoresearch loop for Catan RL")
    parser.add_argument("--experiments", type=int, default=20,
                        help="Number of experiments to run")
    parser.add_argument("--probe-steps", type=int, default=200_000,
                        help="Training steps per experiment")
    parser.add_argument("--probe-envs", type=int, default=4,
                        help="Parallel envs per probe")
    parser.add_argument("--eval-games", type=int, default=50,
                        help="Games to evaluate each probe")
    args = parser.parse_args()

    for path, label in [(PROGRAM_FILE, "program.md"), (PROBE_CONFIG, "config/autoresearch_probe.yaml")]:
        if not os.path.exists(path):
            sys.exit(f"Missing {label} — {path}")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")

    program_md = Path(PROGRAM_FILE).read_text()
    os.makedirs(AUTORESEARCH_DIR, exist_ok=True)
    client = anthropic.Anthropic()

    # Load history if resuming
    history: list[dict] = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            history = [json.loads(l) for l in f if l.strip()]

    best_win_rate = 0.0
    kept_history = [e for e in history if e.get("kept")]
    if kept_history:
        best_win_rate = max(e["win_rate"] for e in kept_history)
        best_record = max(kept_history, key=lambda e: e["win_rate"])
        save_yaml(yaml.safe_load(best_record["config"]), PROBE_CONFIG)
        print(f"Resumed. Best win rate so far: {best_win_rate:.3f} (exp {best_record['exp']})")
    else:
        shutil.copy(PROBE_CONFIG, BEST_CONFIG)

    print(
        f"\nAutoresearch: {args.experiments} experiments, "
        f"{args.probe_steps:,} steps each, "
        f"{args.eval_games} eval games\n"
    )

    start_exp = len(history) + 1
    for exp_num in range(start_exp, start_exp + args.experiments):
        print(f"\n{'='*60}")
        print(f"Experiment {exp_num}")
        print(f"{'='*60}")

        current_config_str = Path(PROBE_CONFIG).read_text()

        # --- Claude proposes a change ---
        try:
            reply = ask_claude(client, program_md, current_config_str, history)
        except anthropic.APIError as e:
            print(f"API error: {e}. Skipping experiment.")
            continue

        print(f"\n[Claude]\n{reply[:700]}\n")

        try:
            reasoning, proposed = parse_reply(reply)
        except (ValueError, yaml.YAMLError) as e:
            print(f"Parse failed ({e}), skipping.")
            continue

        # --- Run training ---
        run_dir = os.path.join(AUTORESEARCH_DIR, f"exp_{exp_num:03d}")
        print(f"Training → {run_dir}")
        t0 = time.time()
        try:
            model_path = run_training(proposed, run_dir, args.probe_steps, args.probe_envs)
        except RuntimeError as e:
            print(f"Training failed:\n{str(e)[:1000]}")
            print("Reverting to best config.")
            if os.path.exists(BEST_CONFIG):
                shutil.copy(BEST_CONFIG, PROBE_CONFIG)
            continue
        elapsed = time.time() - t0
        print(f"Done in {elapsed / 60:.1f}m → {model_path}")

        # --- Evaluate ---
        print(f"Evaluating {args.eval_games} games vs weighted...")
        try:
            stats = run_eval(model_path, args.eval_games)
            win_rate = float(stats["win_rate"])
            mean_vp = float(stats.get("mean_vp", 0))
        except (RuntimeError, KeyError, json.JSONDecodeError, ValueError) as e:
            print(f"Eval failed: {e}")
            continue

        # --- Keep or discard ---
        kept = win_rate >= best_win_rate
        delta = win_rate - best_win_rate

        if kept:
            print(f"NEW BEST  {win_rate:.3f}  (+{delta:.3f})")
            best_win_rate = win_rate
            save_yaml(proposed, PROBE_CONFIG)
            save_yaml(proposed, BEST_CONFIG)
        else:
            print(f"Discarded {win_rate:.3f}  ({delta:+.3f} vs best {best_win_rate:.3f})")
            if os.path.exists(BEST_CONFIG):
                shutil.copy(BEST_CONFIG, PROBE_CONFIG)

        record = {
            "exp": exp_num,
            "reasoning": reasoning,
            "config": yaml.dump(proposed),
            "win_rate": win_rate,
            "mean_vp": mean_vp,
            "kept": kept,
            "elapsed_s": int(elapsed),
            "run_dir": run_dir,
        }
        history.append(record)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(record) + "\n")

        print(f"Win rate: {win_rate:.3f} | Best: {best_win_rate:.3f} | Time: {elapsed / 60:.1f}m")

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"Autoresearch complete after {args.experiments} experiments.")
    print(f"Best win rate: {best_win_rate:.3f}")
    print(f"Best config:   {BEST_CONFIG}")
    print(f"Full log:      {LOG_FILE}")

    if history:
        kept = [e for e in history if e["kept"]]
        print(f"\nKept experiments ({len(kept)}/{len(history)}):")
        for e in kept:
            print(f"  Exp {e['exp']:>2}: win={e['win_rate']:.3f}  {e['reasoning'][:80]}")


if __name__ == "__main__":
    main()
