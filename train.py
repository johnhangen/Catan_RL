#!/usr/bin/env python3
"""Main training script for Catan RL.

Usage:
    python train.py                              # default config
    python train.py --config config/default.yaml
    python train.py --config config/selfplay.yaml --resume runs/.../best_model.zip
    python train.py --config config/fast_debug.yaml
"""

import argparse
import os
import sys
import time
import yaml
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy

from envs.make_env import make_env
from training.curriculum import CurriculumManager, STAGES
from training.callbacks import (
    WinRateCallback,
    CheckpointCallback,
    CurriculumCallback,
    SelfPlayCallback,
)
from agents.checkpoint import save_checkpoint


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_vec_env(config: dict, stage: str = "weighted", model_path: str = None, n_envs: int = None):
    """Build a VecEnv with n_envs parallel environments."""
    n = n_envs or config["env"].get("num_envs", 4)
    config_with_stage = dict(config)
    config_with_stage.setdefault("curriculum", {})["current_stage"] = stage

    if n == 1:
        return DummyVecEnv([lambda rank=i: make_env(config_with_stage, rank=rank, model_path=model_path) for i in range(n)])
    return SubprocVecEnv([lambda rank=i: make_env(config_with_stage, rank=rank, model_path=model_path) for i in range(n)])


def main():
    parser = argparse.ArgumentParser(description="Train a Catan RL agent")
    parser.add_argument("--config", default="config/default.yaml", help="Path to YAML config")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--run-dir", default=None, help="Override run directory")
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config.get("training", {})
    env_cfg = config.get("env", {})
    log_cfg = config.get("logging", {})
    curriculum_cfg = config.get("curriculum", {})

    # Set up run directory
    run_dir = args.run_dir or os.path.join(
        log_cfg.get("run_dir", "runs"),
        time.strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(run_dir, exist_ok=True)
    print(f"Run directory: {run_dir}")

    # Save config snapshot
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Curriculum
    curriculum = CurriculumManager()
    if curriculum_cfg.get("enabled", True):
        curriculum.build_from_config(curriculum_cfg)
    initial_stage = curriculum.current_stage.name

    # Build training envs
    n_envs = env_cfg.get("num_envs", 4)
    vec_env = build_vec_env(config, stage=initial_stage, n_envs=n_envs)
    vec_env = VecMonitor(vec_env, info_keywords=("win", "vp"))

    # Create or load model
    total_timesteps = train_cfg.get("total_timesteps", 1_000_000)
    device = train_cfg.get("device", "auto")

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = MaskablePPO.load(args.resume, env=vec_env, device=device)
    else:
        model = MaskablePPO(
            MaskableActorCriticPolicy,
            vec_env,
            learning_rate=train_cfg.get("learning_rate", 3e-4),
            n_steps=train_cfg.get("n_steps", 2048),
            batch_size=train_cfg.get("batch_size", 256),
            n_epochs=train_cfg.get("n_epochs", 10),
            gamma=train_cfg.get("gamma", 0.99),
            gae_lambda=train_cfg.get("gae_lambda", 0.95),
            clip_range=train_cfg.get("clip_range", 0.2),
            ent_coef=train_cfg.get("ent_coef", 0.01),
            vf_coef=train_cfg.get("vf_coef", 0.5),
            max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
            tensorboard_log=run_dir if log_cfg.get("tensorboard", True) else None,
            verbose=1,
            device=device,
        )

    # Callbacks
    win_rate_cb = WinRateCallback(window=200)
    checkpoint_cb = CheckpointCallback(
        run_dir=run_dir,
        save_freq=log_cfg.get("checkpoint_freq", 100_000),
        keep=log_cfg.get("keep_checkpoints", 5),
        win_rate_cb=win_rate_cb,
        verbose=1,
    )

    callbacks = [win_rate_cb, checkpoint_cb]

    selfplay_path = os.path.join(run_dir, f"{SelfPlayCallback.SELFPLAY_FILENAME}.zip")

    if curriculum_cfg.get("enabled", True):
        def make_env_fn(stage: str, model_path: str = None):
            # Self-play stage uses the fixed file SelfPlayCallback overwrites.
            if stage == "selfplay" and model_path is None:
                model_path = selfplay_path
            env = build_vec_env(config, stage=stage, n_envs=n_envs, model_path=model_path)
            return VecMonitor(env, info_keywords=("win", "vp"))

        curriculum_cb = CurriculumCallback(
            curriculum=curriculum,
            win_rate_cb=win_rate_cb,
            make_env_fn=make_env_fn,
            run_dir=run_dir,
            eval_freq=curriculum_cfg.get("eval_freq", 50_000),
            verbose=1,
        )
        callbacks.append(curriculum_cb)

    # SelfPlayCallback runs whenever self-play is reachable: either as the
    # starting stage or as a later curriculum stage.
    stage_names = {s.name for s in curriculum.stages}
    if initial_stage == "selfplay" or "selfplay" in stage_names:
        if initial_stage == "selfplay":
            save_checkpoint(model, run_dir, SelfPlayCallback.SELFPLAY_FILENAME)
        callbacks.append(SelfPlayCallback(run_dir=run_dir, update_freq=50_000, verbose=1))

    # Train
    print(f"\nStarting training: {total_timesteps:,} timesteps, {n_envs} envs")
    print(f"Initial curriculum stage: {initial_stage}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=args.resume is None,
        progress_bar=True,
    )

    # Save final model
    final_path = save_checkpoint(model, run_dir, "final_model")
    print(f"\nTraining complete. Final model saved to {final_path}")

    vec_env.close()


if __name__ == "__main__":
    main()
