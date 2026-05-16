import os
import time
from typing import Optional
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback as SB3EvalCallback
from stable_baselines3.common.vec_env import VecEnv

from training.curriculum import CurriculumManager
from agents.checkpoint import save_checkpoint, prune_old_checkpoints


class WinRateCallback(BaseCallback):
    """Tracks win rate over recent episodes from EpisodeStatsWrapper info."""

    def __init__(self, window: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self.window = window
        self._wins = []
        self._vps = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._wins.append(ep["win"])
                self._vps.append(ep["vp"])
                if len(self._wins) > self.window:
                    self._wins.pop(0)
                    self._vps.pop(0)
        return True

    @property
    def win_rate(self) -> float:
        return float(np.mean(self._wins)) if self._wins else 0.0

    @property
    def mean_vp(self) -> float:
        return float(np.mean(self._vps)) if self._vps else 0.0


class CheckpointCallback(BaseCallback):
    """Saves model every `save_freq` steps and prunes old checkpoints."""

    def __init__(self, run_dir: str, save_freq: int, keep: int = 5, verbose: int = 0):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.save_freq = save_freq
        self.keep = keep
        self._last_save = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            name = f"step_{self.num_timesteps}"
            path = save_checkpoint(self.model, self.run_dir, name)
            prune_old_checkpoints(self.run_dir, keep=self.keep)
            self._last_save = self.num_timesteps
            if self.verbose:
                print(f"[Checkpoint] Saved {path}")
        return True


class CurriculumCallback(BaseCallback):
    """Advances curriculum stages when graduation criteria are met.

    After graduation, rebuilds the training VecEnv with the new opponent pool
    by calling `make_env_fn(stage_name)` and swapping `self.model.set_env`.
    """

    def __init__(
        self,
        curriculum: CurriculumManager,
        win_rate_cb: WinRateCallback,
        make_env_fn,
        run_dir: str,
        eval_freq: int = 50_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.win_rate_cb = win_rate_cb
        self.make_env_fn = make_env_fn
        self.run_dir = run_dir
        self.eval_freq = eval_freq
        self._last_eval = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval < self.eval_freq:
            return True
        self._last_eval = self.num_timesteps

        win_rate = self.win_rate_cb.win_rate
        mean_vp = self.win_rate_cb.mean_vp
        stage = self.curriculum.current_stage.name

        if self.verbose:
            print(
                f"[Curriculum] Step {self.num_timesteps:,} | Stage: {stage} | "
                f"Win rate: {win_rate:.1%} | Mean VP: {mean_vp:.1f}"
            )

        self.logger.record("curriculum/win_rate", win_rate)
        self.logger.record("curriculum/mean_vp", mean_vp)
        self.logger.record("curriculum/stage_index", self.curriculum.current_index)

        if self.curriculum.check_graduation(win_rate):
            new_stage = self.curriculum.advance()
            if self.verbose:
                print(f"[Curriculum] Advancing to stage: {new_stage.name}")
            # Save checkpoint before env swap
            save_checkpoint(self.model, self.run_dir, f"stage_{new_stage.name}_start")
            # Rebuild env with new opponents
            model_path = os.path.join(self.run_dir, f"stage_{new_stage.name}_start.zip")
            new_env = self.make_env_fn(new_stage.name, model_path=model_path)
            self.model.set_env(new_env)

        return True


class SelfPlayCallback(BaseCallback):
    """After each checkpoint, updates self-play opponents with the latest model.

    Works by calling `update_opponent(path)` on each env wrapper.
    """

    def __init__(self, run_dir: str, update_freq: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.run_dir = run_dir
        self.update_freq = update_freq
        self._last_update = 0
        self._latest_path: Optional[str] = None

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_update < self.update_freq:
            return True
        self._last_update = self.num_timesteps

        path = save_checkpoint(self.model, self.run_dir, f"selfplay_{self.num_timesteps}")
        self._latest_path = path

        # Push the new model path to every env wrapper
        env = self.model.get_env()
        if hasattr(env, "env_method"):
            try:
                env.env_method("update_opponent", path)
            except Exception:
                pass

        if self.verbose:
            print(f"[SelfPlay] Updated opponent to {path}")
        return True
