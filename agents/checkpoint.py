import os
import glob
import shutil
from pathlib import Path


def save_checkpoint(model, run_dir: str, name: str) -> str:
    """Save model to run_dir/name.zip. Returns the saved path."""
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, name)
    model.save(path)
    return path + ".zip"


def load_model(model_cls, path: str, env=None):
    """Load a saved model. env is optional for re-wrapping."""
    return model_cls.load(path, env=env)


def get_best_checkpoint(run_dir: str) -> str | None:
    """Return path to best_model.zip if it exists."""
    p = os.path.join(run_dir, "best_model.zip")
    return p if os.path.exists(p) else None


def prune_old_checkpoints(run_dir: str, keep: int = 5) -> None:
    """Delete oldest step_*.zip checkpoints, keeping the most recent `keep`."""
    pattern = os.path.join(run_dir, "step_*.zip")
    checkpoints = sorted(glob.glob(pattern), key=os.path.getmtime)
    for old in checkpoints[:-keep]:
        os.remove(old)
