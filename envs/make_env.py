import os
import warnings
import numpy as np
import gymnasium
from catanatron.models.player import Color, RandomPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE

from envs.rewards import make_reward_fn
from envs.wrappers import EpisodeStatsWrapper

# Colors assigned to opponents (agent is always BLUE)
_OPPONENT_COLORS = [Color.RED, Color.ORANGE, Color.WHITE]


def _get_opponent_colors(num_players: int):
    return _OPPONENT_COLORS[: num_players - 1]


def _build_opponents(stage: str, num_players: int, model_path: str = None):
    colors = _get_opponent_colors(num_players)
    if stage == "random":
        return [RandomPlayer(c) for c in colors]
    if stage == "weighted":
        return [WeightedRandomPlayer(c) for c in colors]
    if stage == "selfplay":
        if not model_path or not os.path.exists(model_path):
            warnings.warn(
                f"Self-play stage requested but model_path is missing or does not exist "
                f"({model_path!r}); falling back to WeightedRandomPlayer."
            )
            return [WeightedRandomPlayer(c) for c in colors]
        from agents.policy_player import PolicyPlayer
        return [PolicyPlayer(color=c, model_path=model_path) for c in colors]
    raise ValueError(f"Unknown curriculum stage: {stage!r}")


def action_mask_fn(env: gymnasium.Env) -> np.ndarray:
    """Build a boolean action mask from valid actions. Used by ActionMasker."""
    valid = env.unwrapped.get_valid_actions()
    mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
    mask[valid] = True
    return mask


def make_env(config: dict, rank: int = 0, model_path: str = None):
    """Create a single masked Catanatron environment.

    Args:
        config: Merged config dict (from YAML).
        rank: Env index in a VecEnv (used for seeding).
        model_path: Path to opponent model checkpoint (self-play stage).

    Returns:
        A wrapped gymnasium.Env ready for SB3 training.
    """
    from sb3_contrib.common.wrappers import ActionMasker

    env_cfg = config.get("env", {})
    reward_cfg = config.get("rewards", {})
    curriculum_cfg = config.get("curriculum", {})

    num_players = env_cfg.get("num_players", 4)
    stage = curriculum_cfg.get("current_stage", "weighted")

    reward_fn = make_reward_fn(
        name=env_cfg.get("reward_function", "shaped"),
        weights=reward_cfg if reward_cfg else None,
    )
    enemies = _build_opponents(stage, num_players, model_path=model_path)

    env = gymnasium.make(
        "catanatron-v1",
        config={
            "map_type": env_cfg.get("map_type", "BASE"),
            "vps_to_win": env_cfg.get("vps_to_win", 10),
            "representation": env_cfg.get("representation", "vector"),
            "enemies": enemies,
            "reward_function": reward_fn,
        },
    )
    env = EpisodeStatsWrapper(env)
    env = ActionMasker(env, action_mask_fn)
    return env
