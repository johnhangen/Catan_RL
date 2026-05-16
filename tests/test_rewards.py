"""Tests for reward functions."""

import numpy as np
import pytest
import gymnasium
import catanatron_gym
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE

from envs.rewards import sparse_reward, vp_delta_reward, shaped_reward, make_reward_fn
from envs.make_env import action_mask_fn


def _run_one_game(reward_fn_name: str):
    """Run a game to completion with the given reward fn, collect rewards."""
    env = gymnasium.make("catanatron-v1", config={
        "map_type": "BASE",
        "vps_to_win": 10,
        "representation": "vector",
        "enemies": [WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.ORANGE), WeightedRandomPlayer(Color.WHITE)],
        "reward_function": make_reward_fn(reward_fn_name),
    })
    obs, info = env.reset(seed=42)
    rewards = []
    done = False
    u = env.unwrapped
    while not done:
        valid = u.get_valid_actions()
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[valid] = True
        action = np.random.choice(np.where(mask)[0])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        done = terminated or truncated
    env.close()
    return rewards


@pytest.mark.parametrize("name", ["sparse", "vp_delta", "shaped"])
def test_reward_fn_runs(name):
    rewards = _run_one_game(name)
    assert len(rewards) > 0
    assert all(isinstance(r, (int, float)) for r in rewards)


def test_sparse_reward_range():
    rewards = _run_one_game("sparse")
    for r in rewards:
        assert r in (-1.0, 0.0, 1.0), f"Sparse reward out of range: {r}"


def test_vp_delta_terminal_reward():
    """Terminal reward must be +10 (win) or -2 (loss)."""
    rewards = _run_one_game("vp_delta")
    assert rewards[-1] in (10.0, -2.0), f"Last reward was {rewards[-1]}"


def test_shaped_terminal_reward():
    rewards = _run_one_game("shaped")
    assert rewards[-1] in (10.0, -2.0), f"Last reward was {rewards[-1]}"


def test_make_reward_fn_unknown():
    with pytest.raises(ValueError):
        make_reward_fn("nonsense")


def test_make_reward_fn_returns_callable():
    for name in ("sparse", "vp_delta", "shaped"):
        fn = make_reward_fn(name)
        assert callable(fn)
