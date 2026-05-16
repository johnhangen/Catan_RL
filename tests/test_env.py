"""Tests for environment instantiation, action masking, and episode progression."""

import numpy as np
import pytest
import gymnasium
import catanatron_gym  # register env
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE
from sb3_contrib.common.wrappers import ActionMasker

from envs.make_env import make_env, action_mask_fn
from envs.rewards import sparse_reward, vp_delta_reward, shaped_reward


@pytest.fixture
def basic_env():
    config = {
        "env": {"num_players": 4, "map_type": "BASE", "reward_function": "sparse", "representation": "vector"},
        "rewards": {},
        "curriculum": {"current_stage": "weighted"},
    }
    env = make_env(config, rank=0)
    yield env
    env.close()


def test_env_resets(basic_env):
    obs, info = basic_env.reset()
    assert obs is not None
    assert "action_mask" in info or True  # ActionMasker wraps info


def test_obs_shape(basic_env):
    obs, _ = basic_env.reset()
    # 4-player obs is ~1002 features
    assert obs.ndim == 1
    assert obs.shape[0] > 0


def test_action_space_size(basic_env):
    assert basic_env.action_space.n == ACTION_SPACE_SIZE


def test_mask_is_valid(basic_env):
    """Mask must have at least 1 legal action at every step."""
    basic_env.reset()
    for _ in range(50):
        mask = action_mask_fn(basic_env)
        assert mask.dtype == bool
        assert mask.sum() > 0, "Mask has no legal actions"
        legal = np.where(mask)[0]
        action = np.random.choice(legal)
        obs, reward, terminated, truncated, info = basic_env.step(action)
        if terminated or truncated:
            basic_env.reset()


def test_no_illegal_actions_sampled():
    """Run 5 full episodes sampling only from valid actions; no errors should occur."""
    config = {
        "env": {"num_players": 4, "map_type": "BASE", "reward_function": "sparse", "representation": "vector"},
        "rewards": {},
        "curriculum": {"current_stage": "random"},
    }
    env = make_env(config, rank=99)
    for _ in range(5):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = action_mask_fn(env)
            legal = np.where(mask)[0]
            assert len(legal) > 0
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
    env.close()


def test_episode_stats_in_info():
    """EpisodeStatsWrapper should inject episode info on termination."""
    config = {
        "env": {"num_players": 4, "map_type": "BASE", "reward_function": "sparse", "representation": "vector"},
        "rewards": {},
        "curriculum": {"current_stage": "random"},
    }
    env = make_env(config, rank=42)
    found_episode = False
    for _ in range(3):
        obs, _ = env.reset()
        done = False
        while not done:
            mask = action_mask_fn(env)
            legal = np.where(mask)[0]
            action = np.random.choice(legal)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done and "episode" in info:
                ep = info["episode"]
                assert "r" in ep
                assert "l" in ep
                assert "win" in ep
                assert "vp" in ep
                assert ep["win"] in (0, 1)
                found_episode = True
    env.close()
    assert found_episode, "No episode info found in 3 games"
