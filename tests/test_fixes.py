"""Regression tests for bugs found during code review."""

import os
import pytest
import gymnasium
import numpy as np
import catanatron_gym

from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catanatron_gym.envs.catanatron_env import ACTION_SPACE_SIZE

from envs.make_env import make_env, action_mask_fn
from envs.rewards import _prev_vp, make_reward_fn
from envs.wrappers import EpisodeStatsWrapper


def _basic_config(reward_fn="vp_delta"):
    return {
        "env": {
            "num_players": 4,
            "map_type": "BASE",
            "reward_function": reward_fn,
            "representation": "vector",
        },
        "rewards": {},
        "curriculum": {"current_stage": "random"},
    }


def test_reward_state_clears_on_reset():
    """Memory leak fix: _prev_vp must shrink when env resets without a winner."""
    _prev_vp.clear()
    env = make_env(_basic_config("vp_delta"), rank=0)

    # Play 3 partial episodes
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        for _ in range(20):
            mask = action_mask_fn(env)
            action = int(np.random.choice(np.where(mask)[0]))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                break

    # After 3 partial episodes there should be at most 1 entry (current game).
    # Without the fix, this would be 3+ (one per game id).
    assert len(_prev_vp) <= 1, f"Memory leak: _prev_vp has {len(_prev_vp)} entries"
    env.close()


def test_unknown_stage_raises():
    """_build_opponents should fail loudly on unknown stages."""
    from envs.make_env import _build_opponents
    with pytest.raises(ValueError):
        _build_opponents("not_a_stage", 4)


def test_selfplay_without_model_warns_and_falls_back():
    """If selfplay stage requested but no model file exists, fall back with warning."""
    from envs.make_env import _build_opponents
    with pytest.warns(UserWarning):
        opponents = _build_opponents("selfplay", 4, model_path="/nonexistent/path.zip")
    assert len(opponents) == 3
    # Should fall back to WeightedRandomPlayer
    assert all(isinstance(o, WeightedRandomPlayer) for o in opponents)


def test_human_relay_player_auto_plays_single_non_roll_action():
    """HumanRelayPlayer should take the only legal action without prompting,
    except ROLL which always prompts for the real dice."""
    from agents.human_relay_player import HumanRelayPlayer
    from catanatron.models.enums import Action, ActionType

    player = HumanRelayPlayer(Color.RED)
    only_action = Action(Color.RED, ActionType.END_TURN, None)
    chosen = player.decide(game=None, playable_actions=[only_action])
    assert chosen == only_action


def test_human_relay_player_prompts_for_dice_on_roll(monkeypatch):
    """ROLL action should always trigger dice prompt and inject (d1, d2) into value."""
    from agents.human_relay_player import HumanRelayPlayer
    import agents.human_relay_player as hrp
    from catanatron.models.enums import Action, ActionType

    monkeypatch.setattr(hrp, "_prompt_dice", lambda: (3, 4))
    player = HumanRelayPlayer(Color.RED)
    roll = Action(Color.RED, ActionType.ROLL, None)
    chosen = player.decide(game=None, playable_actions=[roll])
    assert chosen.action_type == ActionType.ROLL
    assert chosen.value == (3, 4)


def test_policy_player_clears_model_on_reset():
    """PolicyPlayer must drop its cached model on reset_state so self-play
    opponents pick up new weights from disk."""
    from agents.policy_player import PolicyPlayer
    p = PolicyPlayer(color=Color.RED, model_path="/fake/path.zip")
    # Simulate a loaded model
    p._model = "fake_model_obj"
    p.reset_state()
    assert p._model is None, "reset_state() did not clear cached model"


def test_policy_player_no_reload_when_disabled():
    """When reload_on_reset=False, cached model is preserved."""
    from agents.policy_player import PolicyPlayer
    p = PolicyPlayer(color=Color.RED, model_path="/fake/path.zip", reload_on_reset=False)
    p._model = "fake_model_obj"
    p.reset_state()
    assert p._model == "fake_model_obj"
