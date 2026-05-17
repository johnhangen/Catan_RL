from typing import Dict, Optional, Tuple
from catanatron.state_functions import (
    get_actual_victory_points,
    get_longest_road_color,
    get_largest_army,
)

# Per-game memoized previous state (keyed by game id)
_prev_vp: Dict[int, int] = {}
_prev_has_road: Dict[int, bool] = {}
_prev_has_army: Dict[int, bool] = {}


def _clear(game_id: int) -> None:
    _prev_vp.pop(game_id, None)
    _prev_has_road.pop(game_id, None)
    _prev_has_army.pop(game_id, None)


def clear_reward_state(game_id: int) -> None:
    """Public helper for wrappers to drop bookkeeping for a finished/abandoned game."""
    _clear(game_id)


def sparse_reward(game, p0_color) -> float:
    """Win=+1, Loss=-1, ongoing=0."""
    winner = game.winning_color()
    if winner is None:
        return 0.0
    return 1.0 if winner == p0_color else -1.0


def vp_delta_reward(game, p0_color) -> float:
    """Dense: VP gained this step + win/loss terminal bonus."""
    state = game.state
    current_vp = get_actual_victory_points(state, p0_color)
    key = id(game)
    prev_vp = _prev_vp.get(key, 0)
    _prev_vp[key] = current_vp

    winner = game.winning_color()
    if winner is not None:
        _clear(key)
        return 10.0 if winner == p0_color else -2.0

    return float(current_vp - prev_vp)


def shaped_reward(game, p0_color, weights: Optional[dict] = None) -> float:
    """VP delta + road/army bonuses + turn penalty."""
    w = weights or {
        "win": 10.0,
        "loss": -2.0,
        "vp_delta_scale": 1.0,
        "longest_road": 0.5,
        "largest_army": 0.5,
        "turn_penalty": -0.001,
    }

    state = game.state
    current_vp = get_actual_victory_points(state, p0_color)
    road_color = get_longest_road_color(state)
    army_color, _ = get_largest_army(state)
    has_road = road_color == p0_color
    has_army = army_color == p0_color

    key = id(game)
    prev_vp = _prev_vp.get(key, 0)
    prev_road = _prev_has_road.get(key, False)
    prev_army = _prev_has_army.get(key, False)

    _prev_vp[key] = current_vp
    _prev_has_road[key] = has_road
    _prev_has_army[key] = has_army

    winner = game.winning_color()
    if winner is not None:
        _clear(key)
        return w["win"] if winner == p0_color else w["loss"]

    reward = w["turn_penalty"]
    reward += w["vp_delta_scale"] * (current_vp - prev_vp)
    if has_road and not prev_road:
        reward += w["longest_road"]
    elif not has_road and prev_road:
        reward -= w["longest_road"]
    if has_army and not prev_army:
        reward += w["largest_army"]
    elif not has_army and prev_army:
        reward -= w["largest_army"]

    return reward


def make_reward_fn(name: str, weights: Optional[dict] = None):
    """Return a reward function compatible with catanatron-gym config."""
    if name == "sparse":
        return sparse_reward
    if name == "vp_delta":
        return vp_delta_reward
    if name == "shaped":
        w = weights
        def _shaped(game, p0_color):
            return shaped_reward(game, p0_color, weights=w)
        return _shaped
    raise ValueError(f"Unknown reward function: {name!r}. Choose: sparse, vp_delta, shaped")


REWARD_FUNCTIONS = {
    "sparse": sparse_reward,
    "vp_delta": vp_delta_reward,
    "shaped": shaped_reward,
}
