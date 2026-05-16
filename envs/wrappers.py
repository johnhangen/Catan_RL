import threading
from typing import Optional
import gymnasium
from catanatron.state_functions import get_actual_victory_points


class EpisodeStatsWrapper(gymnasium.Wrapper):
    """Injects per-episode stats into info["episode"] on termination."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_length = 0

    def reset(self, **kwargs):
        self._ep_reward = 0.0
        self._ep_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._ep_reward += reward
        self._ep_length += 1
        if terminated or truncated:
            u = self.env.unwrapped
            game = u.game
            p0_color = u.p0.color
            winner = game.winning_color()
            vp = get_actual_victory_points(game.state, p0_color)
            info["episode"] = {
                "r": self._ep_reward,
                "l": self._ep_length,
                "win": int(winner == p0_color),
                "vp": vp,
            }
            # Also at top-level so VecMonitor can copy them via info_keywords
            info["win"] = int(winner == p0_color)
            info["vp"] = vp
        return obs, reward, terminated, truncated, info


class SelfPlayOpponentWrapper(gymnasium.Wrapper):
    """Hot-swaps the opponent model between episodes during self-play.

    Call `update_opponent(model_path)` from a training callback to refresh
    opponents without rebuilding the VecEnv.
    """

    def __init__(self, env: gymnasium.Env, model_path: Optional[str] = None):
        super().__init__(env)
        self._model_path = model_path
        self._lock = threading.Lock()

    def update_opponent(self, model_path: str) -> None:
        with self._lock:
            self._model_path = model_path

    def reset(self, **kwargs):
        with self._lock:
            path = self._model_path
        if path is not None:
            from agents.policy_player import PolicyPlayer
            u = self.env.unwrapped
            p0_color = u.p0.color
            colors = [p.color for p in u.game.state.players if p.color != p0_color]
            new_enemies = [PolicyPlayer(color=c, model_path=path) for c in colors]
            # Rebuild the unwrapped env config before next reset
            u._enemies = new_enemies
        return self.env.reset(**kwargs)
