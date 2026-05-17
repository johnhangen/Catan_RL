import gymnasium
from catanatron.state_functions import get_actual_victory_points
from envs.rewards import clear_reward_state


class EpisodeStatsWrapper(gymnasium.Wrapper):
    """Injects per-episode stats into info["episode"] on termination."""

    def __init__(self, env: gymnasium.Env):
        super().__init__(env)
        self._ep_reward = 0.0
        self._ep_length = 0
        self._last_game_id = None

    def reset(self, **kwargs):
        # Clear stale reward bookkeeping for the previous game before
        # the env builds a new one (avoids unbounded dict growth on truncation).
        if self._last_game_id is not None:
            clear_reward_state(self._last_game_id)
        self._ep_reward = 0.0
        self._ep_length = 0
        out = self.env.reset(**kwargs)
        self._last_game_id = id(self.env.unwrapped.game)
        return out

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
