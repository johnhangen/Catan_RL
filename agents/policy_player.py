import numpy as np
from catanatron.models.player import Player, Color
from catanatron_gym.envs.catanatron_env import from_action_space, to_action_space, ACTION_SPACE_SIZE
from catanatron_gym.features import create_sample, get_feature_ordering


class PolicyPlayer(Player):
    """A trained SB3 MaskablePPO model acting as a Catanatron opponent.

    Used during self-play: pass instances as `config["enemies"]`.
    The model is loaded lazily on first call so subprocess VecEnvs can
    pickle this object before the model file exists.
    """

    def __init__(self, color: Color, model_path: str, deterministic: bool = True):
        super().__init__(color, is_bot=True)
        self.model_path = model_path
        self.deterministic = deterministic
        self._model = None
        self._features = None

    def _load(self, num_players: int = 4, map_type: str = "BASE"):
        if self._model is None:
            from sb3_contrib.ppo_mask import MaskablePPO
            self._model = MaskablePPO.load(self.model_path)
        if self._features is None:
            self._features = get_feature_ordering(num_players, map_type)

    def decide(self, game, playable_actions):
        num_players = len(game.state.players)
        self._load(num_players=num_players)

        # Build observation from this player's perspective
        sample = create_sample(game, self.color)
        obs = np.array([float(sample[f]) for f in self._features], dtype=np.float64)

        # Build action mask
        valid_ints = list(map(to_action_space, playable_actions))
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        mask[valid_ints] = True

        action_int, _ = self._model.predict(
            obs, action_masks=mask, deterministic=self.deterministic
        )
        return from_action_space(int(action_int), playable_actions)

    def reset_state(self):
        pass
