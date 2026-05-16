from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage:
    name: str
    timesteps: int
    graduate_winrate: Optional[float]  # None = final stage, never graduates


STAGES = [
    Stage(name="random",   timesteps=500_000,   graduate_winrate=0.60),
    Stage(name="weighted", timesteps=1_000_000, graduate_winrate=0.55),
    Stage(name="selfplay", timesteps=2_000_000, graduate_winrate=None),
]


class CurriculumManager:
    """Tracks curriculum progress and decides when to advance stages."""

    def __init__(self, stages: list[Stage] = None):
        self.stages = stages or STAGES
        self.current_index = 0
        self._timesteps_in_stage = 0

    @property
    def current_stage(self) -> Stage:
        return self.stages[self.current_index]

    @property
    def is_final_stage(self) -> bool:
        return self.current_index >= len(self.stages) - 1

    def record_timesteps(self, n: int) -> None:
        self._timesteps_in_stage += n

    def check_graduation(self, win_rate: float) -> bool:
        """Return True if the agent should advance to the next stage."""
        if self.is_final_stage:
            return False
        stage = self.current_stage
        threshold = stage.graduate_winrate
        if threshold is None:
            return False
        over_budget = self._timesteps_in_stage >= stage.timesteps
        good_enough = win_rate >= threshold
        return good_enough or over_budget

    def advance(self) -> Stage:
        """Advance to the next stage. Returns the new stage."""
        if not self.is_final_stage:
            self.current_index += 1
            self._timesteps_in_stage = 0
        return self.current_stage

    def build_from_config(self, cfg: dict) -> None:
        """Overwrite stages from a YAML curriculum config."""
        raw = cfg.get("stages", [])
        if raw:
            self.stages = [
                Stage(
                    name=s["name"],
                    timesteps=int(s["timesteps"]),
                    graduate_winrate=s.get("graduate_winrate"),
                )
                for s in raw
            ]
