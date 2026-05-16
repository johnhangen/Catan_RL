"""Tests for the curriculum manager."""

import pytest
from training.curriculum import CurriculumManager, Stage, STAGES


def test_initial_stage():
    cm = CurriculumManager()
    assert cm.current_stage.name == "random"
    assert cm.current_index == 0


def test_graduation_on_winrate():
    cm = CurriculumManager()
    # Not yet graduated with 0% win rate
    assert not cm.check_graduation(0.0)
    # Graduated with 60% win rate (threshold is 0.60)
    assert cm.check_graduation(0.60)


def test_advance_stage():
    cm = CurriculumManager()
    assert cm.current_stage.name == "random"
    new = cm.advance()
    assert new.name == "weighted"
    assert cm.current_index == 1


def test_no_graduation_from_final_stage():
    cm = CurriculumManager()
    # Fast-forward to selfplay
    cm.current_index = len(cm.stages) - 1
    assert cm.is_final_stage
    assert not cm.check_graduation(1.0)


def test_graduation_on_budget_exceeded():
    cm = CurriculumManager()
    budget = cm.current_stage.timesteps
    cm.record_timesteps(budget + 1)
    # Even with low win rate, budget exceeded should trigger graduation
    assert cm.check_graduation(0.0)


def test_advance_does_not_exceed_stages():
    cm = CurriculumManager()
    for _ in range(100):
        if cm.is_final_stage:
            break
        cm.advance()
    assert cm.current_index == len(cm.stages) - 1


def test_build_from_config():
    cm = CurriculumManager()
    cfg = {
        "stages": [
            {"name": "easy", "timesteps": 100_000, "graduate_winrate": 0.5},
            {"name": "hard", "timesteps": 500_000, "graduate_winrate": None},
        ]
    }
    cm.build_from_config(cfg)
    assert len(cm.stages) == 2
    assert cm.stages[0].name == "easy"
    assert cm.stages[1].graduate_winrate is None
