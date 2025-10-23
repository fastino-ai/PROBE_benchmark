"""Utility functions for difficulty mapping."""

from data_generation.generators import (
    ContextDifficulty,
    DistractorDifficulty,
)
from configs.config_schema import DifficultyLevel


def map_to_context_difficulty(level: DifficultyLevel) -> ContextDifficulty:
    """Map general difficulty level to context-specific difficulty."""
    mapping = {
        DifficultyLevel.EASY: ContextDifficulty.EASY,
        DifficultyLevel.MEDIUM: ContextDifficulty.MEDIUM,
        DifficultyLevel.HARD: ContextDifficulty.HARD,
        DifficultyLevel.PRODUCTION: ContextDifficulty.HARD,
    }
    return mapping[level]


def map_to_distractor_difficulty(level: DifficultyLevel) -> DistractorDifficulty:
    """Map general difficulty level to distractor-specific difficulty."""
    mapping = {
        DifficultyLevel.EASY: DistractorDifficulty.EASY,
        DifficultyLevel.MEDIUM: DistractorDifficulty.MEDIUM,
        DifficultyLevel.HARD: DistractorDifficulty.HARD,
        DifficultyLevel.PRODUCTION: DistractorDifficulty.HARD,
    }
    return mapping[level]
