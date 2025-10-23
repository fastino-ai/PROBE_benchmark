"""
Shared world model interface for PersonaSim generators and evaluators.

Minimal context elements used across modules (e.g., distractors, world-model
generation, evaluation). Extend as needed while preserving backward compat.
"""

from __future__ import annotations

from typing import Any, Dict, List
from pydantic import BaseModel, Field


class WorldModel(BaseModel):
    participants: List[str] = Field(default_factory=list)
    teams: List[str] = Field(default_factory=list)
    locations: List[str] = Field(default_factory=list)
    templates: Dict[str, Any] = Field(default_factory=dict)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    other_info: Dict[str, Any] = Field(default_factory=dict)


__all__ = ["WorldModel"]
