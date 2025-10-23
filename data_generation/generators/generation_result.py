"""
Data structures for maintaining associations across the generation pipeline.

This module provides classes to maintain the full chain of associations:
persona → world_model → bottlenecks → checklists → true_positives → distractors
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .world_model import WorldModel
from .bottleneck_injector import Bottleneck
from configs import ProactiveChecklist, CorpusItem
from data_generation.data.linkedin_profile import LinkedInPersona


@dataclass
class BottleneckResult:
    """Results for a single bottleneck."""

    bottleneck: Bottleneck
    checklist: ProactiveChecklist
    true_positives: List[CorpusItem]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "bottleneck": self.bottleneck.model_dump(),
            "checklist": self.checklist.model_dump(),
            "true_positives": [tp.model_dump() for tp in self.true_positives],
        }


@dataclass
class PersonaResult:
    """Complete results for a single persona/world model."""

    persona: LinkedInPersona
    world_model: WorldModel
    bottleneck_results: List[BottleneckResult]
    distractors: List[CorpusItem]  # Shared across all bottlenecks for this persona

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "persona": self.persona.model_dump(),
            "world_model": self.world_model.model_dump(),
            "bottlenecks": [br.to_dict() for br in self.bottleneck_results],
            "distractors": [d.model_dump() for d in self.distractors],
            "metadata": {
                "num_bottlenecks": len(self.bottleneck_results),
                "num_distractors": len(self.distractors),
                "total_true_positives": sum(
                    len(br.true_positives) for br in self.bottleneck_results
                ),
            },
        }

    def to_json_per_bottleneck(self) -> List[Dict[str, Any]]:
        """
        Export as separate JSON objects per bottleneck.
        Each includes the full context (persona, world_model, distractors).
        """
        results = []
        for br in self.bottleneck_results:
            results.append(
                {
                    "persona": self.persona.model_dump(),
                    "world_model": self.world_model.model_dump(),
                    "bottleneck": br.bottleneck.model_dump(),
                    "checklist": br.checklist.model_dump(),
                    "true_positives": [tp.model_dump() for tp in br.true_positives],
                    "distractors": [
                        d.model_dump() for d in self.distractors
                    ],  # Same for all bottlenecks
                }
            )
        return results


@dataclass
class GenerationBatch:
    """Results for an entire batch of personas."""

    persona_results: List[PersonaResult]

    def to_dict(self) -> Dict[str, Any]:
        """Convert entire batch to dictionary."""
        return {
            "personas": [pr.to_dict() for pr in self.persona_results],
            "summary": {
                "num_personas": len(self.persona_results),
                "total_bottlenecks": sum(
                    len(pr.bottleneck_results) for pr in self.persona_results
                ),
                "total_distractors": sum(
                    len(pr.distractors) for pr in self.persona_results
                ),
                "total_true_positives": sum(
                    sum(len(br.true_positives) for br in pr.bottleneck_results)
                    for pr in self.persona_results
                ),
            },
        }

    def get_persona_by_index(self, idx: int) -> Optional[PersonaResult]:
        """Get results for a specific persona by index."""
        if 0 <= idx < len(self.persona_results):
            return self.persona_results[idx]
        return None

    def get_all_json_per_world_model(self) -> List[Dict[str, Any]]:
        """Export as one JSON object per world model."""
        return [pr.to_dict() for pr in self.persona_results]

    def get_all_json_per_bottleneck(self) -> List[Dict[str, Any]]:
        """Export as one JSON object per bottleneck (flattened)."""
        results = []
        for pr in self.persona_results:
            results.extend(pr.to_json_per_bottleneck())
        return results
