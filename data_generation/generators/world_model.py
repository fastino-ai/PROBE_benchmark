"""
World Model Generator (PR9).

This module generates comprehensive context from LinkedIn personas, including:
- Relationships (colleagues, clients, stakeholders)
- Personal context (work style, preferences, constraints)
- Available actions (actions persona can take)
- Organizational structure (company hierarchy and processes)

The world model provides the contextual foundation for bottleneck injection
and evaluation of proactive AI agents.
"""

import json
import logging
import random
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from .bottleneck_injector import Bottleneck

from concurrent.futures import ThreadPoolExecutor, as_completed
from pydantic import BaseModel, Field, field_validator

from .base import BaseGenerator

from configs.actions import ProactiveAction
from data_generation.data.linkedin_profile import LinkedInPersona

logger = logging.getLogger(__name__)


class ContextDifficulty(str, Enum):
    """Difficulty levels for context complexity."""

    EASY = "easy"  # Minimal context to track
    MEDIUM = "medium"  # Moderate context complexity
    HARD = "hard"  # High context complexity


class DistractorDifficulty(str, Enum):
    """Difficulty levels for distractor content quality and sophistication."""

    EASY = "easy"  # Short, simple distractors (100-200 words)
    MEDIUM = "medium"  # Moderate distractors with some detail (300-400 words)
    HARD = "hard"  # Long, sophisticated distractors (500-700 words)


class RelationshipType(str, Enum):
    """Types of professional relationships."""

    COLLEAGUE = "colleague"
    MANAGER = "manager"
    DIRECT_REPORT = "direct_report"
    CLIENT = "client"
    STAKEHOLDER = "stakeholder"
    VENDOR = "vendor"
    MENTOR = "mentor"
    MENTEE = "mentee"
    CROSS_FUNCTIONAL_PARTNER = "cross_functional_partner"


class Relationship(BaseModel):
    """Represents a professional relationship."""

    name: str = Field(..., description="Name of the person")
    type: RelationshipType = Field(..., description="Type of relationship")
    department: Optional[str] = Field(None, description="Department they work in")
    relationship_context: str = Field(
        ..., description="One sentence description of the state of the relationship"
    )

    @field_validator("name", "relationship_context")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Ensure fields are non-empty."""
        if not v.strip():
            raise ValueError("Must not be empty")
        return v.strip()


class PersonalContext(BaseModel):
    """Simplified personal work context and preferences."""

    work_style: str = Field(..., description="Brief description of work style")
    communication_preferences: List[str] = Field(
        default_factory=list, description="Preferred communication methods"
    )
    current_priorities: List[str] = Field(
        default_factory=list, description="Current top priorities"
    )
    pain_points: List[str] = Field(
        default_factory=list, description="Current challenges or pain points"
    )


class OrgStructure(BaseModel):
    """Simplified organizational structure."""

    company_name: str = Field(..., description="Name of the company")
    department: str = Field(..., description="Department name")
    team_size: int = Field(..., description="Size of immediate team")
    reporting_to: str = Field(..., description="Title of their manager")
    key_meetings: List[str] = Field(
        default_factory=list, description="Regular important meetings"
    )


class WorldModel(BaseModel):
    """Complete world model for a persona."""

    persona_id: str = Field(..., description="Identifier for the source persona")
    persona_full_name: str = Field(..., description="Full name of the persona")
    persona_occupation: str = Field(..., description="Occupation of the persona")
    persona_about: str = Field(..., description="About section of the persona")
    relationships: List[Relationship] = Field(
        default_factory=list, description="Professional relationships"
    )
    personal_context: PersonalContext = Field(..., description="Personal work context")
    available_actions: List[ProactiveAction] = Field(
        default_factory=list, description="Actions the persona can take"
    )
    organizational_structure: OrgStructure = Field(
        ..., description="Company structure and processes"
    )
    context_difficulty: ContextDifficulty = Field(
        ..., description="Complexity level of this world model"
    )

    def get_relationship_count_by_type(self) -> Dict[RelationshipType, int]:
        """Get count of relationships by type."""
        counts = {rel_type: 0 for rel_type in RelationshipType}
        for rel in self.relationships:
            counts[rel.type] += 1
        return counts


class WorldModelGenerator(BaseGenerator):
    """Generates world models from LinkedIn personas using LLM."""

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        llm_generate_func: Optional[Callable[[str], str]] = None,
        debug_dir: Optional[Path] = None,
        max_workers: int = 8,
    ):
        """Initialize the world model generator.

        Args:
            template_dir: Directory containing Jinja2 templates
            llm_generate_func: Optional custom LLM function for making calls (from MultiLLMClient)
            debug_dir: Optional directory for debug output
            max_workers: Maximum number of parallel workers for batch processing
        """
        super().__init__(
            template_dir=template_dir,
            template_subdir="world_model",
            llm_generate_func=llm_generate_func,
            debug_dir=debug_dir,
            max_workers=max_workers,
        )

    def generate(
        self,
        personas: List[LinkedInPersona],
        difficulty: ContextDifficulty,
        available_actions: Optional[List[List[ProactiveAction]]] = None,
        **kwargs,
    ) -> List[WorldModel]:
        """Generate world models for ALL personas in parallel.

        Args:
            personas: List of LinkedIn personas to generate world models for
            difficulty: Context difficulty level
            available_actions: Optional list of lists of available proactive actions (one list per persona)
            **kwargs: Additional arguments (for interface compatibility)

        Returns:
            List of WorldModel instances (1:1 with personas)
        """
        logger.info(
            f"Generating world models for {len(personas)} personas with difficulty={difficulty}"
        )

        # Process ALL personas in parallel
        world_models = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all persona processing tasks
            futures_to_idx = {
                executor.submit(
                    self._generate_single_world_model,
                    idx,
                    persona,
                    difficulty,
                    available_actions[idx] if available_actions else None,
                ): idx
                for idx, persona in enumerate(personas)
            }

            # Collect results in order
            results = {}
            for future in as_completed(futures_to_idx):
                idx = futures_to_idx[future]
                try:
                    results[idx] = future.result()
                    logger.info(
                        f"Generated world model for persona {idx + 1}/{len(personas)}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate world model for persona {idx}: {e}"
                    )
                    raise

            # Sort by index to maintain order
            world_models = [results[i] for i in range(len(personas))]

        logger.info(f"✓ Generated {len(world_models)} world models successfully")

        return world_models

    def _generate_single_world_model(
        self,
        idx: int,
        persona: LinkedInPersona,
        difficulty: ContextDifficulty,
        persona_actions: Optional[List[ProactiveAction]] = None,
    ) -> WorldModel:
        """Generate a world model for a single persona."""
        # Generate all components sequentially (we're already in a thread)
        relationships = self._generate_relationships(persona, difficulty)
        personal_context = self._generate_personal_context(persona, difficulty)
        org_structure = self._generate_organizational_structure(persona, difficulty)

        # Actions are generated later after bottlenecks are created
        # They should be provided via persona_actions parameter or set to empty list
        if persona_actions is None:
            persona_actions = []

        # Create persona ID from name
        persona_id = persona.name.lower().replace(" ", "_").replace(",", "")

        return WorldModel(
            persona_id=persona_id,
            persona_full_name=persona.name,
            persona_occupation=persona.occupation,
            persona_about=persona.about,
            relationships=relationships,
            personal_context=personal_context,
            available_actions=persona_actions,
            organizational_structure=org_structure,
            context_difficulty=difficulty,
        )

    def _generate_relationships(
        self, persona: LinkedInPersona, difficulty: ContextDifficulty
    ) -> List[Relationship]:
        """Generate relationships based on persona and difficulty."""
        template = self.jinja_env.get_template("generate_relationships.j2")

        # Determine number of relationships based on difficulty
        relationship_counts = {
            ContextDifficulty.EASY: {"min": 3, "max": 5},
            ContextDifficulty.MEDIUM: {"min": 8, "max": 12},
            ContextDifficulty.HARD: {"min": 15, "max": 20},
        }

        prompt = template.render(
            persona=persona,
            difficulty=difficulty.value,
            min_relationships=relationship_counts[difficulty]["min"],
            max_relationships=relationship_counts[difficulty]["max"],
        )

        response = self.llm_generate_func(prompt)
        response_data = self._parse_json_response(response, "relationships")
        return self._parse_relationships(response_data)

    def _generate_personal_context(
        self, persona: LinkedInPersona, difficulty: ContextDifficulty
    ) -> PersonalContext:
        """Generate personal context based on persona."""
        template = self.jinja_env.get_template("generate_personal_context.j2")

        prompt = template.render(
            persona=persona,
            difficulty=difficulty.value,
        )

        response = self.llm_generate_func(prompt)
        response_data = self._parse_json_response(response, "personal_context")
        return PersonalContext(**response_data)

    def _generate_organizational_structure(
        self, persona: LinkedInPersona, difficulty: ContextDifficulty
    ) -> OrgStructure:
        """Generate organizational structure based on persona."""
        template = self.jinja_env.get_template("generate_org_structure.j2")

        # Team size based on difficulty
        team_sizes = {
            ContextDifficulty.EASY: {"min": 3, "max": 5},
            ContextDifficulty.MEDIUM: {"min": 5, "max": 10},
            ContextDifficulty.HARD: {"min": 10, "max": 20},
        }

        prompt = template.render(
            persona=persona,
            difficulty=difficulty.value,
            min_team_size=team_sizes[difficulty]["min"],
            max_team_size=team_sizes[difficulty]["max"],
        )

        response = self.llm_generate_func(prompt)
        response_data = self._parse_json_response(response, "organizational_structure")
        return OrgStructure(**response_data)

    def _generate_default_actions(
        self,
        persona: LinkedInPersona,
        org_structure: OrgStructure,
        difficulty: ContextDifficulty,
    ) -> List[ProactiveAction]:
        """Generate default proactive actions if none provided."""
        template = self.jinja_env.get_template("generate_actions_for_bottleneck.j2")

        # Number of actions based on difficulty
        action_counts = {
            ContextDifficulty.EASY: 5,
            ContextDifficulty.MEDIUM: 12,
            ContextDifficulty.HARD: 25,
        }

        prompt = template.render(
            persona=persona,
            org_structure=org_structure,
            bottlenecks=[],  # Empty bottlenecks list - just generate general actions
            difficulty=difficulty.value,
            num_actions=action_counts[difficulty],
        )

        response = self.llm_generate_func(prompt)
        response_data = self._parse_json_response(response, "actions")
        return self._parse_actions(response_data)

    def generate_actions_for_bottlenecks(
        self,
        persona: LinkedInPersona,
        org_structure: OrgStructure,
        bottlenecks: List["Bottleneck"],
        difficulty: ContextDifficulty,
    ) -> List[ProactiveAction]:
        """Generate actions where exactly one action solves each bottleneck.

        Args:
            persona: The LinkedIn persona
            org_structure: Organization structure information
            bottlenecks: List of bottlenecks that need solutions
            difficulty: Context difficulty level

        Returns:
            List of ProactiveAction objects, where exactly one action solves each bottleneck
        """
        template = self.jinja_env.get_template("generate_actions_for_bottleneck.j2")

        # Calculate total number of actions based on difficulty
        # Ensure we have at least one action per bottleneck plus some extras
        action_counts = {
            ContextDifficulty.EASY: max(5, len(bottlenecks) + 2),
            ContextDifficulty.MEDIUM: max(12, len(bottlenecks) + 5),
            ContextDifficulty.HARD: max(25, len(bottlenecks) + 10),
        }

        prompt = template.render(
            persona=persona,
            org_structure=org_structure,
            bottlenecks=bottlenecks,
            difficulty=difficulty.value,
            num_actions=action_counts[difficulty],
        )

        response = self.llm_generate_func(prompt)
        response_data = self._parse_json_response(response, "actions_for_bottlenecks")

        # Parse actions and preserve the solves_bottleneck field for ground truth tracking
        actions_data = response_data.get("actions", [])
        actions = []

        for action_data in actions_data:
            try:
                action = ProactiveAction(**action_data)
                actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to parse action: {e}")
                continue

        logger.info(
            f"Generated {len(actions)} actions for {len(bottlenecks)} bottlenecks"
        )

        # Shuffle the actions to randomize their order
        random.shuffle(actions)

        return actions

    def _parse_json_response(self, response: str, response_type: str) -> Dict[str, Any]:
        """Parse JSON response from LLM with error handling."""

        # Clean up response - sometimes there are extra characters
        content = response.replace("```json", "").replace("```", "").strip()

        # Extract JSON if content doesn't start with '{'
        if not content.startswith("{"):
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                content = content[start_idx:end_idx]

        # Handle double braces from template rendering ({{...}}) early
        if content.startswith("{{") and content.endswith("}}"):
            content = content[1:-1].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed for {response_type}: {e}")

            # Try to extract JSON from the content more aggressively
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1

            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                json_content = content[start_idx:end_idx]

                # Try again with double brace handling
                if json_content.startswith("{{") and json_content.endswith("}}"):
                    json_content = json_content[1:-1].strip()

                try:
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass

            raise ValueError(
                f"Could not parse JSON for {response_type}: {e}. Content preview: {content[:200]}..."
            )

    def _parse_relationships(self, response: Dict[str, Any]) -> List[Relationship]:
        """Parse relationships from LLM response."""
        relationships = []

        for rel_data in response.get("relationships", []):
            try:
                # Convert string type to enum if needed
                if isinstance(rel_data.get("type"), str):
                    rel_data["type"] = RelationshipType(rel_data["type"])

                relationship = Relationship(**rel_data)
                relationships.append(relationship)
            except Exception as e:
                logger.warning(f"Failed to parse relationship: {e}")
                continue

        return relationships

    def _parse_actions(self, response: Dict[str, Any]) -> List[ProactiveAction]:
        """Parse proactive actions from LLM response."""
        actions = []

        for action_data in response.get("actions", []):
            try:
                action = ProactiveAction(**action_data)
                actions.append(action)
            except Exception as e:
                logger.warning(f"Failed to parse action: {e}")
                continue

        # Shuffle the actions to randomize their order
        random.shuffle(actions)

        return actions
