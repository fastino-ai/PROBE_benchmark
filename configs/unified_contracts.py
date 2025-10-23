"""
Unified contracts for proactivity benchmark data structures.

This module defines Pydantic models for the unified data format that can be used
by both agent-based generators and RAG generators, as well as the evaluation stack.
The evaluation is based on checklist satisfaction.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, field_serializer
from enum import Enum

# Import at runtime for Pydantic model building
try:
    from data_generation.generators.interfaces.artifacts import PayloadType
except ImportError:
    # Fallback type annotation if module not available
    PayloadType = Any  # type: ignore


class CorpusItemType(str, Enum):
    """Types of items in the corpus."""

    EMAIL = "email"
    EVENT = "event"
    DOCUMENT = "document"


class ChecklistStepType(str, Enum):
    """Types of steps in the proactive workflow checklist."""

    QUERY = "query"
    RETRIEVAL = "retrieval"
    IDENTIFICATION = "identification"
    SOLUTION = "solution"
    TASK_EXECUTION = "task_execution"


class CorpusItem(BaseModel):
    """
    Unified representation of a corpus item (email, event, or document).

    Uses typed payloads defined in generators/interfaces/artifacts.py to ensure
    consistent shapes for all artifact types.
    """

    id: str = Field(..., description="Unique identifier for the item")
    type: CorpusItemType = Field(..., description="Type of the corpus item")
    payload: PayloadType = Field(..., description="Typed payload for this item")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional meta flags such as distractor, bottleneck_evidence, technique, etc.",
    )

    # Proactivity benchmark metadata
    created_at: Optional[datetime] = Field(
        None, description="When this item was created"
    )
    updated_at: Optional[datetime] = Field(
        None, description="When this item was last updated"
    )
    tags: List[str] = Field(default_factory=list, description="Tags for categorization")

    @field_serializer("created_at", "updated_at")
    def serialize_datetime(self, value: Optional[datetime]) -> Optional[str]:
        """Serialize datetime fields to ISO format strings."""
        return value.isoformat() if value else None


class ChecklistItem(BaseModel):
    """
    Individual evaluation step in the proactive workflow checklist.

    Represents a specific step that the agent must complete during the proactive
    productivity workflow, with clear success criteria and evidence requirements.
    """

    checklist_idx: str = Field(
        ..., description="Unique identifier for this checklist item"
    )
    step_type: ChecklistStepType = Field(
        ..., description="Type of step in the workflow"
    )
    description: str = Field(
        ..., description="Human-readable description of what this step requires"
    )
    success_criteria: Dict[str, Any] = Field(
        default_factory=dict,
        description="Criteria that define successful completion of this step",
    )
    evidence_required: List[str] = Field(
        default_factory=list,
        description="List of corpus item IDs that should be accessed/referenced",
    )
    previous_action_idx: Optional[str] = Field(
        None,
        description="ID of previous action that should be completed before this step",
    )

    @field_validator("checklist_idx")
    @classmethod
    def validate_checklist_idx(cls, v: str) -> str:
        """Ensure checklist_idx is not empty."""
        if not v.strip():
            raise ValueError("checklist_idx cannot be empty")
        return v

    @field_validator("description")
    @classmethod
    def validate_description(cls, v: str) -> str:
        """Ensure description is not empty."""
        if not v.strip():
            raise ValueError("description cannot be empty")
        return v


class TaskExecution(BaseModel):
    """
    Final output object representing the agent's selected action.

    This is the concrete action that the agent has decided to take based on
    its analysis of the productivity bottleneck.
    """

    action_type: str = Field(..., description="Type of action selected by the agent")
    target_artifacts: List[str] = Field(
        default_factory=list,
        description="List of corpus item IDs that the action targets",
    )
    execution_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific parameters for executing this action",
    )
    expected_outcomes: List[str] = Field(
        default_factory=list, description="Agent's prediction of action outcomes"
    )

    # Confidence in this action selection
    confidence: float = Field(
        1.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the selected action (0.0-1.0)",
    )

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        """Ensure action_type is not empty."""
        if not v.strip():
            raise ValueError("action_type cannot be empty")
        return v


class ProactiveChecklist(BaseModel):
    """
    Step-by-step evaluation criteria for proactive productivity workflow.

    Defines the complete set of steps that an agent must complete to successfully
    identify and resolve productivity bottlenecks. This checklist measures that
    agents take each necessary action at some step, not necessarily in strict order.
    """

    required_steps: List[ChecklistItem] = Field(
        default_factory=list, description="List of steps that must be completed"
    )
    checklist_id: str = Field(..., description="Unique identifier for this checklist")
    description: Optional[str] = Field(
        None, description="Human-readable description of this checklist's purpose"
    )

    @field_validator("checklist_id")
    @classmethod
    def validate_checklist_id(cls, v: str) -> str:
        """Ensure checklist_id is not empty."""
        if not v.strip():
            raise ValueError("checklist_id cannot be empty")
        return v

    def get_step_by_idx(self, checklist_idx: str) -> Optional[ChecklistItem]:
        """Get a specific checklist item by its index."""
        return next(
            (
                step
                for step in self.required_steps
                if step.checklist_idx == checklist_idx
            ),
            None,
        )

    def get_steps_by_type(self, step_type: ChecklistStepType) -> List[ChecklistItem]:
        """Get all checklist items of a specific type."""
        return [step for step in self.required_steps if step.step_type == step_type]


class PatternInstance(BaseModel):
    """
    Instance of a pattern detected in the corpus.

    This represents a specific occurrence of a bottleneck or issue pattern
    that the agent should identify and resolve.
    """

    pattern_id: str = Field(..., description="Identifier for the pattern type")
    artifacts: Dict[str, Any] = Field(
        ..., description="Artifacts involved in this pattern"
    )
    evidence: List[List[str]] = Field(
        default_factory=list,
        description="Evidence triples as [source, relation, target]",
    )
    expected_actions: List[str] = Field(
        default_factory=list, description="Expected actions to resolve this pattern"
    )
    required_params: List[str] = Field(
        default_factory=list, description="Required parameters for actions"
    )
    difficulty_tag: str = Field(..., description="Difficulty level tag")
    notes: Optional[str] = Field(None, description="Additional notes")

    @field_validator("evidence")
    @classmethod
    def validate_evidence_format(cls, v: List[List[str]]) -> List[List[str]]:
        """Ensure evidence follows the expected triple format."""
        for edge in v:
            if len(edge) != 3:
                raise ValueError("Evidence must be triples [source, relation, target]")
        return v

    @field_validator("pattern_id")
    @classmethod
    def validate_pattern_id(cls, v: str) -> str:
        """Ensure pattern_id is not empty."""
        if not v.strip():
            raise ValueError("pattern_id cannot be empty")
        return v


class TruePositive(BaseModel):
    """True positive specification for distractor avoidance in generation."""

    target_problem: str = Field(
        ..., description="Target problem description to avoid in distractors"
    )
    required_items: List[str] = Field(
        default_factory=list, description="Required corpus item IDs"
    )
    task_execution: "TaskExecution" = Field(
        ..., description="Task execution specification"
    )

    @field_validator("target_problem")
    @classmethod
    def _non_empty_problem(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value


class DatasetBundle(BaseModel):
    """
    Container for a complete dataset bundle using checklist-based evaluation.

    This is the main container that holds all corpus items, proactive checklists,
    and pattern instances for evaluating agent proactivity.
    """

    name: str = Field(..., description="Name of the dataset")
    description: Optional[str] = Field(None, description="Description of the dataset")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(),
        description="When the bundle was created",
    )

    @field_serializer("created_at")
    def serialize_created_at(self, value: datetime) -> str:
        """Serialize created_at to ISO format string."""
        return value.isoformat()

    # Core data
    corpus: List[CorpusItem] = Field(
        default_factory=list, description="All corpus items"
    )
    proactive_checklists: List[ProactiveChecklist] = Field(
        default_factory=list, description="Proactive workflow checklists for evaluation"
    )

    # Pattern evaluation data
    patterns: List[PatternInstance] = Field(
        default_factory=list, description="Pattern instances representing bottlenecks"
    )

    # Metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not empty."""
        if not v.strip():
            raise ValueError("name cannot be empty")
        return v

    def add_corpus_item(self, item: CorpusItem) -> None:
        """Add a corpus item to the bundle."""
        self.corpus.append(item)

    def add_checklist(self, checklist: ProactiveChecklist) -> None:
        """Add a proactive checklist to the bundle."""
        self.proactive_checklists.append(checklist)

    def add_pattern(self, pattern: PatternInstance) -> None:
        """Add a pattern instance to the bundle."""
        self.patterns.append(pattern)

    def get_corpus_item(self, item_id: str) -> Optional[CorpusItem]:
        """Get a corpus item by ID."""
        return next((item for item in self.corpus if item.id == item_id), None)

    def get_checklist(self, checklist_id: str) -> Optional[ProactiveChecklist]:
        """Get a proactive checklist by ID."""
        return next(
            (
                checklist
                for checklist in self.proactive_checklists
                if checklist.checklist_id == checklist_id
            ),
            None,
        )

    def validate_bundle(self) -> List[str]:
        """
        Validate the bundle for consistency.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        # Check that all corpus item IDs are unique
        corpus_ids = [item.id for item in self.corpus]
        if len(corpus_ids) != len(set(corpus_ids)):
            errors.append("Duplicate corpus item IDs found")

        # Check that checklist evidence references exist in corpus
        corpus_id_set = {item.id for item in self.corpus}
        for checklist in self.proactive_checklists:
            for step in checklist.required_steps:
                for evidence_id in step.evidence_required:
                    if evidence_id not in corpus_id_set:
                        errors.append(
                            f"Checklist {checklist.checklist_id} step {step.checklist_idx} "
                            f"references non-existent corpus item: {evidence_id}"
                        )

        # Check that checklist IDs are unique
        checklist_ids = [
            checklist.checklist_id for checklist in self.proactive_checklists
        ]
        if len(checklist_ids) != len(set(checklist_ids)):
            errors.append("Duplicate checklist IDs found")

        # Check that pattern IDs are unique
        pattern_ids = [pattern.pattern_id for pattern in self.patterns]
        if len(pattern_ids) != len(set(pattern_ids)):
            errors.append("Duplicate pattern IDs found")

        return errors
