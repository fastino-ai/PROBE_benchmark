"""
Action configuration models and utilities (PR6).

This module defines the configuration schema for proactive actions that an
agent may select/execute in the proactivity benchmark system. It provides:

- ProactiveAction: Declarative action configuration (id, type, constraints, params)
- ActionRegistry: Lightweight registry for lookup and validation
- Loader helpers to build actions/registry from an experiment specification

Design goals:
- Keep the schema minimal and flexible, similar to the simplified Persona model
- Avoid over-constraining action types; allow experiment authors to define new ones
- Provide helpful validation without requiring a full JSON Schema validator
"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, ConfigDict, field_validator

from .unified_contracts import TaskExecution


class ProactiveAction(BaseModel):
    """Declarative configuration for a proactive action.

    Fields are intentionally minimal to enable flexible experiment authoring.
    ``params_schema`` can optionally include a subset of JSON Schema keys, such as
    ``required`` (list of parameter names) to express minimal parameter contracts
    consumed by downstream generators or validators.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(
        ..., description="Stable identifier for this action, e.g., 'send_followup'"
    )
    type: str = Field(..., description="Category/type of action, e.g., 'email_action'")
    description: Optional[str] = Field(
        None, description="Human-readable description of what this action does"
    )
    constraints: List[str] = Field(
        default_factory=list,
        description=(
            "Named preconditions or policies this action must respect. Semantics are"
            " experiment-defined (e.g., 'thread_stalled_3days', 'same_attendees')."
        ),
    )
    params_schema: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional parameter schema. Supports minimal JSON Schema-like fields,"
            " e.g., {'required': ['tone']} for simple presence checks."
        ),
    )
    solves_bottleneck: Optional[int] = Field(
        None,
        description="The bottleneck that this action solves, if any.",
    )

    @field_validator("id", "type")
    @classmethod
    def _strip_and_require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("constraints")
    @classmethod
    def _validate_constraints(cls, value: List[str]) -> List[str]:
        normalized: List[str] = [
            c.strip() for c in value if isinstance(c, str) and c.strip()
        ]
        # allow empty list; treat invalid entries as errors rather than silently dropping
        if any(not isinstance(c, str) for c in value):
            raise ValueError("constraints must be a list of strings")
        # ensure no empty strings remain
        if len(normalized) != len(value):
            raise ValueError("constraints items must be non-empty strings")
        return normalized

    def to_prompt_context(self) -> str:
        """Render a compact, readable description suitable for prompts/logging."""
        lines: List[str] = [
            f"Action ID: {self.id}",
            f"Type: {self.type}",
        ]
        if self.description:
            lines.append(f"Description: {self.description}")
        if self.constraints:
            lines.append("Constraints:")
            lines.extend([f"- {c}" for c in self.constraints])
        if self.params_schema:
            lines.append("Params Schema:")
            for key, val in self.params_schema.items():
                lines.append(f"- {key}: {val}")
        return "\n".join(lines)


class ActionRegistry:
    """Lightweight registry for configured actions.

    Provides lookup by id, grouping by type, and a helper to validate that a
    TaskExecution references a known action and minimally satisfies required params.
    """

    def __init__(self, actions: Optional[List[ProactiveAction]] = None) -> None:
        self._by_id: Dict[str, ProactiveAction] = {}
        self._by_type: Dict[str, List[ProactiveAction]] = {}
        if actions:
            for action in actions:
                self.add_action(action)

    def add_action(self, action: ProactiveAction) -> None:
        if action.id in self._by_id:
            raise ValueError(f"Duplicate action id: {action.id}")
        self._by_id[action.id] = action
        self._by_type.setdefault(action.type, []).append(action)

    def get_action_by_id(self, action_id: str) -> Optional[ProactiveAction]:
        return self._by_id.get(action_id)

    def get_actions_by_type(self, action_type: str) -> List[ProactiveAction]:
        return list(self._by_type.get(action_type, []))

    def list_ids(self) -> List[str]:
        return sorted(self._by_id.keys())

    @property
    def actions(self) -> List[ProactiveAction]:
        """Return all registered actions."""
        return list(self._by_id.values())

    def validate_task_execution(self, execution: TaskExecution) -> List[str]:
        """Validate that a TaskExecution references a known action and params.

        Rules:
        - action_type must match a configured action id
        - if params_schema has a 'required' list, those keys must be present in execution_parameters
        """
        errors: List[str] = []
        action = self.get_action_by_id(execution.action_type.strip())
        if not action:
            errors.append(
                f"Unknown action_type: {execution.action_type}. Known ids: {', '.join(self.list_ids()) or '∅'}"
            )
            return errors

        required = (
            action.params_schema.get("required", [])
            if isinstance(action.params_schema, dict)
            else []
        )
        if required and isinstance(required, list):
            missing = [
                key for key in required if key not in execution.execution_parameters
            ]
            if missing:
                errors.append(
                    f"Missing required parameters for action '{action.id}': {missing}"
                )
        return errors


def load_actions_from_experiment_spec(spec: Dict[str, Any]) -> List[ProactiveAction]:
    """Load action configurations from an experiment spec dictionary.

    Expected structure (YAML/JSON):
    generation:
      proactive_actions:
        - id: send_followup
          type: email_action
          constraints: [thread_stalled_3days]
          params_schema:
            required: [tone]

    Returns a list of ProactiveAction objects. Raises ValueError on malformed input.
    """
    gen = spec.get("generation") if isinstance(spec, dict) else None
    actions_raw = gen.get("proactive_actions") if isinstance(gen, dict) else None
    if actions_raw is None:
        return []
    if not isinstance(actions_raw, list):
        raise ValueError("generation.proactive_actions must be a list")

    actions: List[ProactiveAction] = []
    for i, item in enumerate(actions_raw):
        if not isinstance(item, dict):
            raise ValueError(f"proactive_actions[{i}] must be an object")
        try:
            action = ProactiveAction(**item)
        except Exception as e:
            raise ValueError(f"Invalid proactive_actions[{i}]: {e}")
        actions.append(action)
    return actions


def build_registry_from_spec(spec: Dict[str, Any]) -> ActionRegistry:
    """Convenience helper to construct an ActionRegistry from an experiment spec."""
    actions = load_actions_from_experiment_spec(spec)
    return ActionRegistry(actions)
