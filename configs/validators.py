"""
Validators for unified contracts schema.

This module provides validation utilities for the proactivity benchmark
data models using checklist-based evaluation.
"""

from typing import List, Dict, Any, Optional, Union
import json
from pathlib import Path

from configs.unified_contracts import (
    CorpusItem,
    CorpusItemType,
    ChecklistItem,
    ChecklistStepType,
    TaskExecution,
    ProactiveChecklist,
    PatternInstance,
    DatasetBundle,
)


class ValidationError(Exception):
    """Raised when validation fails."""

    pass


class SchemaValidator:
    """
    Validator for proactivity benchmark schemas.

    Provides validation methods for all unified contract models using
    checklist-based evaluation.
    """

    def __init__(self):
        """Initialize the validator."""
        self._email_schema = None
        self._event_schema = None
        self._document_schema = None

    def validate_corpus_item(self, item: CorpusItem) -> List[str]:
        """
        Validate a CorpusItem.

        Args:
            item: The corpus item to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not item.id.strip():
            errors.append("CorpusItem id cannot be empty")

        if not isinstance(item.eval_json, dict):
            errors.append("CorpusItem eval_json must be a dictionary")

        return errors

    def validate_checklist_item(self, item: ChecklistItem) -> List[str]:
        """
        Validate a ChecklistItem.

        Args:
            item: The checklist item to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not item.checklist_idx.strip():
            errors.append("ChecklistItem checklist_idx cannot be empty")

        if not item.description.strip():
            errors.append("ChecklistItem description cannot be empty")

        if not isinstance(item.success_criteria, dict):
            errors.append("ChecklistItem success_criteria must be a dictionary")

        if not isinstance(item.evidence_required, list):
            errors.append("ChecklistItem evidence_required must be a list")

        # Validate step type
        valid_step_types = {step_type.value for step_type in ChecklistStepType}
        if item.step_type.value not in valid_step_types:
            errors.append(
                f"Invalid step_type: {item.step_type}. Must be one of {valid_step_types}"
            )

        # Validate specific step type criteria
        if item.step_type == ChecklistStepType.RETRIEVAL:
            if not item.success_criteria:
                errors.append("RETRIEVAL step must have success_criteria defined")
        elif item.step_type == ChecklistStepType.TASK_EXECUTION:
            if "action_selection" not in item.success_criteria:
                errors.append(
                    "TASK_EXECUTION step must have 'action_selection' in success_criteria"
                )
            if "parameter_configuration" not in item.success_criteria:
                errors.append(
                    "TASK_EXECUTION step must have 'parameter_configuration' in success_criteria"
                )

        return errors

    def validate_task_execution(self, execution: TaskExecution) -> List[str]:
        """
        Validate a TaskExecution.

        Args:
            execution: The task execution to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not execution.action_type.strip():
            errors.append("TaskExecution action_type cannot be empty")

        if not isinstance(execution.target_artifacts, list):
            errors.append("TaskExecution target_artifacts must be a list")

        if not isinstance(execution.execution_parameters, dict):
            errors.append("TaskExecution execution_parameters must be a dictionary")

        if not isinstance(execution.expected_outcomes, list):
            errors.append("TaskExecution expected_outcomes must be a list")

        if not (0.0 <= execution.confidence <= 1.0):
            errors.append("TaskExecution confidence must be between 0.0 and 1.0")

        return errors

    def validate_proactive_checklist(self, checklist: ProactiveChecklist) -> List[str]:
        """
        Validate a ProactiveChecklist.

        Args:
            checklist: The proactive checklist to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not checklist.checklist_id.strip():
            errors.append("ProactiveChecklist checklist_id cannot be empty")

        # Validate each required step
        for i, step in enumerate(checklist.required_steps):
            step_errors = self.validate_checklist_item(step)
            if step_errors:
                errors.extend(
                    [f"Step {i} ({step.checklist_idx}): {e}" for e in step_errors]
                )

        # Validate step references
        step_indices = {step.checklist_idx for step in checklist.required_steps}
        for step in checklist.required_steps:
            if (
                step.previous_action_idx
                and step.previous_action_idx not in step_indices
            ):
                errors.append(
                    f"Step {step.checklist_idx} references non-existent previous step: {step.previous_action_idx}"
                )

        return errors

    def validate_pattern_instance(self, pattern: PatternInstance) -> List[str]:
        """
        Validate a PatternInstance.

        Args:
            pattern: The pattern instance to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not pattern.pattern_id.strip():
            errors.append("PatternInstance pattern_id cannot be empty")

        if not pattern.difficulty_tag.strip():
            errors.append("PatternInstance difficulty_tag cannot be empty")

        # Validate expected_actions and required_params consistency
        if pattern.expected_actions and not pattern.required_params:
            errors.append("PatternInstance has expected_actions but no required_params")

        # Validate evidence format (already done by Pydantic validator, but double-check)
        for i, edge in enumerate(pattern.evidence):
            if len(edge) != 3:
                errors.append(f"PatternInstance evidence edge {i} is not a triple")

        return errors

    def validate_dataset_bundle(self, bundle: DatasetBundle) -> List[str]:
        """
        Validate a DatasetBundle comprehensively.

        Args:
            bundle: The dataset bundle to validate

        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []

        if not bundle.name.strip():
            errors.append("DatasetBundle name cannot be empty")

        # Validate all corpus items
        corpus_ids = set()
        for i, item in enumerate(bundle.corpus):
            item_errors = self.validate_corpus_item(item)
            errors.extend(
                [f"Corpus item {i} ({item.id}): {err}" for err in item_errors]
            )

            if item.id in corpus_ids:
                errors.append(f"Duplicate corpus item ID: {item.id}")
            corpus_ids.add(item.id)

        # Validate proactive checklists
        checklist_ids = set()
        for i, checklist in enumerate(bundle.proactive_checklists):
            checklist_errors = self.validate_proactive_checklist(checklist)
            errors.extend(
                [
                    f"Checklist {i} ({checklist.checklist_id}): {err}"
                    for err in checklist_errors
                ]
            )

            if checklist.checklist_id in checklist_ids:
                errors.append(f"Duplicate checklist ID: {checklist.checklist_id}")
            checklist_ids.add(checklist.checklist_id)

            # Validate that evidence references exist in corpus
            for step in checklist.required_steps:
                for evidence_id in step.evidence_required:
                    if evidence_id not in corpus_ids:
                        errors.append(
                            f"Checklist {checklist.checklist_id} step {step.checklist_idx} references non-existent corpus item: {evidence_id}"
                        )

        # Validate patterns
        pattern_ids = set()
        for i, pattern in enumerate(bundle.patterns):
            pattern_errors = self.validate_pattern_instance(pattern)
            errors.extend(
                [f"Pattern {i} ({pattern.pattern_id}): {err}" for err in pattern_errors]
            )

            if pattern.pattern_id in pattern_ids:
                errors.append(f"Duplicate pattern ID: {pattern.pattern_id}")
            pattern_ids.add(pattern.pattern_id)

        return errors

    def _validate_email_eval_json(self, eval_json: Dict[str, Any]) -> List[str]:
        """Validate email-specific eval_json structure."""
        errors = []
        required_fields = ["title", "content", "source", "metadata", "user_email"]

        for field in required_fields:
            if field not in eval_json:
                errors.append(f"Email eval_json missing required field: {field}")

        if "metadata" in eval_json:
            metadata = eval_json["metadata"]
            if "document_id" not in metadata:
                errors.append("Email metadata missing document_id")

        return errors

    def _validate_event_eval_json(self, eval_json: Dict[str, Any]) -> List[str]:
        """Validate event-specific eval_json structure."""
        errors = []
        # Add event-specific validation logic here
        # This would depend on the actual event schema structure
        return errors

    def _validate_document_eval_json(self, eval_json: Dict[str, Any]) -> List[str]:
        """Validate document-specific eval_json structure."""
        errors = []
        # Add document-specific validation logic here
        # This would depend on the actual document schema structure
        return errors


def validate_bundle_file(filepath: Union[str, Path]) -> DatasetBundle:
    """
    Load and validate a dataset bundle from a JSON file.

    Args:
        filepath: Path to the bundle JSON file

    Returns:
        Validated DatasetBundle instance

    Raises:
        ValidationError: If validation fails
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Bundle file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Parse into Pydantic model
    try:
        bundle = DatasetBundle(**data)
    except Exception as e:
        raise ValidationError(f"Failed to parse bundle: {e}")

    # Validate the bundle
    validator = SchemaValidator()
    errors = validator.validate_dataset_bundle(bundle)

    if errors:
        error_msg = "Bundle validation failed:\n" + "\n".join(
            f"- {err}" for err in errors
        )
        raise ValidationError(error_msg)

    return bundle


def create_bundle_from_existing_data(
    emails_path: Optional[Union[str, Path]] = None,
    events_path: Optional[Union[str, Path]] = None,
    documents_path: Optional[Union[str, Path]] = None,
    bundle_name: str = "converted_bundle",
) -> DatasetBundle:
    """
    Create a DatasetBundle from existing evaluation data files.

    Note: This function has been simplified for checklist-based evaluation.
    It no longer processes queries separately as they are integrated into checklists.

    Args:
        emails_path: Path to emails.json
        events_path: Path to events.json
        documents_path: Path to documents.json
        bundle_name: Name for the bundle

    Returns:
        DatasetBundle with converted data
    """
    bundle = DatasetBundle(name=bundle_name)

    # Convert emails
    if emails_path and Path(emails_path).exists():
        with open(emails_path, "r", encoding="utf-8") as f:
            emails_data = json.load(f)

        for email in emails_data.get("emails", []):
            corpus_item = CorpusItem(
                id=email.get("metadata", {}).get(
                    "document_id", f"email_{len(bundle.corpus)}"
                ),
                type=CorpusItemType.EMAIL,
                eval_json=email,
            )
            bundle.add_corpus_item(corpus_item)

    # Convert events
    if events_path and Path(events_path).exists():
        with open(events_path, "r", encoding="utf-8") as f:
            events_data = json.load(f)

        for event in events_data.get("events", []):
            corpus_item = CorpusItem(
                id=event.get("id", f"event_{len(bundle.corpus)}"),
                type=CorpusItemType.EVENT,
                eval_json=event,
            )
            bundle.add_corpus_item(corpus_item)

    # Convert documents
    if documents_path and Path(documents_path).exists():
        with open(documents_path, "r", encoding="utf-8") as f:
            docs_data = json.load(f)

        for doc in docs_data.get("documents", []):
            corpus_item = CorpusItem(
                id=doc.get("id", f"doc_{len(bundle.corpus)}"),
                type=CorpusItemType.DOCUMENT,
                eval_json=doc,
            )
            bundle.add_corpus_item(corpus_item)

    return bundle
