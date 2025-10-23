#!/usr/bin/env python3
"""
Annotation format handler for PersonaSim evaluations.

This module defines the standard format for annotations and predictions,
and provides utilities for saving, loading, and validating them.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class ActionSelection:
    """Represents the selected action and its parameters."""

    name: str
    schema: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {"name": self.name, "schema": self.schema}


@dataclass
class Annotation:
    """
    Standard annotation format for PersonaSim evaluations.

    This format is used for both human annotations and model predictions.
    """

    retrieved_document_ids: List[str]
    bottleneck_description: str
    action_selection: ActionSelection

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "retrieved_document_ids": self.retrieved_document_ids,
            "bottleneck_description": self.bottleneck_description,
            "action_selection": self.action_selection.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Annotation":
        """Create Annotation from dictionary."""
        action_data = data.get("action_selection", {})
        action_selection = ActionSelection(
            name=action_data.get("name", ""), schema=action_data.get("schema", {})
        )

        return cls(
            retrieved_document_ids=data.get("retrieved_document_ids", []),
            bottleneck_description=data.get("bottleneck_description", ""),
            action_selection=action_selection,
        )

    def validate(self) -> List[str]:
        """
        Validate the annotation format.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if not self.retrieved_document_ids:
            errors.append("retrieved_document_ids cannot be empty")

        if not self.bottleneck_description:
            errors.append("bottleneck_description cannot be empty")

        if not self.action_selection.name:
            errors.append("action_selection.name cannot be empty")

        if not isinstance(self.action_selection.schema, dict):
            errors.append("action_selection.schema must be a dictionary")

        return errors


def save_annotation(
    annotation: Annotation,
    example_id: str,
    output_dir: Path,
    annotation_type: str = "annotation",
    metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """
    Save an annotation to a JSON file.

    Args:
        annotation: The annotation to save
        example_id: ID of the example being annotated
        output_dir: Directory to save the annotation
        annotation_type: Either "annotation" or "prediction"
        metadata: Optional metadata to include

    Returns:
        Path to the saved file
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{example_id}_{annotation_type}_{timestamp}.json"
    filepath = output_dir / filename

    # Prepare data
    data = annotation.to_dict()

    # Add metadata if provided
    if metadata:
        data["_metadata"] = {
            **metadata,
            "annotation_type": annotation_type,
            "timestamp": datetime.now().isoformat(),
            "example_id": example_id,
        }

    # Save to file
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return filepath


def load_annotation(filepath: Path) -> Annotation:
    """
    Load an annotation from a JSON file.

    Args:
        filepath: Path to the annotation file

    Returns:
        Loaded Annotation object
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Remove metadata if present
    data.pop("_metadata", None)

    return Annotation.from_dict(data)


def load_annotations_batch(
    directory: Path, pattern: str = "*.json"
) -> Dict[str, Annotation]:
    """
    Load all annotations from a directory.

    Args:
        directory: Directory containing annotation files
        pattern: Glob pattern for files to load

    Returns:
        Dictionary mapping example IDs to Annotations
    """
    annotations = {}

    if not directory.exists():
        return annotations

    for filepath in directory.glob(pattern):
        try:
            # Extract example ID from filename
            # Expected format: example_id_annotation/prediction_timestamp.json
            parts = filepath.stem.split("_")
            if len(parts) >= 3:
                # Join all parts except the last two (type and timestamp)
                example_id = "_".join(parts[:-2])

                annotation = load_annotation(filepath)
                annotations[example_id] = annotation
        except Exception as e:
            print(f"Error loading {filepath}: {e}")

    return annotations
