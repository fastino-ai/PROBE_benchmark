"""PersonaSim-specific configs and validators.

Exports:
- unified_contracts: core data models
- validators: schema validation utilities
- actions: proactive action configuration and registry (PR6)
"""

from configs.unified_contracts import (
    CorpusItem,
    CorpusItemType,
    ChecklistItem,
    ChecklistStepType,
    TaskExecution,
    ProactiveChecklist,
    PatternInstance,
    TruePositive,
    DatasetBundle,
)

from configs.validators import (
    SchemaValidator,
    ValidationError,
    validate_bundle_file,
    create_bundle_from_existing_data,
)

from configs.actions import (
    ProactiveAction,
    ActionRegistry,
    load_actions_from_experiment_spec,
    build_registry_from_spec,
)

__all__ = [
    # unified_contracts
    "CorpusItem",
    "CorpusItemType",
    "ChecklistItem",
    "ChecklistStepType",
    "TaskExecution",
    "ProactiveChecklist",
    "PatternInstance",
    "TruePositive",
    "DatasetBundle",
    # validators
    "SchemaValidator",
    "ValidationError",
    "validate_bundle_file",
    "create_bundle_from_existing_data",
    # actions (PR6)
    "ProactiveAction",
    "ActionRegistry",
    "load_actions_from_experiment_spec",
    "build_registry_from_spec",
]
