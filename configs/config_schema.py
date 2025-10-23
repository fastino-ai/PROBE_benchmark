"""
Simple unified configuration for the PersonaSim data generation system.

Focuses on main execution parameters without duplicating existing generator configs.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


class ExecutionMode(str, Enum):
    """Execution modes for the data generation system."""

    BATCH = "batch"  # Generate multiple examples in batch
    SINGLE = "single"  # Generate a single example
    TEST = "test"  # Test mode with minimal generation


class DifficultyLevel(str, Enum):
    """Difficulty levels for generation complexity."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    PRODUCTION = "production"


class DataGenerationConfig(BaseModel):
    """Simple configuration for PersonaSim data generation."""

    # Execution mode
    mode: ExecutionMode = ExecutionMode.BATCH

    # Core parameters
    count: int = Field(default=30, ge=1, description="Number of examples to generate")
    difficulty: DifficultyLevel = Field(
        default=DifficultyLevel.MEDIUM, description="Generation difficulty"
    )
    start_persona_index: int = Field(
        default=0, ge=0, description="Starting persona index"
    )

    # Output settings
    output_directory: Path = Field(
        default=Path("generated_data"), description="Base output directory"
    )

    # Feature flags
    generate_distractors: bool = Field(default=True, description="Generate distractors")
    distractor_count: int = Field(
        default=20, ge=0, description="Number of distractors per example"
    )
    prepare_annotations: bool = Field(
        default=True, description="Prepare data for annotation"
    )
    coordinated_generation: bool = Field(
        default=True, description="Use coordinated true positive generation"
    )

    # LLM settings
    use_multi_llm: bool = Field(
        default=True, description="Use multi-LLM client with fallback"
    )
    llm_model: str = Field(default="gpt-4.1-mini", description="LLM model to use")

    # Example data for grounding
    example_emails_path: Optional[Path] = Field(
        default=None,
        description="Path to JSON file containing example emails for grounding generation",
    )

    # Parallel processing
    parallel: bool = Field(default=True, description="Use parallel processing")
    max_workers: int = Field(
        default=8, ge=1, le=32, description="Maximum number of parallel workers"
    )

    # Utility methods
    @classmethod
    def batch_config(
        cls,
        count: int = 30,
        difficulty: DifficultyLevel = DifficultyLevel.MEDIUM,
        output_dir: Optional[str] = None,
    ) -> "DataGenerationConfig":
        """Quick batch configuration."""
        return cls(
            mode=ExecutionMode.BATCH,
            count=count,
            difficulty=difficulty,
            output_directory=Path(output_dir) if output_dir else Path("batch_examples"),
        )

    @classmethod
    def test_config(cls) -> "DataGenerationConfig":
        """Quick test configuration."""
        return cls(
            mode=ExecutionMode.TEST,
            count=1,
            difficulty=DifficultyLevel.EASY,
            generate_distractors=True,
            distractor_count=5,
            parallel=False,
            output_directory=Path("test_output"),
        )

    def validate_environment(self) -> Optional[str]:
        """Check required environment variables."""
        import os

        if not os.getenv("OPENAI_API_KEY"):
            return "OPENAI_API_KEY environment variable is required"
        return None
