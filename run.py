#!/usr/bin/env python3
"""PROBE Data Generation Runner.

Main pipeline for generating PROBE benchmark data:
personas → world models → bottlenecks → checklists → true positives → distractors

Usage:
    python run.py --mode batch --count 30 --difficulty medium
    python run.py --config my_config.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any

import yaml
from dotenv import load_dotenv

# Config
from configs.config_schema import DataGenerationConfig, DifficultyLevel, ExecutionMode

# Data
from data_generation.data.linkedin_profile import (
    load_linkedin_personas,
    LinkedInPersona,
)

# Generators - they handle all parallelization internally
from data_generation.generators.world_model import WorldModelGenerator, WorldModel
from data_generation.generators.bottleneck_injector import (
    BottleneckInjector,
    Bottleneck,
)
from data_generation.generators.checklist_generator import ChecklistGenerator
from data_generation.generators.true_positives_generator import TruePositivesGenerator
from data_generation.generators.distractor_generator import DistractorGenerator
from data_generation.generators.multi_llm_client import create_default_multi_llm_client
from data_generation.generators.generation_result import (
    GenerationBatch,
    PersonaResult,
    BottleneckResult,
)
from configs.unified_contracts import ProactiveChecklist, CorpusItem

# Difficulty utilities
from data_generation.utils.difficulty import (
    map_to_context_difficulty,
    map_to_distractor_difficulty,
)

# Constants
DEFAULT_WORKER_COUNT = 8
DEFAULT_SAMPLE_COUNT = 30
DEFAULT_DISTRACTOR_COUNT = 10
MAX_FILENAME_LENGTH = 50


class PROBEPipeline:
    """PROBE benchmark data generation pipeline.

    Orchestrates end-to-end workflow: personas → world models → bottlenecks+actions →
    checklists → true positives → distractors. Maintains 1:1 associations and
    uses parallel processing internally.

    Args:
        config: DataGenerationConfig with mode, difficulty, count, etc.
    """

    def __init__(self, config: DataGenerationConfig) -> None:
        """Initialize pipeline with config."""
        self.config = config
        self.setup()

    def setup(self) -> None:
        """Set up logging, output directory, LLM clients, generators, and load personas."""
        # Logging
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = (
            self.config.output_directory / f"{timestamp}_{self.config.mode.value}"
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # LLM client setup based on config
        if self.config.use_multi_llm:
            self.logger.info("Setting up Multi-LLM client with all providers...")
            self.llm_client = create_default_multi_llm_client()
        else:
            self.logger.info(
                f"Setting up Multi-LLM client with single model: {self.config.llm_model}"
            )
            self.llm_client = self._create_single_model_multi_llm_client(
                self.config.llm_model
            )

        # Use the same interface regardless of single or multi-LLM
        llm_func = self.llm_client.create_llm_function()
        json_llm_func = self.llm_client.create_json_llm_function()

        # Initialize generators with parallel processing enabled
        # All generators should support batch processing internally
        self.world_gen = WorldModelGenerator(
            llm_generate_func=json_llm_func, max_workers=self.config.max_workers
        )
        self.bottleneck_gen = BottleneckInjector(
            llm_generate_func=json_llm_func,
            max_workers=self.config.max_workers,
            world_model_generator=self.world_gen,
        )
        self.checklist_gen = ChecklistGenerator(
            llm_generate_func=json_llm_func, max_workers=self.config.max_workers
        )
        self.true_pos_gen = TruePositivesGenerator(
            llm_generate_func=llm_func,
            max_workers=self.config.max_workers,
        )
        self.distractor_gen = DistractorGenerator(
            llm_generate_func=llm_func,
            max_workers=self.config.max_workers,
        )

        # Load ALL personas
        self.logger.info("Loading personas...")
        all_personas = load_linkedin_personas()

        # Select the ones we need based on config
        if self.config.mode == ExecutionMode.TEST:
            indices = [self.config.start_persona_index % len(all_personas)]
        else:
            indices = [
                (self.config.start_persona_index + i) % len(all_personas)
                for i in range(self.config.count)
            ]

        self.personas = [all_personas[i] for i in indices]
        self.logger.info(f"Selected {len(self.personas)} personas for processing")

        self.start_time = time.time()

    def _create_single_model_multi_llm_client(self, model_name: str):
        """Create MultiLLMClient with single model (same infra as multi-LLM)."""
        from data_generation.generators.multi_llm_client import (
            MultiLLMClient,
            LLMConfig,
            LLMProvider,
        )

        # Map model name to provider config
        if model_name.startswith("claude"):
            provider, api_key = LLMProvider.CLAUDE, "ANTHROPIC_API_KEY"
        elif model_name.startswith(("gpt-", "o1")):
            provider, api_key = LLMProvider.OPENAI, "OPENAI_API_KEY"
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        config = LLMConfig(
            provider=provider,
            model_name=model_name,
            api_key_env_var=api_key,
            max_retries=5,
        )

        return MultiLLMClient(configs=[config], fallback_order=[provider])

    def run(self) -> None:
        """Execute complete pipeline: world models → bottlenecks+actions →
        checklists → true positives → distractors → save results.
        Maintains 1:1 associations throughout.
        """
        # Convert difficulty settings once
        context_diff = map_to_context_difficulty(self.config.difficulty)
        distractor_diff = map_to_distractor_difficulty(self.config.difficulty)

        # Step 1: Generate ALL world models in parallel (1:1 with personas)
        self.logger.info(
            f"Generating world models for {len(self.personas)} personas..."
        )
        world_models = self.world_gen.generate(
            personas=self.personas, difficulty=context_diff
        )
        assert len(world_models) == len(
            self.personas
        ), "World models must be 1:1 with personas"
        self.logger.info(f"✓ Generated {len(world_models)} world models")

        # Step 2: Generate bottlenecks and actions for each world model (maintains persona association)
        # Returns List[List[Bottleneck]] - outer list indexed by persona
        # Also generates actions and injects them into world models
        self.logger.info("Injecting bottlenecks and generating actions...")
        bottlenecks_per_persona = self.bottleneck_gen.generate(
            world_models=world_models,
            difficulty=context_diff,
            personas=self.personas,
        )
        assert len(bottlenecks_per_persona) == len(
            self.personas
        ), "Must maintain persona association"
        total_bottlenecks = sum(len(b) for b in bottlenecks_per_persona)
        self.logger.info(
            f"✓ Generated {total_bottlenecks} bottlenecks and actions across {len(self.personas)} personas"
        )

        # Step 3: Generate checklists for each bottleneck (maintains persona→bottleneck association)
        # Returns List[List[Checklist]] - outer by persona, inner by bottleneck
        self.logger.info("Generating checklists...")
        checklists_per_persona = self.checklist_gen.generate(
            world_models=world_models, bottlenecks_per_persona=bottlenecks_per_persona
        )
        assert len(checklists_per_persona) == len(
            self.personas
        ), "Must maintain persona association"
        for p_idx in range(len(self.personas)):
            assert len(checklists_per_persona[p_idx]) == len(
                bottlenecks_per_persona[p_idx]
            ), f"Persona {p_idx}: checklists must be 1:1 with bottlenecks"
        self.logger.info(
            f"✓ Generated checklists for all {total_bottlenecks} bottlenecks"
        )

        # Step 4: Generate true positives for each checklist (maintains full chain of associations)
        # Returns List[List[Dict]] with checklist + true_positives for each
        self.logger.info("Generating true positives...")
        results_with_tp = self.true_pos_gen.generate(
            world_models=world_models,
            bottlenecks_per_persona=bottlenecks_per_persona,
            checklists_per_persona=checklists_per_persona,
            coordinated=self.config.coordinated_generation,
        )
        assert len(results_with_tp) == len(
            self.personas
        ), "Must maintain persona association"
        self.logger.info("✓ Generated true positives for all checklists")

        # Step 5: Generate distractors - shared per persona but avoiding ALL that persona's bottlenecks
        # Returns List[List[Distractor]] - one list per persona
        distractors_per_persona = []
        if self.config.generate_distractors:
            self.logger.info("Generating distractors...")
            distractors_per_persona = self.distractor_gen.generate(
                world_models=world_models,
                bottlenecks_per_persona=bottlenecks_per_persona,
                k=self.config.distractor_count,
                difficulty=distractor_diff,
            )
            assert len(distractors_per_persona) == len(
                self.personas
            ), "Must have distractors for each persona"
            self.logger.info("✓ Generated distractors for all personas")
        else:
            distractors_per_persona = [[] for _ in self.personas]

        # Step 6: Build structured results maintaining ALL associations
        self.logger.info("Building structured results...")
        batch = self._build_generation_batch(
            personas=self.personas,
            world_models=world_models,
            bottlenecks_per_persona=bottlenecks_per_persona,
            checklists_per_persona=checklists_per_persona,
            results_with_tp=results_with_tp,
            distractors_per_persona=distractors_per_persona,
        )

        # Step 7: Save results
        self.logger.info("Saving results...")
        self.save_results_per_world_model(batch)

        # Final report
        duration = time.time() - self.start_time
        summary = batch.to_dict()["summary"]
        print(f"\n{'=' * 50}")
        print("GENERATION COMPLETE")
        print(f"{'=' * 50}")
        print(f"Time: {duration:.1f}s")
        print(f"Personas: {summary['num_personas']}")
        print(f"Total bottlenecks: {summary['total_bottlenecks']}")
        print(f"Total true positives: {summary['total_true_positives']}")
        print(f"Total distractors: {summary['total_distractors']}")
        print(f"Output: {self.output_dir}")

        # Save summary
        self.save_summary(duration, batch)

    def _build_bottleneck_result(
        self,
        persona_idx: int,
        bottleneck_idx: int,
        bottlenecks_per_persona: List[List[Bottleneck]],
        results_with_tp: List[List[Dict[str, Any]]],
    ) -> BottleneckResult:
        """Build BottleneckResult with bottleneck, checklist, and true positives."""
        return BottleneckResult(
            bottleneck=bottlenecks_per_persona[persona_idx][bottleneck_idx],
            checklist=results_with_tp[persona_idx][bottleneck_idx]["checklist"],
            true_positives=results_with_tp[persona_idx][bottleneck_idx][
                "true_positives"
            ],
        )

    def _build_persona_result(
        self,
        persona_idx: int,
        personas: List[LinkedInPersona],
        world_models: List[WorldModel],
        bottleneck_results: List[BottleneckResult],
        distractors_per_persona: List[List[CorpusItem]],
    ) -> PersonaResult:
        """Build PersonaResult aggregating persona, world model, bottlenecks, and distractors."""
        return PersonaResult(
            persona=personas[persona_idx],
            world_model=world_models[persona_idx],
            bottleneck_results=bottleneck_results,
            distractors=distractors_per_persona[persona_idx],
        )

    def _build_generation_batch(
        self,
        personas: List[LinkedInPersona],
        world_models: List[WorldModel],
        bottlenecks_per_persona: List[List[Bottleneck]],
        checklists_per_persona: List[List[ProactiveChecklist]],
        results_with_tp: List[List[Dict[str, Any]]],
        distractors_per_persona: List[List[CorpusItem]],
    ) -> GenerationBatch:
        """Assemble all generated data into GenerationBatch with proper associations."""
        persona_results = []

        for p_idx in range(len(personas)):
            # Build bottleneck results for this persona
            bottleneck_results = [
                self._build_bottleneck_result(
                    p_idx, b_idx, bottlenecks_per_persona, results_with_tp
                )
                for b_idx in range(len(bottlenecks_per_persona[p_idx]))
            ]

            # Build complete persona result
            persona_result = self._build_persona_result(
                p_idx,
                personas,
                world_models,
                bottleneck_results,
                distractors_per_persona,
            )
            persona_results.append(persona_result)

        return GenerationBatch(persona_results=persona_results)

    def _sanitize_filename(
        self, name: str, max_length: int = MAX_FILENAME_LENGTH
    ) -> str:
        """Convert name to safe filename (alphanumeric + underscores, lowercase, truncated)."""
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name).lower()
        return safe_name[:max_length]

    def save_results_per_world_model(self, batch: GenerationBatch) -> None:
        """Save input/output format for inference pipeline (multiple bottlenecks per world model)."""
        # Create input and output directories
        input_dir = self.output_dir / "inputs"
        output_dir = self.output_dir / "outputs"
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        for p_idx, persona_result in enumerate(batch.persona_results):
            safe_name = self._sanitize_filename(persona_result.persona.name)

            # Save each bottleneck as a separate input/output file pair
            for b_idx, br in enumerate(persona_result.bottleneck_results):
                # Format: bottleneck_{persona_num}_{bottleneck_num}_{persona_name}_{input/output}.json
                base_filename = f"bottleneck_{p_idx+1:03d}_{b_idx+1:02d}_{safe_name}"

                # Combine true positives and distractors into data_points
                all_data_points = []
                true_positive_ids = []
                distractor_ids = []

                for tp in br.true_positives:
                    tp_dict = tp.model_dump()
                    all_data_points.append(tp_dict)
                    true_positive_ids.append(tp_dict["id"])

                # Share distractors across all bottlenecks for this persona
                for d in persona_result.distractors:
                    d_dict = d.model_dump()
                    all_data_points.append(d_dict)
                    distractor_ids.append(d_dict["id"])

                # INPUT file - strip solves_bottleneck to avoid leaking ground truth
                world_model_dict = persona_result.world_model.model_dump()

                # Remove solves_bottleneck from all actions in the input file
                if "available_actions" in world_model_dict:
                    for action in world_model_dict["available_actions"]:
                        action.pop("solves_bottleneck", None)

                input_data = {
                    "world_model": world_model_dict,
                    "data_points": all_data_points,
                }

                with open(input_dir / f"{base_filename}_input.json", "w") as f:
                    json.dump(input_data, f, indent=2)

                # OUTPUT file
                output_data = {
                    "world_model": persona_result.world_model.model_dump(),
                    "bottleneck": br.bottleneck.model_dump(),
                    "checklist": br.checklist.model_dump(),
                    "true_positive_ids": true_positive_ids,
                    "distractor_ids": distractor_ids,
                    "metadata": {
                        "model": getattr(self.config, "llm_model", "gpt-4"),
                        "difficulty": self.config.difficulty.value,
                        "persona_index": p_idx,
                        "use_multi_llm": True,
                        "coordinated_generation": self.config.coordinated_generation,
                        "example_index": p_idx + 1,
                        "bottleneck_index": b_idx + 1,
                        "generated_at": datetime.now().isoformat(),
                    },
                }

                with open(output_dir / f"{base_filename}_output.json", "w") as f:
                    json.dump(output_data, f, indent=2)

                self.logger.info(f"Saved {base_filename} (input + output)")

    def save_summary(self, duration: float, batch: GenerationBatch) -> None:
        """Save summary.json with config, statistics, and execution time."""
        summary = batch.to_dict()["summary"]

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(
                {
                    "config": {
                        "mode": self.config.mode.value,
                        "difficulty": self.config.difficulty.value,
                        "count": self.config.count,
                        "generate_distractors": self.config.generate_distractors,
                        "coordinated_generation": self.config.coordinated_generation,
                    },
                    "results": summary,
                    "duration": duration,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )


def main() -> None:
    """Main entry point for PROBE data generation pipeline.

    Command-line Arguments:
        --config, -c: Path to YAML configuration file (overrides other args)
        --mode, -m: Generation mode (batch|single|test), default: batch
        --count, -n: Number of samples to generate, default: 30
        --difficulty, -d: Difficulty level (easy|medium|hard), default: medium
        --no-distractors: Skip distractor generation (faster execution)
        --workers, -w: Number of parallel workers, default: 8

    Environment Variables:
        OPENAI_API_KEY: Required for LLM API calls
        ANTHROPIC_API_KEY: Optional for Claude models

    Examples:
        >>> # Generate 30 medium difficulty samples with distractors
        >>> python run.py --mode batch --count 30 --difficulty medium

        >>> # Quick test with 1 sample, no distractors
        >>> python run.py --mode test --count 1 --no-distractors

        >>> # Use configuration file
        >>> python run.py --config my_config.yaml
    """
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="PROBE Data Generation - Generate synthetic benchmark data for "
        "evaluating proactive AI agent capabilities",
        epilog="For more information, see README.md",
    )
    parser.add_argument(
        "--config", "-c", help="Path to YAML config file (overrides other arguments)"
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["batch", "single", "test"],
        default="batch",
        help="Generation mode: batch (default), single, or test",
    )
    parser.add_argument(
        "--count",
        "-n",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"Number of samples to generate (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument(
        "--difficulty",
        "-d",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty level: easy, medium (default), or hard",
    )
    parser.add_argument(
        "--no-distractors",
        action="store_true",
        help="Skip distractor generation (faster, for testing)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=DEFAULT_WORKER_COUNT,
        help=f"Number of parallel workers (default: {DEFAULT_WORKER_COUNT})",
    )
    args = parser.parse_args()

    # Check environment - API key is required
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is required")
        print("Please set it in your .env file or environment")
        sys.exit(1)

    # Build config from args or YAML file
    if args.config:
        try:
            with open(args.config) as f:
                config = DataGenerationConfig(**yaml.safe_load(f))
            print(f"Loaded configuration from {args.config}")
        except FileNotFoundError:
            print(f"Error: Config file not found: {args.config}")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    else:
        config = DataGenerationConfig(
            mode=ExecutionMode(args.mode),
            count=args.count,
            difficulty=DifficultyLevel(args.difficulty),
            generate_distractors=not args.no_distractors,
            max_workers=args.workers,
            parallel=True,  # Always use internal parallelization
        )

    # Execute pipeline
    try:
        pipeline = PROBEPipeline(config)
        pipeline.run()
    except KeyboardInterrupt:
        print("\n\nGeneration interrupted by user")
        sys.exit(130)  # Standard exit code for SIGINT
    except Exception as e:
        print(f"\n\nError during generation: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
