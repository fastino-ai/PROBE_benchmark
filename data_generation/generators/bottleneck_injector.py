"""
Bottleneck Injection System (PR11).

This module injects realistic productivity bottlenecks into world models.
Bottlenecks are LLM-generated, contextual problems that must be addressable
by the persona's available actions.

Key features:
- LLM-driven bottleneck generation based on world model context
- Highly specific bottlenecks tied to relationships, projects, and documents
- Difficulty-based scaling
- Action alignment to ensure solvability
"""

import logging
from typing import List, Optional
from pathlib import Path
import time
import json
from pydantic import BaseModel, Field
from jinja2 import Environment, FileSystemLoader

from .world_model import (
    WorldModel,
    ContextDifficulty,
)
from data_generation.utils.clients.openai_client import get_openai_client

logger = logging.getLogger(__name__)


class Bottleneck(BaseModel):
    """A productivity bottleneck that can be addressed by available actions."""

    description: str = Field(
        ...,
        description="Highly specific, contextual description of a single bottleneck",
    )
    searchable_artifacts: List[str] = Field(
        default_factory=list,
        description="List of artifacts (docs, emails, tickets) mentioned in the bottleneck",
    )

    class Config:
        arbitrary_types_allowed = True


class BottleneckInjector:
    """Generates and injects contextual bottlenecks into world models."""

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        model: str = "gpt-4.1",
        temperature: float = 1,
        llm_generate_func: Optional[callable] = None,
        max_workers: int = 8,
        world_model_generator: Optional[any] = None,
    ):
        """Initialize the bottleneck injector.

        Args:
            template_dir: Directory containing Jinja2 templates
            model: OpenAI model to use (ignored if llm_generate_func is provided)
            temperature: Temperature for generation (higher = more creative)
            llm_generate_func: Optional custom LLM function for making calls
            max_workers: Maximum number of parallel workers for batch processing
            world_model_generator: WorldModelGenerator instance for generating actions
        """
        self.model = model
        self.temperature = temperature
        self.llm_generate_func = llm_generate_func
        self.max_workers = max_workers
        self.world_model_generator = world_model_generator

        # Set up Jinja2 environment
        if template_dir is None:
            template_dir = (
                Path(__file__).parent.parent / "prompts" / "bottleneck_injection"
            )

        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Initialize OpenAI client only if no custom LLM function is provided
        if llm_generate_func is None:
            self.client = get_openai_client()
        else:
            self.client = None

        if llm_generate_func is not None:
            logger.info("Initialized BottleneckInjector with custom LLM function")
        else:
            logger.info(f"Initialized BottleneckInjector with model={model}")

    def generate(
        self,
        world_models: List[WorldModel],
        difficulty: ContextDifficulty,
        personas: Optional[List[any]] = None,
    ) -> List[List[Bottleneck]]:
        """
        Generate bottlenecks per world model in parallel based on difficulty level,
        and generate actions for each bottleneck.

        Number of bottlenecks per persona based on difficulty:
        - EASY: 1-2 bottlenecks
        - MEDIUM: 2-4 bottlenecks
        - HARD: 4-6 bottlenecks

        Args:
            world_models: List of world models to inject bottlenecks into
            difficulty: Context difficulty level
            personas: List of personas (needed for action generation)

        Returns:
            List of bottleneck lists (variable count per world model based on difficulty)
        """
        # Determine bottleneck count range based on difficulty
        bottleneck_counts = {
            ContextDifficulty.EASY: {"min": 1, "max": 2},
            ContextDifficulty.MEDIUM: {"min": 2, "max": 4},
            ContextDifficulty.HARD: {"min": 4, "max": 6},
        }

        min_count = bottleneck_counts[difficulty]["min"]
        max_count = bottleneck_counts[difficulty]["max"]

        logger.info(
            f"Generating {min_count}-{max_count} bottlenecks per world model for {len(world_models)} world models "
            f"with difficulty={difficulty}"
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed
        import random

        # Set seed for deterministic bottleneck counts across runs
        random.seed(42)
        
        # Pre-generate bottleneck counts for each persona (deterministic order)
        bottleneck_counts_per_persona = [
            random.randint(min_count, max_count) for _ in range(len(world_models))
        ]
        
        logger.info(
            f"Bottleneck counts per persona: {bottleneck_counts_per_persona}"
        )

        # Process ALL world models in parallel
        bottlenecks_per_persona = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks with pre-determined bottleneck counts
            futures_to_idx = {
                executor.submit(
                    self._generate_multiple_bottlenecks,
                    idx,
                    wm,
                    difficulty,
                    bottleneck_counts_per_persona[idx],
                ): idx
                for idx, wm in enumerate(world_models)
            }

            # Collect results in order
            results = {}
            for future in as_completed(futures_to_idx):
                idx = futures_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(
                        f"Failed to generate bottlenecks for world model {idx}: {e}"
                    )
                    raise

            # Sort by index to maintain order
            bottlenecks_per_persona = [results[i] for i in range(len(world_models))]

        total_bottlenecks = sum(len(b) for b in bottlenecks_per_persona)
        logger.info(
            f"✓ Generated {total_bottlenecks} bottlenecks across {len(world_models)} personas "
            f"(avg {total_bottlenecks / len(world_models):.1f} per persona)"
        )

        # Generate actions for each bottleneck
        if self.world_model_generator and personas:
            logger.info("Generating actions for bottlenecks...")
            for idx, (world_model, bottlenecks) in enumerate(
                zip(world_models, bottlenecks_per_persona)
            ):
                actions = self.world_model_generator.generate_actions_for_bottlenecks(
                    persona=personas[idx],
                    org_structure=world_model.organizational_structure,
                    bottlenecks=bottlenecks,
                    difficulty=difficulty,
                )
                world_model.available_actions = actions
            logger.info("✓ Generated actions for all personas")

        return bottlenecks_per_persona

    def _generate_multiple_bottlenecks(
        self,
        idx: int,
        world_model: WorldModel,
        difficulty: ContextDifficulty,
        num_bottlenecks: int,
    ) -> List[Bottleneck]:
        """Generate multiple bottlenecks for a single world model.

        Uses batch generation (all bottlenecks in single LLM call) for efficiency and coherence,
        with fallback to individual generation if batch fails.

        Args:
            idx: Index of the world model
            world_model: The world model to generate bottlenecks for
            difficulty: Context difficulty level
            num_bottlenecks: Number of bottlenecks to generate

        Returns:
            List of generated Bottlenecks
        """
        # Try batch generation first (more efficient and coherent)
        try:
            bottlenecks = self._generate_all_bottlenecks(
                world_model, difficulty, num_bottlenecks
            )
            logger.info(
                f"Generated {len(bottlenecks)} bottlenecks for persona {idx + 1} via batch generation"
            )
            return bottlenecks
        except Exception as e:
            logger.warning(
                f"Batch bottleneck generation failed for persona {idx + 1}, falling back to individual: {e}"
            )
            # Fallback to individual generation
            bottlenecks = []
            for i in range(num_bottlenecks):
                try:
                    bottleneck = self._generate_bottleneck(world_model, difficulty)
                    bottlenecks.append(bottleneck)
                    logger.debug(
                        f"Generated bottleneck {i + 1}/{num_bottlenecks} for persona {idx + 1}"
                    )
                except Exception as bottleneck_error:
                    logger.error(
                        f"Failed to generate bottleneck {i + 1} for persona {idx + 1}: {bottleneck_error}"
                    )
                    # Continue trying remaining bottlenecks

            logger.info(
                f"Generated {len(bottlenecks)} bottlenecks for persona {idx + 1} via individual fallback"
            )
            return bottlenecks

    def _generate_all_bottlenecks(
        self,
        world_model: WorldModel,
        difficulty: ContextDifficulty,
        num_bottlenecks: int,
    ) -> List[Bottleneck]:
        """Generate all bottlenecks in a single LLM call for efficiency.

        This matches Research_PROBE's approach of generating bottlenecks together
        so they're aware of each other and avoid overlap.

        Args:
            world_model: The world model for context
            difficulty: Context difficulty level
            num_bottlenecks: Number of bottlenecks to generate

        Returns:
            List of generated bottlenecks
        """
        # Load template for batch generation
        template = self.template_env.get_template("generate_bottlenecks_batch.j2")

        # Prepare context for generation
        prompt = template.render(
            persona_name=world_model.persona_full_name,
            persona_occupation=world_model.persona_occupation,
            persona_about=world_model.persona_about,
            difficulty=difficulty.value,
            relationships=world_model.relationships,
            org_structure=world_model.organizational_structure,
            personal_context=world_model.personal_context,
            num_bottlenecks=num_bottlenecks,
        )

        # Generate all bottlenecks at once
        _start = time.time()

        if self.llm_generate_func is not None:
            # Use custom LLM function
            content = self.llm_generate_func(prompt).strip()
        else:
            # Use OpenAI client directly
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating highly specific, realistic productivity bottlenecks that reference real names, documents, and deadlines. Each bottleneck must be a SINGLE, focused problem that is directly relevant to the persona's specific occupation and professional background. The bottleneck should reflect the type of work challenges someone in their role would actually encounter.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content.strip()

        # Parse JSON response
        result = json.loads(content)

        model_info = "custom_llm" if self.llm_generate_func else self.model
        logger.info(
            f"LLM generate_all_bottlenecks ok | count={num_bottlenecks} model={model_info} duration={time.time() - _start:.2f}s"
        )

        # Parse bottlenecks from response
        bottlenecks = []
        for bottleneck_data in result.get("bottlenecks", []):
            description = bottleneck_data.get("description", "").strip()
            if description:
                artifacts = self._extract_artifacts(description)
                bottlenecks.append(
                    Bottleneck(description=description, searchable_artifacts=artifacts)
                )

        return bottlenecks

    def _generate_bottleneck(
        self,
        world_model: WorldModel,
        difficulty: ContextDifficulty,
    ) -> Bottleneck:
        """Generate exactly 1 bottleneck for a world model.

        Args:
            world_model: The world model for context
            difficulty: Context difficulty level

        Returns:
            A single generated Bottleneck
        """
        # Render prompt template
        template = self.template_env.get_template("generate_bottleneck.j2")
        prompt = template.render(
            persona_name=world_model.persona_full_name,
            persona_occupation=world_model.persona_occupation,
            persona_about=world_model.persona_about,
            difficulty=difficulty.value,
            relationships=world_model.relationships,
            org_structure=world_model.organizational_structure,
            personal_context=world_model.personal_context,
        )

        # Generate bottleneck JSON
        _start = time.time()
        response_text = ""

        try:
            if self.llm_generate_func is not None:
                # Use custom LLM function (returns JSON string)
                response_text = self.llm_generate_func(prompt)
            else:
                # Use OpenAI client directly with JSON mode
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert at creating highly specific, realistic productivity bottlenecks that reference real names, documents, and deadlines. The bottleneck must be a SINGLE, focused problem that is directly relevant to the persona's specific occupation and professional background. The bottleneck should reflect the type of work challenges someone in their role would actually encounter.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                response_text = response.choices[0].message.content

            # Parse JSON response
            data = json.loads(response_text)

            _duration = time.time() - _start
            model_info = "custom_llm" if self.llm_generate_func else self.model
            logger.info(
                f"LLM generate_bottleneck ok | model={model_info} duration={_duration:.2f}s"
            )

            # Extract and validate fields
            description = data.get("description", "").strip()
            if not description:
                raise ValueError(
                    f"Missing or empty 'description' field in response: {data}"
                )

            # Get artifacts with fallback to extraction
            artifacts = data.get("searchable_artifacts", [])
            if not artifacts:
                artifacts = self._extract_artifacts(description)

            return Bottleneck(description=description, searchable_artifacts=artifacts)

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response: {e}")
            logger.debug(
                f"Response text: {response_text[:500] if response_text else 'Empty response'}"
            )
            raise ValueError(f"LLM returned invalid JSON: {e}")
        except Exception as e:
            logger.error(f"Bottleneck generation failed: {e}")
            raise

    def _extract_artifacts(self, description: str) -> List[str]:
        """Extract mentioned artifacts from bottleneck description.

        Args:
            description: The bottleneck description

        Returns:
            List of artifact references found in the description
        """
        artifacts = []

        # Look for document references
        import re

        # Find URLs
        urls = re.findall(
            r"(?:docs\.google\.com/[\w-]+|confluence/[\w-]+)", description
        )
        artifacts.extend(urls)

        # Find ticket/request IDs
        ids = re.findall(
            r"(?:ticket #|JIRA |request #|Meeting ID: |#)([A-Z0-9-]+)", description
        )
        artifacts.extend(ids)

        # Find quoted document names
        docs = re.findall(r'"([^"]+)"', description)
        artifacts.extend([doc for doc in docs if len(doc) > 5 and len(doc) < 100])

        return list(set(artifacts))  # Remove duplicates
