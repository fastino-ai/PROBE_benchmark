"""
Proactive Checklist Generator for PersonaSim benchmark system.

This module implements PR12: generating three-step checklists (Search → Identification → Task Selection)
using WorldModel and Bottlenecks as primary inputs for evaluating proactive productivity workflows.
"""

import uuid
import logging
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path

from .base import BaseGenerator
from configs import (
    ProactiveChecklist,
    ChecklistItem,
    ChecklistStepType,
    ActionRegistry,
)
from .world_model import WorldModel
from .bottleneck_injector import Bottleneck

# Set up logging
logger = logging.getLogger(__name__)


class ChecklistGenerationError(Exception):
    """Raised when checklist generation fails after all retries."""

    pass


class ChecklistGenerator(BaseGenerator):
    """
    Generates three-step proactive workflow checklists for evaluating agent capabilities.

    This generator creates structured evaluation checklists that measure an agent's
    ability to complete the proactive productivity workflow: search for data sources,
    identify the problem, and select the appropriate task with parameters.

    The generator uses LLM-powered templates to create contextually appropriate
    checklists based on WorldModel and Bottlenecks.
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        llm_generate_func: Optional[Callable[[str], str]] = None,
        debug_dir: Optional[Path] = None,
        max_workers: int = 8,
    ) -> None:
        """
        Initialize the checklist generator.

        Args:
            template_dir: Directory containing Jinja2 templates. Defaults to prompts/checklist_generation/
            llm_generate_func: Function that takes a prompt string and returns LLM response (from MultiLLMClient)
            debug_dir: Optional directory for debug output
            max_workers: Maximum number of parallel workers for batch processing
        """
        super().__init__(
            template_dir=template_dir,
            template_subdir="checklist_generation",
            llm_generate_func=llm_generate_func,
            debug_dir=debug_dir,
            max_workers=max_workers,
        )

    def generate(
        self,
        world_models: List[WorldModel],
        bottlenecks_per_persona: List[List[Bottleneck]],
        action_registry: Optional[ActionRegistry] = None,
        **kwargs,
    ) -> List[List[ProactiveChecklist]]:
        """
        Generate checklists for ALL bottlenecks across ALL personas in parallel.

        Args:
            world_models: List of world models (one per persona)
            bottlenecks_per_persona: List of lists of bottlenecks (outer list matches world_models)
            action_registry: Optional action registry (uses world_model.available_actions if not provided)
            **kwargs: Additional arguments (for interface compatibility)

        Returns:
            List of lists of ProactiveChecklists (maintains persona→bottleneck association)
        """
        logger.info(
            f"Generating checklists for {sum(len(b) for b in bottlenecks_per_persona)} bottlenecks "
            f"across {len(world_models)} personas"
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Prepare all tasks
        tasks = []
        for persona_idx, (world_model, bottlenecks) in enumerate(
            zip(world_models, bottlenecks_per_persona)
        ):
            for bottleneck_idx, bottleneck in enumerate(bottlenecks):
                tasks.append(
                    (
                        persona_idx,
                        bottleneck_idx,
                        world_model,
                        bottleneck,
                        action_registry,
                    )
                )

        # Process ALL checklists in parallel
        all_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures_to_task = {
                executor.submit(self._generate_single_checklist, *task): task
                for task in tasks
            }

            # Collect results
            for future in as_completed(futures_to_task):
                task = futures_to_task[future]
                persona_idx, bottleneck_idx = task[0], task[1]

                if persona_idx not in all_results:
                    all_results[persona_idx] = {}

                try:
                    all_results[persona_idx][bottleneck_idx] = future.result()
                    logger.info(
                        f"Generated checklist for persona {persona_idx + 1}, bottleneck {bottleneck_idx + 1}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate checklist for persona {persona_idx}, bottleneck {bottleneck_idx}: {e}"
                    )
                    raise

        # Reconstruct nested list structure maintaining order
        checklists_per_persona = []
        for persona_idx in range(len(world_models)):
            persona_checklists = []
            for bottleneck_idx in range(len(bottlenecks_per_persona[persona_idx])):
                persona_checklists.append(all_results[persona_idx][bottleneck_idx])
            checklists_per_persona.append(persona_checklists)

        total_checklists = sum(len(c) for c in checklists_per_persona)
        logger.info(f"✓ Generated {total_checklists} checklists successfully")

        return checklists_per_persona

    def _generate_single_checklist(
        self,
        persona_idx: int,
        bottleneck_idx: int,
        world_model: WorldModel,
        bottleneck: Bottleneck,
        action_registry: Optional[ActionRegistry] = None,
    ) -> ProactiveChecklist:
        """Generate a single checklist."""
        checklist_id = (
            f"checklist_p{persona_idx:02d}_b{bottleneck_idx:02d}_{uuid.uuid4().hex[:8]}"
        )

        # Use registry if provided, otherwise create one from world model actions
        if action_registry is None:
            action_registry = ActionRegistry(actions=world_model.available_actions)

        # Find the ground truth action for this bottleneck (using 1-based indexing from prompt)
        # This is THE source of truth - never use LLM to determine correct action
        correct_action = None
        for action in world_model.available_actions:
            if action.solves_bottleneck == bottleneck_idx + 1:
                correct_action = action
                break

        if not correct_action:
            raise ValueError(
                f"No action found with solves_bottleneck={bottleneck_idx + 1}. "
                f"Every bottleneck must have exactly one action with matching solves_bottleneck field."
            )

        # Generate the three-step checklist using LLM with the registry
        checklist_data = self._generate_three_step_checklist_data(
            world_model=world_model,
            bottleneck=bottleneck,
            available_actions=action_registry.actions,
        )

        # Create the three required steps using ground truth action from solves_bottleneck field
        required_steps = self._create_three_steps(
            checklist_data=checklist_data,
            bottleneck=bottleneck,
            world_model=world_model,
            correct_action=correct_action,  # Pass the full action object
        )

        # Create description
        persona_name = world_model.persona_full_name
        description = (
            f"Three-step evaluation checklist for {persona_name}: "
            f"Resolving bottleneck: {bottleneck.description[:60]}..."
        )

        return ProactiveChecklist(
            checklist_id=checklist_id,
            description=description,
            required_steps=required_steps,
        )

    def _generate_three_step_checklist_data(
        self,
        world_model: WorldModel,
        bottleneck: Bottleneck,
        available_actions: List[Any],
    ) -> Dict[str, Any]:
        """Generate the three-step checklist data using LLM."""
        try:
            # Load the three-step checklist template
            template = self.jinja_env.get_template("three_step_checklist.j2")

            prompt_context = {
                "world_model": world_model,
                "bottleneck": bottleneck,
                "available_actions": available_actions,
                "difficulty": world_model.context_difficulty.value,
            }

            prompt = template.render(**prompt_context)
            response = self.llm_generate_func(prompt)

            # Parse JSON response
            import json

            # Log the response for debugging
            logger.debug(f"LLM response for checklist generation: {response[:500]}...")

            if not response or not response.strip():
                raise ValueError("Empty response from LLM")

            # Clean up response - sometimes there are extra characters
            response = response.strip()

            # Try to extract JSON if response has extra text
            if not response.startswith("{"):
                # Look for JSON content
                json_start = response.find("{")
                json_end = response.rfind("}")
                if json_start != -1 and json_end != -1:
                    response = response[json_start : json_end + 1]
                else:
                    logger.error(f"No valid JSON found in response: {response}")
                    raise ValueError(f"Invalid JSON response: {response[:200]}...")

            # Handle double braces from template rendering ({{...}})
            if response.startswith("{{") and response.endswith("}}"):
                response = response[1:-1].strip()

            return json.loads(response)

        except Exception as e:
            logger.error(f"Failed to generate three-step checklist: {e}")
            raise ChecklistGenerationError(
                f"Three-step checklist generation failed: {str(e)}"
            )

    def _create_three_steps(
        self,
        checklist_data: Dict[str, Any],
        bottleneck: Bottleneck,
        world_model: WorldModel,
        correct_action: Any,
    ) -> List[ChecklistItem]:
        """Create the three evaluation steps from the generated data.

        Args:
            checklist_data: LLM-generated checklist data (for search steps and parameters)
            bottleneck: The bottleneck being addressed
            world_model: World model context
            correct_action: Ground truth ProactiveAction object (from solves_bottleneck field)
        """
        steps = []

        # Step 1: Search - Check which data sources were returned
        search_data = checklist_data.get("search_step", {})
        search_step = ChecklistItem(
            checklist_idx="search_001",
            step_type=ChecklistStepType.RETRIEVAL,
            description=search_data.get(
                "description", "Verify correct data sources were retrieved"
            ),
            success_criteria={
                "expected_sources": search_data.get("expected_sources", []),
                "required_items": [],  # Will be populated by PR13
            },
            evidence_required=[],  # Will be populated by PR13
            previous_action_idx=None,
        )
        steps.append(search_step)

        # Step 2: Identification - Check that the correct problem was recognized
        identification_data = checklist_data.get("identification_step", {})
        identification_step = ChecklistItem(
            checklist_idx="identification_001",
            step_type=ChecklistStepType.IDENTIFICATION,
            description=identification_data.get(
                "description", "Verify the correct problem was identified"
            ),
            success_criteria={
                "target_problem": bottleneck.description,
                "problem_description": bottleneck.description,
            },
            evidence_required=[],
            previous_action_idx="search_001",
        )
        steps.append(identification_step)

        # Step 3: Task Selection - Use ground truth action from solves_bottleneck field
        # NEVER use LLM's action choice - only use solves_bottleneck field
        task_data = checklist_data.get("task_selection_step", {})

        # Log if LLM suggested a different action (for monitoring/debugging)
        llm_suggested_action = task_data.get("correct_action", "")
        if llm_suggested_action and llm_suggested_action != correct_action.id:
            logger.debug(
                f"LLM suggested action '{llm_suggested_action}' but using ground truth "
                f"action '{correct_action.id}' from solves_bottleneck field"
            )

        task_step = ChecklistItem(
            checklist_idx="task_selection_001",
            step_type=ChecklistStepType.TASK_EXECUTION,
            description=task_data.get(
                "description", "Select appropriate task and parameters"
            ),
            success_criteria={
                "correct_action": correct_action.id,  # Always from solves_bottleneck
                "correct_parameters": task_data.get("correct_parameters", {}),
            },
            evidence_required=[],
            previous_action_idx="identification_001",
        )
        steps.append(task_step)

        return steps
