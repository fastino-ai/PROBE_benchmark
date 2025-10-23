"""
True Positives Generator for PR13.

This module generates corpus items that contain evidence of bottlenecks
and injects their IDs into the checklist's search step for validation.
"""

import uuid
import logging
from typing import List, Tuple, Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import json
from pathlib import Path

from .base import BaseGenerator
from configs.unified_contracts import (
    CorpusItem,
    CorpusItemType,
    ProactiveChecklist,
)
from .interfaces.artifacts import (
    EmailPayload,
    CalendarPayload,
    DocumentPayload,
)
from .world_model import WorldModel, ContextDifficulty
from .bottleneck_injector import Bottleneck

logger = logging.getLogger(__name__)


class TruePositivesGenerator(BaseGenerator):
    """
    Generates true positive corpus items that contain evidence of specific bottlenecks.

    This generator supports coordinated batch generation where multiple documents
    are planned together to cover different aspects of a bottleneck without contamination
    from other bottlenecks.
    """

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        llm_generate_func: Optional[Callable[[str], str]] = None,
        debug_dir: Optional[Path] = None,
        max_workers: int = 8,
    ):
        """
        Initialize the true positives generator.

        Args:
            template_dir: Directory containing Jinja2 templates. Defaults to prompts/true_positives/
            llm_generate_func: Function that takes a prompt string and returns LLM response
            debug_dir: Optional directory for debug output
            max_workers: Maximum number of parallel workers for batch processing
        """
        super().__init__(
            template_dir=template_dir,
            template_subdir="true_positives",
            llm_generate_func=llm_generate_func,
            debug_dir=debug_dir,
            max_workers=max_workers,
        )

    def generate(
        self,
        world_models: List[WorldModel],
        bottlenecks_per_persona: List[List[Bottleneck]],
        checklists_per_persona: List[List[ProactiveChecklist]],
        coordinated: bool = True,
        **kwargs,
    ) -> List[List[Dict]]:
        """
        Generate true positives for ALL checklists across ALL personas in parallel.

        Args:
            world_models: List of world models (one per persona)
            bottlenecks_per_persona: List of lists of bottlenecks (maintains persona association)
            checklists_per_persona: List of lists of checklists (matches bottlenecks structure)
            coordinated: Whether to use coordinated generation to avoid contamination
            **kwargs: Additional arguments for interface compatibility

        Returns:
            List of lists of dicts with 'checklist' and 'true_positives' keys
            (maintains persona→bottleneck→checklist association)
        """
        logger.info(
            f"Generating true positives for {sum(len(c) for c in checklists_per_persona)} checklists "
            f"across {len(world_models)} personas"
        )

        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Prepare all tasks
        tasks = []
        for p_idx, (wm, bottlenecks, checklists) in enumerate(
            zip(world_models, bottlenecks_per_persona, checklists_per_persona)
        ):
            for b_idx, (bottleneck, checklist) in enumerate(
                zip(bottlenecks, checklists)
            ):
                tasks.append(
                    (p_idx, b_idx, wm, bottleneck, checklist, bottlenecks, coordinated)
                )

        # Process ALL true positives in parallel
        all_results = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures_to_task = {
                executor.submit(self._generate_single_tp, *task): task for task in tasks
            }

            # Collect results
            for future in as_completed(futures_to_task):
                task = futures_to_task[future]
                p_idx, b_idx = task[0], task[1]

                if p_idx not in all_results:
                    all_results[p_idx] = {}

                try:
                    all_results[p_idx][b_idx] = future.result()
                    logger.info(
                        f"Generated true positives for persona {p_idx + 1}, bottleneck {b_idx + 1}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate true positives for persona {p_idx}, bottleneck {b_idx}: {e}"
                    )
                    raise

        # Reconstruct nested list structure maintaining order
        results_per_persona = []
        for p_idx in range(len(world_models)):
            persona_results = []
            for b_idx in range(len(bottlenecks_per_persona[p_idx])):
                persona_results.append(all_results[p_idx][b_idx])
            results_per_persona.append(persona_results)

        total_tp = sum(
            sum(len(r["true_positives"]) for r in persona_results)
            for persona_results in results_per_persona
        )
        logger.info(f"✓ Generated {total_tp} true positives successfully")

        return results_per_persona

    def _generate_single_tp(
        self,
        persona_idx: int,
        bottleneck_idx: int,
        world_model: WorldModel,
        bottleneck: Bottleneck,
        checklist: ProactiveChecklist,
        all_persona_bottlenecks: List[Bottleneck],
        coordinated: bool,
    ) -> Dict:
        """Generate true positives for a single checklist."""
        # Prepare other bottlenecks for contamination avoidance
        other_bottlenecks = []
        if coordinated:
            # Include all other bottlenecks from this persona
            other_bottlenecks = [
                b for i, b in enumerate(all_persona_bottlenecks) if i != bottleneck_idx
            ]

        # Call the existing single-item generation logic
        updated_checklist, true_positives = self._generate_single(
            checklist=checklist,
            world_model=world_model,
            bottleneck=bottleneck,
            other_bottlenecks=other_bottlenecks,
        )

        return {"checklist": updated_checklist, "true_positives": true_positives}

    def _generate_single(
        self,
        checklist: ProactiveChecklist,
        world_model: WorldModel,
        bottleneck: Bottleneck,
        other_bottlenecks: Optional[List[Bottleneck]] = None,
    ) -> Tuple[ProactiveChecklist, List[CorpusItem]]:
        """
        Internal method for single checklist generation (extracted from old generate).
        """
        # Get search step for ID injection later
        search_step = next(
            (
                step
                for step in checklist.required_steps
                if step.checklist_idx == "search_001"
            ),
            None,
        )
        if not search_step:
            raise ValueError("No search step found in checklist")

        # Generate appropriate source types dynamically based on bottleneck and difficulty
        difficulty = world_model.context_difficulty

        # Define min/max sources per difficulty
        source_limits = {
            ContextDifficulty.EASY: {"min": 1, "max": 2},
            ContextDifficulty.MEDIUM: {"min": 2, "max": 3},
            ContextDifficulty.HARD: {"min": 3, "max": 6},
        }

        limits = source_limits[difficulty]

        # Generate dynamic source types based on bottleneck characteristics
        expected_sources = self._generate_dynamic_source_types(
            bottleneck=bottleneck,
            world_model=world_model,
            min_sources=limits["min"],
            max_sources=limits["max"],
        )

        # Step 1: Create a unified coordinated plan for all evidence distribution
        evidence_plan = self._create_unified_evidence_plan(
            bottleneck=bottleneck,
            world_model=world_model,
            expected_sources=expected_sources,
            other_bottlenecks=other_bottlenecks or [],
        )

        # Step 2: Generate corpus items according to the unified plan
        true_positives = []
        corpus_item_ids = []

        for source_plan in evidence_plan["source_plans"]:
            source = source_plan["source"]
            evidence_role = source_plan["evidence_role"]
            document_type = source_plan["document_type"]  # email, calendar, or document

            # Generate corpus item based on planned document type
            if document_type == "email":
                corpus_item = self._generate_email_evidence_unified(
                    bottleneck=bottleneck,
                    world_model=world_model,
                    evidence_role=evidence_role,
                    other_bottlenecks=other_bottlenecks or [],
                )
            elif document_type == "calendar":
                corpus_item = self._generate_calendar_evidence_unified(
                    bottleneck=bottleneck,
                    world_model=world_model,
                    evidence_role=evidence_role,
                    other_bottlenecks=other_bottlenecks or [],
                )
            elif document_type == "document":
                corpus_item = self._generate_document_evidence_unified(
                    bottleneck=bottleneck,
                    world_model=world_model,
                    evidence_role=evidence_role,
                    other_bottlenecks=other_bottlenecks or [],
                )
            else:
                # Fallback to document type
                corpus_item = self._generate_document_evidence_unified(
                    bottleneck=bottleneck,
                    world_model=world_model,
                    evidence_role=evidence_role,
                    other_bottlenecks=other_bottlenecks or [],
                )

            # Use source name for ID traceability
            source_type = source.split("_")[0]
            source_type_for_id = (
                "document" if source_type not in ["email", "calendar"] else source_type
            )
            corpus_item.id = f"{source_type_for_id}_{uuid.uuid4().hex[:8]}"

            true_positives.append(corpus_item)
            corpus_item_ids.append(corpus_item.id)

        # Step 3: Review and refine the evidence to ensure coordination and no contamination
        true_positives = self._review_and_refine_evidence_unified(
            true_positives=true_positives,
            world_model=world_model,
            bottleneck=bottleneck,
            difficulty=difficulty,
            evidence_plan=evidence_plan,
            other_bottlenecks=other_bottlenecks or [],
        )

        # Inject true positive IDs into checklist search step
        search_step.success_criteria["required_items"] = corpus_item_ids
        search_step.evidence_required = corpus_item_ids

        logger.info(
            f"Generated {len(true_positives)} coordinated true positive corpus items using unified planning"
        )
        logger.info(f"Injected IDs into checklist: {corpus_item_ids}")

        return checklist, true_positives

    def _generate_dynamic_source_types(
        self,
        bottleneck: Bottleneck,
        world_model: WorldModel,
        min_sources: int,
        max_sources: int,
    ) -> List[str]:
        """
        Generate appropriate source types dynamically based on bottleneck characteristics.

        This analyzes the bottleneck description and world model context to determine
        what types of evidence sources would realistically contain information about
        this specific bottleneck.

        Args:
            bottleneck: The bottleneck to analyze
            world_model: World model context
            min_sources: Minimum number of sources required
            max_sources: Maximum number of sources allowed

        Returns:
            List of dynamically generated source type names
        """
        try:
            # Load template for dynamic source generation
            template = self.jinja_env.get_template("generate_dynamic_sources.j2")
            prompt = template.render(
                bottleneck=bottleneck,
                world_model=world_model,
                min_sources=min_sources,
                max_sources=max_sources,
            )

            response = self.llm_generate_func(prompt)
            sources_data = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )

            # Extract source types from response
            dynamic_sources = sources_data.get("source_types", [])

            # Validate count
            if len(dynamic_sources) < min_sources:
                logger.warning(
                    f"Generated only {len(dynamic_sources)} sources, need at least {min_sources}"
                )
                # Add fallback sources
                fallback_sources = [
                    "team_communications",
                    "project_documents",
                    "email_updates",
                    "meeting_notes",
                    "status_reports",
                ]
                while len(dynamic_sources) < min_sources and fallback_sources:
                    source = fallback_sources.pop(0)
                    if source not in dynamic_sources:
                        dynamic_sources.append(source)

            elif len(dynamic_sources) > max_sources:
                logger.info(
                    f"Generated {len(dynamic_sources)} sources, limiting to {max_sources}"
                )
                dynamic_sources = dynamic_sources[:max_sources]

            logger.info(
                f"Generated {len(dynamic_sources)} dynamic source types: {dynamic_sources}"
            )
            return dynamic_sources

        except Exception:
            logger.warning("Dynamic source generation failed")

    def _create_unified_evidence_plan(
        self,
        bottleneck: Bottleneck,
        world_model: WorldModel,
        expected_sources: List[str],
        other_bottlenecks: List[Bottleneck],
    ) -> Dict[str, Any]:
        """
        Create a unified coordinated plan for distributing bottleneck evidence across
        different document types (emails, calendars, documents).

        This planning step determines:
        1. Which document type each source should be (email/calendar/document)
        2. What specific evidence role each document should fulfill
        3. How the evidence should be distributed to require correlation

        Args:
            bottleneck: The target bottleneck to create evidence for
            world_model: World model context
            expected_sources: List of source types to generate
            other_bottlenecks: Other bottlenecks to avoid contaminating

        Returns:
            Unified evidence plan dictionary with source assignments and document types
        """
        # Load and render the unified planning template
        template = self.jinja_env.get_template("plan_evidence_distribution.j2")

        # Prepare other bottlenecks descriptions for avoidance
        other_bottleneck_descriptions = (
            [f"- {b.description}" for b in other_bottlenecks]
            if other_bottlenecks
            else ["None"]
        )

        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            expected_sources=expected_sources,
            other_bottlenecks=other_bottleneck_descriptions,
            difficulty=world_model.context_difficulty.value,
        )

        try:
            response = self.llm_generate_func(prompt)
            evidence_plan = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )

            # Validate the plan structure
            if "source_plans" not in evidence_plan:
                raise ValueError("Evidence plan missing 'source_plans' key")

            # Validate that each source plan has a document_type
            for plan in evidence_plan["source_plans"]:
                if "document_type" not in plan:
                    # If missing, assign based on source name
                    source = plan["source"]
                    if "email" in source.lower():
                        plan["document_type"] = "email"
                    elif (
                        "calendar" in source.lower()
                        or "event" in source.lower()
                        or "meeting" in source.lower()
                    ):
                        plan["document_type"] = "calendar"
                    else:
                        plan["document_type"] = "document"

            # Ensure we have a plan for each expected source
            planned_sources = {plan["source"] for plan in evidence_plan["source_plans"]}
            expected_sources_set = set(expected_sources)

            if planned_sources != expected_sources_set:
                logger.warning(
                    f"Plan sources {planned_sources} don't match expected {expected_sources_set}"
                )
                # Add missing sources with generic roles
                for missing_source in expected_sources_set - planned_sources:
                    evidence_plan["source_plans"].append(
                        {
                            "source": missing_source,
                            "evidence_role": f"Provide supporting context for {bottleneck.description}",
                            "key_elements": [
                                "General workplace context",
                                "Indirect references",
                            ],
                            "document_type": "document",  # Default to document type
                        }
                    )

            logger.info(
                f"Created unified evidence plan with {len(evidence_plan['source_plans'])} source assignments"
            )
            return evidence_plan

        except Exception as e:
            logger.warning(
                f"Unified evidence planning failed, using fallback plan: {e}"
            )
            # Create a simple fallback plan with document type assignments
            fallback_plans = []
            for i, source in enumerate(expected_sources):
                # Determine document type
                if "email" in source.lower():
                    document_type = "email"
                elif "calendar" in source.lower() or "event" in source.lower():
                    document_type = "calendar"
                else:
                    type_options = ["document", "email", "calendar"]
                    document_type = type_options[i % len(type_options)]

                fallback_plans.append(
                    {
                        "source": source,
                        "evidence_role": f"Provide evidence for: {bottleneck.description}",
                        "key_elements": ["Direct evidence", "Contextual clues"],
                        "document_type": document_type,
                    }
                )

            return {
                "source_plans": fallback_plans,
                "coordination_strategy": "Each source provides partial evidence requiring correlation",
            }

    def _review_and_refine_evidence(
        self,
        true_positives: List[CorpusItem],
        world_model: WorldModel,
        bottleneck: Bottleneck,
        difficulty: ContextDifficulty,
    ) -> List[CorpusItem]:
        """
        Review and refine the generated evidence to ensure:
        1. Collective sufficiency with world model
        2. Individual insufficiency
        3. Appropriate difficulty level
        """
        # Prepare evidence items for review
        evidence_items = []
        for tp in true_positives:
            item = {"type": tp.type.value, "id": tp.id}

            if tp.type == CorpusItemType.EMAIL:
                item["subject"] = tp.payload.subject
                item["content"] = tp.payload.body
            elif tp.type == CorpusItemType.EVENT:
                item["title"] = tp.payload.title
                item["description"] = tp.payload.description
            elif tp.type == CorpusItemType.DOCUMENT:
                item["title"] = tp.payload.title
                item["content"] = tp.payload.content

            evidence_items.append(item)

        # Load and render the review template
        template = self.jinja_env.get_template("review_evidence.j2")
        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            evidence_items=evidence_items,
            difficulty=difficulty.value,
        )

        try:
            response = self.llm_generate_func(prompt)
            response = response.replace("```json", "").replace("```", "")
            review_result = json.loads(response)

            if review_result.get("needs_refinement", False):
                logger.info(
                    f"Evidence needs refinement: {review_result.get('refinement_reason')}"
                )

                # Apply refinements
                for refinement in review_result.get("refined_evidence", []):
                    idx = refinement["index"]
                    if 0 <= idx < len(true_positives):
                        tp = true_positives[idx]
                        refined_content = refinement["refined_content"]

                        # Update the corpus item with refined content
                        if tp.type == CorpusItemType.EMAIL:
                            tp.payload.subject = refined_content.get(
                                "subject", tp.payload.subject
                            )
                            tp.payload.body = refined_content.get(
                                "content", tp.payload.body
                            )
                        elif tp.type == CorpusItemType.EVENT:
                            tp.payload.title = refined_content.get(
                                "title", tp.payload.title
                            )
                            tp.payload.description = refined_content.get(
                                "description", tp.payload.description
                            )
                        elif tp.type == CorpusItemType.DOCUMENT:
                            tp.payload.title = refined_content.get(
                                "title", tp.payload.title
                            )
                            tp.payload.content = refined_content.get(
                                "content", tp.payload.content
                            )

                logger.info(
                    f"Applied {len(review_result.get('refined_evidence', []))} refinements"
                )
            else:
                logger.info("Evidence meets all criteria, no refinement needed")

        except Exception as e:
            logger.warning(f"Evidence review failed, using original evidence: {e}")

        return true_positives

    def _review_and_refine_evidence_unified(
        self,
        true_positives: List[CorpusItem],
        world_model: WorldModel,
        bottleneck: Bottleneck,
        difficulty: ContextDifficulty,
        evidence_plan: Dict[str, Any],
        other_bottlenecks: List[Bottleneck],
    ) -> List[CorpusItem]:
        """
        Enhanced review and refinement that ensures coordination and no contamination.

        This method checks:
        1. Collective sufficiency with world model
        2. Individual insufficiency
        3. Appropriate difficulty level
        4. No contamination from other bottlenecks
        5. Proper coordination according to the evidence plan
        """
        # Prepare evidence items for review
        evidence_items = []
        for tp in true_positives:
            item = {"type": tp.type.value, "id": tp.id}

            if tp.type == CorpusItemType.EMAIL:
                item["subject"] = tp.payload.subject
                item["content"] = tp.payload.body
            elif tp.type == CorpusItemType.EVENT:
                item["title"] = tp.payload.title
                item["description"] = tp.payload.description
            elif tp.type == CorpusItemType.DOCUMENT:
                item["title"] = tp.payload.title
                item["content"] = tp.payload.content

            evidence_items.append(item)

        # Prepare other bottlenecks for contamination check
        other_bottleneck_descriptions = (
            [f"- {b.description}" for b in other_bottlenecks]
            if other_bottlenecks
            else ["None"]
        )

        # Load and render the enhanced review template
        template = self.jinja_env.get_template("review_evidence.j2")
        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            evidence_items=evidence_items,
            difficulty=difficulty.value,
            evidence_plan=evidence_plan,
            other_bottlenecks=other_bottleneck_descriptions,
        )

        try:
            response = self.llm_generate_func(prompt)
            review_result = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )

            # Check for bias issues
            bias_issues = review_result.get("bias_issues", {})
            if any(bias_issues.values()):
                bias_problems = [k for k, v in bias_issues.items() if v]
                logger.warning(f"Bias issues detected: {bias_problems}")
                # Force refinement if bias issues found
                review_result["needs_refinement"] = True
                review_result["refinement_reason"] = (
                    f"Bias issues: {', '.join(bias_problems)}"
                )

            if review_result.get("needs_refinement", False):
                logger.info(
                    f"Evidence needs refinement: {review_result.get('refinement_reason')}"
                )

                # Apply refinements
                for refinement in review_result.get("refined_evidence", []):
                    idx = refinement["index"]
                    if 0 <= idx < len(true_positives):
                        tp = true_positives[idx]
                        refined_content = refinement["refined_content"]

                        # Update the corpus item with refined content
                        if tp.type == CorpusItemType.EMAIL:
                            tp.payload.subject = refined_content.get(
                                "subject", tp.payload.subject
                            )
                            tp.payload.body = refined_content.get(
                                "content", tp.payload.body
                            )
                        elif tp.type == CorpusItemType.EVENT:
                            tp.payload.title = refined_content.get(
                                "title", tp.payload.title
                            )
                            tp.payload.description = refined_content.get(
                                "description", tp.payload.description
                            )
                        elif tp.type == CorpusItemType.DOCUMENT:
                            tp.payload.title = refined_content.get(
                                "title", tp.payload.title
                            )
                            tp.payload.content = refined_content.get(
                                "content", tp.payload.content
                            )

                logger.info(
                    f"Applied {len(review_result.get('refined_evidence', []))} refinements"
                )
            else:
                logger.info("Evidence meets all criteria, no refinement needed")

            # Check for contamination warnings
            if review_result.get("contamination_detected", False):
                logger.warning(
                    f"Potential contamination detected: {review_result.get('contamination_details', 'Unknown')}"
                )

        except Exception as e:
            logger.warning(
                f"Enhanced evidence review failed, using original evidence: {e}"
            )

        return true_positives

    def _generate_email_evidence_unified(
        self,
        bottleneck: Bottleneck,
        world_model: WorldModel,
        evidence_role: str,
        other_bottlenecks: List[Bottleneck],
    ) -> CorpusItem:
        """Generate an email corpus item with unified coordinated evidence and contamination prevention."""
        # Determine email length based on difficulty
        difficulty = world_model.context_difficulty
        if difficulty == ContextDifficulty.EASY:
            min_words, max_words = 100, 200
        elif difficulty == ContextDifficulty.MEDIUM:
            min_words, max_words = 300, 400
        else:  # HARD
            min_words, max_words = 500, 700

        # Prepare other bottlenecks for avoidance
        other_bottleneck_descriptions = (
            [f"- {b.description}" for b in other_bottlenecks]
            if other_bottlenecks
            else ["None"]
        )

        # Load and render the email template
        template = self.jinja_env.get_template("generate_email_evidence.j2")
        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            evidence_role=evidence_role,
            other_bottlenecks=other_bottleneck_descriptions,
            min_words=min_words,
            max_words=max_words,
        )

        try:
            response = self.llm_generate_func(prompt)
            email_data = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )
            # Validate required fields
            for field in ["subject", "sender", "content", "timestamp", "to", "cc"]:
                if field not in email_data:
                    raise ValueError(f"Missing required field: {field}")

            # Ensure to/cc are lists
            if not isinstance(email_data.get("to", []), list):
                email_data["to"] = [email_data["to"]] if email_data.get("to") else []
            if not isinstance(email_data.get("cc", []), list):
                email_data["cc"] = [email_data["cc"]] if email_data.get("cc") else []
        except Exception as e:
            logger.warning(f"Unified LLM generation failed, using fallback: {e}")
            # Fallback to template-based generation
            email_data = {
                "subject": f"Update on {world_model.organizational_structure.department} Project",
                "sender": f"{world_model.relationships[0].name if world_model.relationships else 'colleague'}@{world_model.organizational_structure.company_name.lower().replace(' ', '')}.com",
                "to": [
                    f"{rel.name}@{world_model.organizational_structure.company_name.lower().replace(' ', '')}.com"
                    for rel in world_model.relationships[1:3]
                    if len(world_model.relationships) > 1
                ]
                or ["team@company.com"],
                "cc": [
                    f"{rel.name}@{world_model.organizational_structure.company_name.lower().replace(' ', '')}.com"
                    for rel in world_model.relationships[3:4]
                    if len(world_model.relationships) > 3
                ]
                or [],
                "content": f"Hi,\n\nJust following up on our discussion. {evidence_role}\n\nLet me know your thoughts.\n\nBest regards",
                "timestamp": datetime.now().isoformat(),
            }

        corpus_id = f"email_{uuid.uuid4().hex[:8]}"

        return CorpusItem(
            id=corpus_id,
            type=CorpusItemType.EMAIL,
            payload=EmailPayload(
                subject=email_data["subject"],
                sender=email_data["sender"],
                to=email_data.get("to", []),
                cc=email_data.get("cc", []),
                timestamp=email_data["timestamp"],
                body=email_data["content"],
            ),
            created_at=datetime.now(),
        )

    def _generate_calendar_evidence_unified(
        self,
        bottleneck: Bottleneck,
        world_model: WorldModel,
        evidence_role: str,
        other_bottlenecks: List[Bottleneck],
    ) -> CorpusItem:
        """Generate a calendar corpus item with coordinated evidence and contamination prevention."""
        # Prepare other bottlenecks for avoidance
        other_bottleneck_descriptions = (
            [f"- {b.description}" for b in other_bottlenecks]
            if other_bottlenecks
            else ["None"]
        )

        # Load and render the calendar template
        template = self.jinja_env.get_template("generate_calendar_evidence.j2")
        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            evidence_role=evidence_role,
            other_bottlenecks=other_bottleneck_descriptions,
        )

        try:
            response = self.llm_generate_func(prompt)
            calendar_data = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )
            # Validate required fields
            for field in ["title", "description", "start_time", "end_time"]:
                if field not in calendar_data:
                    raise ValueError(f"Missing required field: {field}")
        except Exception as e:
            logger.warning(
                f"Coordinated calendar generation failed, using fallback: {e}"
            )
            # Fallback to template-based generation
            calendar_data = {
                "title": f"{world_model.organizational_structure.department} Meeting",
                "description": f"Discussion about project status. {evidence_role}",
                "start_time": (datetime.now() + timedelta(days=1)).isoformat(),
                "end_time": (datetime.now() + timedelta(days=1, hours=1)).isoformat(),
                "location": "Conference Room A",
                "attendees": [world_model.persona_full_name],
            }

        corpus_id = f"calendar_{uuid.uuid4().hex[:8]}"

        return CorpusItem(
            id=corpus_id,
            type=CorpusItemType.EVENT,
            payload=CalendarPayload(
                title=calendar_data["title"],
                description=calendar_data["description"],
                start_time=calendar_data["start_time"],
                end_time=calendar_data["end_time"],
                location=calendar_data.get("location", "TBD"),
                attendees=calendar_data.get(
                    "attendees", [world_model.persona_full_name]
                ),
            ),
            created_at=datetime.now(),
        )

    def _generate_document_evidence_unified(
        self,
        bottleneck: Bottleneck,
        world_model: WorldModel,
        evidence_role: str,
        other_bottlenecks: List[Bottleneck],
    ) -> CorpusItem:
        """Generate a document corpus item with coordinated evidence and contamination prevention."""
        # Determine document length based on difficulty
        difficulty = world_model.context_difficulty
        if difficulty == ContextDifficulty.EASY:
            min_words, max_words = 200, 400
        elif difficulty == ContextDifficulty.MEDIUM:
            min_words, max_words = 500, 800
        else:  # HARD
            min_words, max_words = 800, 1200

        # Prepare other bottlenecks for avoidance
        other_bottleneck_descriptions = (
            [f"- {b.description}" for b in other_bottlenecks]
            if other_bottlenecks
            else ["None"]
        )

        # Load and render the document template
        template = self.jinja_env.get_template("generate_document_evidence.j2")
        prompt = template.render(
            bottleneck=bottleneck,
            world_model=world_model,
            evidence_role=evidence_role,
            other_bottlenecks=other_bottleneck_descriptions,
            min_words=min_words,
            max_words=max_words,
        )

        try:
            response = self.llm_generate_func(prompt)
            document_data = json.loads(
                response.replace("```json", "").replace("```", "").strip()
            )

            # Auto-add mime field if missing (common LLM oversight)
            if "mime" not in document_data:
                document_data["mime"] = "text/markdown"
                logger.debug("Auto-added missing 'mime' field to document data")

            # Validate required fields
            for field in ["title", "content", "author", "created_date", "mime"]:
                if field not in document_data:
                    raise ValueError(f"Missing required field: {field}")
        except Exception as e:
            logger.warning(
                f"Coordinated document generation failed, using fallback: {e}"
            )
            # Fallback to template-based generation
            document_data = {
                "title": f"{world_model.organizational_structure.department} Project Notes",
                "content": f"Project status update:\n\n{evidence_role}\n\nNext steps to be determined.",
                "author": world_model.persona_full_name,
                "created_date": datetime.now().isoformat(),
                "mime": "text/markdown",
            }

        corpus_id = f"document_{uuid.uuid4().hex[:8]}"

        return CorpusItem(
            id=corpus_id,
            type=CorpusItemType.DOCUMENT,
            payload=DocumentPayload(
                title=document_data["title"],
                content=document_data["content"],
                author=document_data.get("author", world_model.persona_full_name),
                created_date=document_data.get(
                    "created_date", datetime.now().isoformat()
                ),
                mime=document_data.get("mime", "text/markdown"),
            ),
            created_at=datetime.now(),
        )