"""
Distractor Generator for PersonaSim.

This module generates natural-looking workplace communications (emails, documents, calendar events)
that are contextually appropriate but do not contain information helpful for resolving bottlenecks.
"""

import json
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import BaseGenerator
from .world_model import WorldModel, DistractorDifficulty
from .bottleneck_injector import Bottleneck
from configs.unified_contracts import CorpusItem, CorpusItemType
from .interfaces.artifacts import (
    EmailPayload,
    CalendarPayload,
    DocumentPayload,
)

logger = logging.getLogger(__name__)


class DistractorGenerator(BaseGenerator):
    """
    Generates natural-looking workplace distractors based on world model context.
    """

    def __init__(
        self,
        llm_generate_func: Optional[Callable[[str], str]] = None,
        template_dir: Optional[Path] = None,
        debug_dir: Optional[Path] = None,
        max_workers: int = 8,
    ):
        """
        Initialize the distractor generator.

        Args:
            llm_generate_func: Function that takes a prompt and returns LLM response (from MultiLLMClient)
            template_dir: Directory containing Jinja2 templates
            debug_dir: Optional directory for debug output
        """
        # Set default template directory for distractors
        if template_dir is None:
            template_dir = Path(__file__).parent.parent / "prompts" / "distractors"

        super().__init__(
            template_dir=template_dir,
            template_subdir="distractors",
            llm_generate_func=llm_generate_func,
            debug_dir=debug_dir,
            max_workers=max_workers,
        )

    def generate(
        self,
        world_models: List[WorldModel],
        bottlenecks_per_persona: List[List[Bottleneck]],
        k: int = 6,
        difficulty: DistractorDifficulty = DistractorDifficulty.MEDIUM,
        **kwargs,
    ) -> List[List[CorpusItem]]:
        """
        Generate ALL distractors in a single massive parallel batch.

        Maintains non-contamination: each distractor avoids ALL bottlenecks
        from its associated world model/persona.

        Args:
            world_models: List of world models (one per persona)
            bottlenecks_per_persona: List of lists of bottlenecks (maintains persona association)
            k: Number of distractors per persona
            difficulty: Distractor difficulty level
            **kwargs: Additional arguments

        Returns:
            List of lists of CorpusItems (one list per persona)
        """
        total_tasks = len(world_models) * 3  # 3 types per persona
        logger.info(
            f"Generating {k} distractors per persona for {len(world_models)} personas "
            f"({total_tasks} parallel generation tasks) with difficulty={difficulty}"
        )

        @dataclass
        class GenerationTask:
            """Track task metadata for proper result association."""

            persona_idx: int
            kind: str  # "email", "calendar", or "document"
            count: int
            world_model: WorldModel
            bottlenecks: List[Bottleneck]

        # Create ALL generation tasks upfront
        tasks = []
        for idx, (wm, bottlenecks) in enumerate(
            zip(world_models, bottlenecks_per_persona)
        ):
            # Distribute k across types: 50% emails, 30% documents, 20% calendar
            # This matches Research_PROBE's natural distribution
            email_count = max(1, int(k * 0.5))  # 50%
            document_count = max(1, int(k * 0.3))  # 30%
            calendar_count = max(1, k - email_count - document_count)  # ~20%

            for kind, count in [
                ("email", email_count),
                ("calendar", calendar_count),
                ("document", document_count),
            ]:
                if count > 0:
                    tasks.append(
                        GenerationTask(
                            persona_idx=idx,
                            kind=kind,
                            count=count,
                            world_model=wm,
                            bottlenecks=bottlenecks,
                        )
                    )

        # Execute ALL tasks in massive parallel batch
        results_by_persona_and_type = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks at once
            future_to_task = {
                executor.submit(
                    self.generate_for_bottlenecks,
                    task.world_model,
                    task.bottlenecks,
                    task.kind,
                    task.count,
                    difficulty,
                    1.0,  # temperature
                    task.world_model.available_actions,
                ): task
                for task in tasks
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    items = future.result()

                    # Store results indexed by persona and type
                    if task.persona_idx not in results_by_persona_and_type:
                        results_by_persona_and_type[task.persona_idx] = {}
                    results_by_persona_and_type[task.persona_idx][task.kind] = items

                    completed += 1
                    logger.info(
                        f"[{completed}/{len(tasks)}] Generated {len(items)} {task.kind} "
                        f"distractors for persona {task.persona_idx + 1}"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate {task.kind} distractors for persona {task.persona_idx}: {e}"
                    )
                    raise

        # Assemble results maintaining persona order
        distractors_per_persona = []
        for idx in range(len(world_models)):
            persona_distractors = []
            if idx in results_by_persona_and_type:
                # Combine all types for this persona
                for kind in ["email", "calendar", "document"]:
                    if kind in results_by_persona_and_type[idx]:
                        persona_distractors.extend(
                            results_by_persona_and_type[idx][kind]
                        )
            distractors_per_persona.append(persona_distractors)

        total_distractors = sum(len(d) for d in distractors_per_persona)
        logger.info(
            f"✓ Generated {total_distractors} distractors across {len(world_models)} personas"
        )

        return distractors_per_persona

    def generate_for_bottlenecks(
        self,
        world_model: WorldModel,
        bottlenecks: List[Bottleneck],
        kind: str,  # "email" | "calendar" | "document"
        count: int = 6,
        difficulty: DistractorDifficulty = DistractorDifficulty.HARD,
        temperature: float = 1,
        actions: Optional[List[Any]] = None,  # Add actions parameter
    ) -> List[CorpusItem]:
        """
        Generate natural distractors that avoid ALL provided bottlenecks.
        Now generates in batches of 5 for better quality.

        Args:
            world_model: World model containing context
            bottlenecks: List of bottlenecks to avoid in distractors
            kind: Type of items to generate
            count: Number of items to generate
            difficulty: Context difficulty (affects length)
            temperature: LLM temperature
            actions: Optional list of available actions to consider for distractor generation

        Returns:
            List of CorpusItem distractors
        """
        # Determine content length based on difficulty
        if difficulty == DistractorDifficulty.EASY:
            min_words, max_words = 100, 200
        elif difficulty == DistractorDifficulty.MEDIUM:
            min_words, max_words = 300, 400
        else:  # HARD
            min_words, max_words = 500, 700

        # Generate in batches of 5 for better quality
        batch_size = 5
        all_corpus_items = []

        # Calculate number of batches needed
        batches_needed = (count + batch_size - 1) // batch_size

        # Prepare template context that will be reused
        combined_bottleneck_desc = "\n".join(
            [f"{i + 1}. {b.description}" for i, b in enumerate(bottlenecks)]
        )
        combined_bottleneck = type(
            "Bottleneck", (), {"description": combined_bottleneck_desc}
        )()

        # Use specific template based on kind
        template_name = f"generate_{kind}_distractors.j2"
        try:
            template = self.jinja_env.get_template(template_name)
        except Exception:
            logger.warning(
                f"Specific template {template_name} not found, using generic template"
            )
            template = self.jinja_env.get_template("generate_natural_distractor.j2")

        # Multi-threaded batch generation
        def generate_batch(batch_num: int, items_in_batch: int) -> List[CorpusItem]:
            """Generate a single batch of distractors."""
            logger.info(
                f"Generating batch {batch_num + 1}/{batches_needed} with {items_in_batch} {kind} distractors"
            )

            context = {
                "world_model": world_model,
                "bottleneck": combined_bottleneck,
                "kind": kind,
                "count": items_in_batch,
                "min_words": min_words,
                "max_words": max_words,
                "difficulty": difficulty.value,
                "actions": actions or [],  # Include actions in context
            }

            prompt = template.render(context)

            # Call LLM
            if self.debug_dir:
                debug_seq = self._debug_seq + batch_num + 1
                self._dump_json(
                    f"{debug_seq:03d}_natural_{kind}_batch_{batch_num}_prompt",
                    {"prompt": prompt},
                )

            try:
                response = self._call_llm_with_prompt(prompt, temperature)
                if self.debug_dir:
                    debug_seq = self._debug_seq + batch_num + 1
                    self._dump_text(
                        f"{debug_seq:03d}_natural_{kind}_batch_{batch_num}_response",
                        response,
                    )

                # Parse the response
                items = self._parse_response(response, kind)

                # Convert to CorpusItems
                batch_corpus_items = []
                for idx, item in enumerate(items):
                    corpus_item = self._create_corpus_item(
                        item, kind, len(all_corpus_items) + idx
                    )
                    if corpus_item:
                        batch_corpus_items.append(corpus_item)

                return batch_corpus_items

            except Exception as e:
                logger.error(
                    f"Failed to generate batch {batch_num + 1} of {kind} distractors: {e}"
                )
                return []

        # Generate all batches in parallel
        logger.info(f"Starting multi-threaded generation of {batches_needed} batches")

        # Determine optimal number of workers for batch generation
        num_workers = min(batches_needed, self.max_workers)
        logger.info(f"Using {num_workers} worker threads for batch generation")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all batch generation tasks
            future_to_batch = {}
            for batch_num in range(batches_needed):
                items_in_batch = min(batch_size, count - batch_num * batch_size)
                if items_in_batch > 0:
                    future = executor.submit(generate_batch, batch_num, items_in_batch)
                    future_to_batch[future] = batch_num

            # Collect results as they complete
            batch_results = {}
            for future in as_completed(future_to_batch):
                batch_num = future_to_batch[future]
                try:
                    batch_items = future.result()
                    batch_results[batch_num] = batch_items
                    logger.info(
                        f"Completed batch {batch_num + 1}/{batches_needed} with {len(batch_items)} items"
                    )
                except Exception as e:
                    logger.error(f"Batch {batch_num + 1} failed: {e}")
                    batch_results[batch_num] = []

            # Combine results in order
            for batch_num in sorted(batch_results.keys()):
                all_corpus_items.extend(batch_results[batch_num])

        # Enhance medium/hard distractors with urgency and proper length
        if difficulty in [DistractorDifficulty.MEDIUM, DistractorDifficulty.HARD]:
            all_corpus_items = self._enhance_distractors(
                all_corpus_items,
                world_model,
                bottlenecks,
                difficulty,
                temperature,
                actions,
            )

        logger.info(
            f"Generated {len(all_corpus_items)} natural {kind} distractors avoiding {len(bottlenecks)} bottlenecks"
        )
        return all_corpus_items[:count]  # Ensure we don't exceed requested count

    def _enhance_distractors(
        self,
        corpus_items: List[CorpusItem],
        world_model: WorldModel,
        bottlenecks: List[Bottleneck],
        difficulty: DistractorDifficulty,
        temperature: float,
        actions: Optional[List[Any]] = None,  # Add actions parameter
    ) -> List[CorpusItem]:
        """
        Enhance medium/hard distractors with proper urgency, length, and technical details
        while ensuring they don't contain actual bottlenecks.
        Processes in batches of 5 with multi-threading for efficiency.

        Args:
            corpus_items: List of corpus items to enhance
            world_model: World model containing context
            bottlenecks: List of bottlenecks to avoid
            difficulty: Context difficulty level
            temperature: LLM temperature
            actions: Optional list of available actions to consider

        Returns:
            List of enhanced CorpusItems
        """
        if not corpus_items:
            return corpus_items

        # Determine target word counts and urgency level
        if difficulty == DistractorDifficulty.MEDIUM:
            min_words, max_words = 300, 400
            urgency_level = "moderate"
        else:  # HARD
            min_words, max_words = 500, 700
            urgency_level = "high"

        # Process items directly in parallel (no batching needed)
        logger.info(f"Enhancing {len(corpus_items)} distractors using multi-threading")

        # Determine optimal number of workers
        num_workers = min(len(corpus_items), self.max_workers)
        logger.info(f"Using {num_workers} worker threads for enhancement")

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all enhancement tasks directly
            future_to_idx = {
                executor.submit(
                    self._enhance_single_item,
                    item,
                    world_model,
                    min_words,
                    max_words,
                    urgency_level,
                    bottlenecks,
                    actions,
                    temperature,
                    difficulty,
                ): idx
                for idx, item in enumerate(corpus_items)
            }

            # Collect results as they complete
            item_results = {}
            completed = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    item_results[idx] = future.result()
                    completed += 1
                    if completed % 10 == 0:  # Progress logging every 10 items
                        logger.info(f"Enhanced {completed}/{len(corpus_items)} items")
                except Exception as e:
                    logger.error(f"Failed to enhance item {idx}: {e}")
                    # Keep original item if enhancement fails
                    item_results[idx] = corpus_items[idx]

            # Sort results by index to maintain order
            all_enhanced_items = [item_results[i] for i in range(len(corpus_items))]

        enhanced_count = sum(
            1 for item in all_enhanced_items if item.metadata.get("enhanced")
        )
        logger.info(
            f"Enhanced {enhanced_count} out of {len(corpus_items)} {difficulty.value.lower()} distractors using template-based processing"
        )
        return all_enhanced_items

    def _parse_response(self, response: str, kind: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured items."""
        try:
            # Try to extract JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            items = json.loads(response.strip())
            if isinstance(items, dict) and "items" in items:
                items = items["items"]
            return items if isinstance(items, list) else []
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response}")
            return []

    def _create_corpus_item(
        self, item_data: Dict[str, Any], kind: str, idx: int
    ) -> Optional[CorpusItem]:
        """Create a CorpusItem from parsed data."""
        try:
            import uuid

            corpus_id = f"{kind}_{uuid.uuid4().hex[:8]}"

            if kind == "email":
                payload = EmailPayload(
                    subject=item_data.get("subject", ""),
                    sender=item_data.get("sender", ""),
                    to=item_data.get("to", []),
                    cc=item_data.get("cc", []),
                    timestamp=item_data.get("timestamp", datetime.now().isoformat()),
                    body=item_data.get("body", ""),
                )
                corpus_type = CorpusItemType.EMAIL
            elif kind == "calendar":
                payload = CalendarPayload(
                    title=item_data.get("title", ""),
                    start_time=item_data.get("start_time", datetime.now().isoformat()),
                    end_time=item_data.get("end_time", datetime.now().isoformat()),
                    location=item_data.get("location", ""),
                    attendees=item_data.get("attendees", []),
                    description=item_data.get("description", ""),
                )
                corpus_type = CorpusItemType.EVENT
            else:  # document
                payload = DocumentPayload(
                    title=item_data.get("title", ""),
                    content=item_data.get("content", ""),
                    mime=item_data.get("mime", "text/markdown"),
                )
                corpus_type = CorpusItemType.DOCUMENT

            return CorpusItem(
                id=corpus_id,
                type=corpus_type,
                payload=payload,
                created_at=datetime.now(),
            )
        except Exception as e:
            logger.error(f"Failed to create corpus item: {e}")
            return None

    def _enhance_single_item(
        self,
        item: CorpusItem,
        world_model,
        min_words: int,
        max_words: int,
        urgency_level: str,
        bottlenecks: List,
        actions: List,
        temperature: float,
        difficulty,
    ) -> CorpusItem:
        """Enhance a single item using the original individual processing method."""

        # Extract current content based on item type
        if item.type == CorpusItemType.EMAIL:
            current_content = item.payload.body
            item_type = "EMAIL"
        elif item.type == CorpusItemType.EVENT:
            current_content = item.payload.description
            item_type = "EVENT"
        else:  # DOCUMENT
            current_content = item.payload.content
            item_type = "DOCUMENT"

        # Load individual enhancement template
        enhancement_template = self.jinja_env.get_template("enhance_distractor.j2")

        # Render enhancement prompt
        prompt = enhancement_template.render(
            item_type=item_type,
            world_model=world_model,
            min_words=min_words,
            max_words=max_words,
            urgency_level=urgency_level,
            bottlenecks=bottlenecks,
            current_content=current_content,
            actions=actions or [],
        )

        enhanced_content = self._call_llm_with_prompt(prompt, temperature * 0.8)
        enhanced_content = enhanced_content.strip()

        if enhanced_content:
            # Create new payload with enhanced content
            if item.type == CorpusItemType.EMAIL:
                new_payload = EmailPayload(
                    subject=item.payload.subject,
                    sender=item.payload.sender,
                    to=item.payload.to,
                    cc=item.payload.cc,
                    timestamp=item.payload.timestamp,
                    body=enhanced_content,
                )
            elif item.type == CorpusItemType.EVENT:
                new_payload = CalendarPayload(
                    title=item.payload.title,
                    start_time=item.payload.start_time,
                    end_time=item.payload.end_time,
                    location=item.payload.location,
                    attendees=item.payload.attendees,
                    description=enhanced_content,
                )
            else:  # DOCUMENT
                new_payload = DocumentPayload(
                    title=item.payload.title,
                    content=enhanced_content,
                    mime=item.payload.mime,
                )

            # Create enhanced corpus item
            enhanced_item = CorpusItem(
                id=item.id,
                type=item.type,
                payload=new_payload,
                created_at=item.created_at,
            )
            return enhanced_item
        else:
            # Return original if enhancement failed
            return item

    def _call_llm_with_prompt(self, prompt: str, temperature: float = 1.0) -> str:
        """
        Call LLM with prompt using BaseGenerator interface.

        Args:
            prompt: The prompt to send to the LLM
            temperature: Temperature for generation (currently ignored as MultiLLMClient handles this)

        Returns:
            LLM response
        """
        try:
            return self.llm_generate_func(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise

    def _dump_json(self, name: str, data: Any) -> None:
        """Write JSON data to debug directory using BaseGenerator naming."""
        if not self.debug_dir:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            filename = self._next_debug_name(name, "json")
            path = self.debug_dir / filename
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception:
            pass

    def _dump_text(self, name: str, text: str) -> None:
        """Write raw text to debug directory using BaseGenerator naming."""
        if not self.debug_dir:
            return
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            filename = self._next_debug_name(name, "txt")
            path = self.debug_dir / filename
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
