#!/usr/bin/env python3
"""
Scoring module for PersonaSim agent evaluations.

This module provides scoring functionality that evaluates:
1. Retrieved data points against expected sources
2. Identified bottleneck against the correct bottleneck
3. Selected task execution against the correct action and parameters

Uses both exact matching and LLM-as-judge for nuanced evaluation.
"""

import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from jinja2 import Environment, FileSystemLoader

from annotation.annotation_format import (
    Annotation,
    load_annotations_batch,
)
from evaluation.llm_json_postprocessor import (
    create_robust_json_parser,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    """Results from scoring an agent's performance."""

    retrieval_score: float  # 0.0 to 1.0
    retrieval_details: Dict[str, Any]
    identification_score: float  # 0.0 to 1.0
    identification_details: Dict[str, Any]
    task_selection_score: float  # 0.0 to 1.0
    task_selection_details: Dict[str, Any]
    overall_score: float  # 0.0 to 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "retrieval_score": self.retrieval_score,
            "retrieval_details": self.retrieval_details,
            "identification_score": self.identification_score,
            "identification_details": self.identification_details,
            "task_selection_score": self.task_selection_score,
            "task_selection_details": self.task_selection_details,
            "overall_score": self.overall_score,
        }


class ChecklistScorer:
    """Scores agent performance against evaluation checklists."""

    def __init__(
        self,
        llm_generate_func: Optional[callable] = None,
        template_dir: Optional[Path] = None,
    ):
        """
        Initialize the scorer.

        Args:
            llm_generate_func: Function that takes a prompt and returns LLM response
            template_dir: Directory containing scoring prompt templates
        """
        self.llm_generate_func = llm_generate_func

        # Set up template directory
        if template_dir is None:
            template_dir = Path(__file__).parent / "prompts" / "scoring"

        self.template_env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            trim_blocks=True,
            lstrip_blocks=True,
        )

        # Initialize JSON post-processor for LLM responses
        self.json_processor = (
            create_robust_json_parser(llm_generate_func) if llm_generate_func else None
        )

    def score_annotation(
        self,
        annotation: Annotation,
        output_data: Dict[str, Any],
        input_data: Optional[Dict[str, Any]] = None,
    ) -> ScoringResult:
        """
        Score an annotation against ground truth.

        Args:
            annotation: The annotation to score
            output_data: Ground truth output data
            input_data: Optional input data for additional context

        Returns:
            ScoringResult with detailed scores
        """
        # Extract components from output data
        world_model = output_data.get("world_model", {})
        bottleneck = output_data.get("bottleneck", {})
        checklist = output_data.get("checklist", {})
        true_positive_ids = set(output_data.get("true_positive_ids", []))

        # Score retrieval
        retrieval_score, retrieval_details = self._score_retrieval(
            annotation.retrieved_document_ids, true_positive_ids
        )

        # Score identification
        identification_score, identification_details = self._score_identification(
            annotation.bottleneck_description, bottleneck, checklist, world_model
        )

        # Score task selection
        task_score, task_details = self._score_task_selection(
            annotation.action_selection, checklist, world_model, bottleneck
        )

        # Calculate overall score (weighted average)
        weights = {"retrieval": 0.334, "identification": 0.333, "task_selection": 0.333}

        overall_score = (
            retrieval_score * weights["retrieval"]
            + identification_score * weights["identification"]
            + task_score * weights["task_selection"]
        )

        return ScoringResult(
            retrieval_score=retrieval_score,
            retrieval_details=retrieval_details,
            identification_score=identification_score,
            identification_details=identification_details,
            task_selection_score=task_score,
            task_selection_details=task_details,
            overall_score=overall_score,
        )

    def _score_retrieval(
        self,
        retrieved_ids: List[str],
        true_positive_ids: set,
    ) -> Tuple[float, Dict[str, Any]]:
        """Score retrieval based on document IDs."""
        retrieved_set = set(retrieved_ids)

        if not true_positive_ids:
            return 0.0, {"error": "No true positive IDs defined"}

        # Calculate metrics
        true_positives = len(true_positive_ids.intersection(retrieved_set))
        false_positives = len(retrieved_set - true_positive_ids)
        false_negatives = len(true_positive_ids - retrieved_set)

        precision = true_positives / len(retrieved_set) if retrieved_set else 0.0
        recall = true_positives / len(true_positive_ids) if true_positive_ids else 0.0

        # F1 score
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        details = {
            "expected_ids": list(true_positive_ids),
            "retrieved_ids": list(retrieved_set),
            "missing_ids": list(true_positive_ids - retrieved_set),
            "extra_ids": list(retrieved_set - true_positive_ids),
            "true_positives": true_positives,
            "false_positives": false_positives,
            "false_negatives": false_negatives,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score,
        }

        return f1_score, details

    def _score_identification(
        self,
        identified_bottleneck: str,
        actual_bottleneck: Dict[str, Any],
        checklist: Dict[str, Any],
        world_model: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score bottleneck identification using LLM as judge with enforced two-tier precision system."""
        if not self.llm_generate_func:
            # Fallback to exact match
            actual_description = actual_bottleneck.get("description", "")
            is_match = (
                identified_bottleneck.strip().lower()
                == actual_description.strip().lower()
            )
            return float(is_match), {"method": "exact_match", "match": is_match}

        # Use LLM to judge if the identification is correct with two-tier evaluation
        template = self.template_env.get_template("judge_identification.j2")
        prompt = template.render(
            identified_bottleneck=identified_bottleneck,
            actual_bottleneck=actual_bottleneck.get("description", ""),
            world_model=world_model,
        )

        try:
            response = self.llm_generate_func(prompt).replace("```json", "").replace("```", "").strip()
            result = json.loads(response)

            # Extract essential and non-essential detail analyses
            essential_analysis = result.get("essential_details_analysis", {})
            non_essential_analysis = result.get("non_essential_details_analysis", {})
            llm_judgment = result.get("judgment", "INCORRECT")

            # Enforce two-tier precision scoring rules
            score, enforced_judgment = self._enforce_two_tier_scoring(
                essential_analysis, non_essential_analysis, llm_judgment
            )

            # Log when enforcement overrides LLM judgment
            if llm_judgment != enforced_judgment:
                logger.debug(
                    f"Two-tier enforcement override: LLM said '{llm_judgment}' but enforced '{enforced_judgment}'"
                )

            details = {
                "method": "llm_judge_two_tier_enforced",
                "identified": identified_bottleneck,
                "expected": actual_bottleneck.get("description", ""),
                "llm_judgment": llm_judgment,  # Original LLM judgment
                "enforced_judgment": enforced_judgment,  # Final enforced judgment
                "essential_details_analysis": essential_analysis,
                "non_essential_details_analysis": non_essential_analysis,
                "reasoning": result.get("reasoning", ""),
                "score": score,
                "enforcement_override": llm_judgment != enforced_judgment,
            }

            return score, details

        except Exception as e:
            logger.error(f"Error in LLM identification scoring: {e}")
            return 0.0, {"error": str(e), "method": "llm_judge_failed"}

    def _enforce_two_tier_scoring(
        self,
        essential_analysis: Dict[str, str],
        non_essential_analysis: Dict[str, str],
        llm_judgment: str,
    ) -> Tuple[float, str]:
        """
        Enforce the two-tier precision scoring rules:
        - CORRECT (1.0): All essential + all non-essential correct
        - PARTIALLY_CORRECT (0.5): All essential correct BUT errors in non-essential
        - INCORRECT (0.0): Any essential detail wrong/missing
        """
        # Define essential detail keys
        essential_keys = ["who_blocked", "who_blocker", "what_task", "why_root_cause"]

        # Check if all essential details are correct
        all_essential_correct = True
        for key in essential_keys:
            status = essential_analysis.get(key, "missing").lower()
            if status not in ["correct"]:
                all_essential_correct = False
                break

        # If any essential detail is wrong/missing -> INCORRECT
        if not all_essential_correct:
            return 0.0, "INCORRECT"

        # All essential details are correct, now check non-essential
        non_essential_keys = [
            "when_deadline",
            "where_system",
            "how_mechanism",
            "impact_scope",
        ]
        all_non_essential_correct = True

        for key in non_essential_keys:
            status = non_essential_analysis.get(key, "missing").lower()
            # "n/a" is considered correct (not applicable)
            if status not in ["correct", "n/a"]:
                all_non_essential_correct = False
                break

        # Determine final score based on two-tier rules
        if all_essential_correct and all_non_essential_correct:
            return 1.0, "CORRECT"
        elif all_essential_correct:  # But some non-essential errors
            return 0.5, "PARTIALLY_CORRECT"
        else:
            # This shouldn't happen given our logic above, but safety fallback
            return 0.0, "INCORRECT"

    def _score_task_selection(
        self,
        action_selection: Any,  # ActionSelection object
        checklist: Dict[str, Any],
        world_model: Dict[str, Any],
        bottleneck: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Score task selection from annotation format."""
        # Get expected action from checklist
        task_step = None
        for step in checklist.get("required_steps", []):
            if step.get("step_type") == "task_execution":
                task_step = step
                break

        if not task_step:
            return 0.0, {"error": "No task execution step found in checklist"}

        expected_action = task_step.get("success_criteria", {}).get(
            "correct_action", ""
        )
        expected_params = task_step.get("success_criteria", {}).get(
            "correct_parameters", {}
        )

        # Score action selection (binary)
        action_correct = action_selection.name == expected_action
        action_score = 1.0 if action_correct else 0.0

        # Score parameters
        if action_correct and self.llm_generate_func:
            param_score, param_details = self._score_parameters_with_llm(
                action_selection.schema,
                expected_params,
                world_model,
                bottleneck,
                {
                    "action_id": action_selection.name,
                    "parameters": action_selection.schema,
                },
            )
        else:
            # Fallback to exact match
            param_score = 1.0 if action_selection.schema == expected_params else 0.0
            param_details = {
                "method": "exact_match",
                "match": action_selection.schema == expected_params,
            }

        total_score = min(action_score, param_score)

        details = {
            "action_correct": action_correct,
            "selected_action": action_selection.name,
            "expected_action": expected_action,
            "parameter_score": param_score,
            "parameter_details": param_details,
        }

        return total_score, details

    def _score_parameters_with_llm(
        self,
        selected_params: Dict[str, Any],
        expected_params: Dict[str, Any],
        world_model: Dict[str, Any],
        bottleneck: Dict[str, Any],
        selected_action: Dict[str, Any],
    ) -> Tuple[float, Dict[str, Any]]:
        """Use LLM to score parameter selection."""
        template = self.template_env.get_template("judge_parameters.j2")
        prompt = template.render(
            selected_parameters=selected_params,
            expected_parameters=expected_params,
            world_model=world_model,
            bottleneck=bottleneck,
            selected_action=selected_action,
            true_positives=[],  # Not needed for parameter scoring
        )

        try:
            response = self.llm_generate_func(prompt)
            result = json.loads(response.replace("```json", "").replace("```", "").strip())

            # Map judgment to score
            judgment = result.get("judgment", "INCORRECT")
            score_map = {"CORRECT": 1.0, "PARTIALLY_CORRECT": 0.5, "INCORRECT": 0.0}
            score = score_map.get(judgment, 0.0)

            details = {
                "method": "llm_judge",
                "selected": selected_params,
                "expected": expected_params,
                "judgment": judgment,
                "reasoning": result.get("reasoning", ""),
            }

            return score, details

        except Exception as e:
            logger.error(f"Error in LLM parameter scoring: {e}")
            # Fallback to exact match
            is_match = selected_params == expected_params
            return float(is_match), {
                "error": str(e),
                "method": "exact_match_fallback",
                "match": is_match,
            }


def score_batch_annotations(
    annotation_dir: Path,
    input_dir: Path,
    output_dir: Path,
    output_file: Optional[Path] = None,
    llm_generate_func: Optional[callable] = None,
    annotation_type: str = "annotation",
) -> Dict[str, Any]:
    """
    Score a batch of annotations against ground truth.

    Args:
        annotation_dir: Directory containing annotation/prediction files
        input_dir: Directory containing input files
        output_dir: Directory containing ground truth output files
        output_file: Optional path to save scoring results
        llm_generate_func: Optional LLM function for advanced scoring
        annotation_type: Type of annotations ("annotation" or "prediction")

    Returns:
        Dictionary with scoring results for all examples
    """
    # Initialize scorer
    scorer = ChecklistScorer(llm_generate_func=llm_generate_func)

    # Load all annotations
    annotations = load_annotations_batch(annotation_dir)

    if not annotations:
        logger.error(f"No {annotation_type}s found in {annotation_dir}")
        return {"error": f"No {annotation_type}s found"}

    # Score each annotation
    results = {}
    total_scores = {
        "overall": [],
        "retrieval": [],
        "identification": [],
        "task_selection": [],
    }

    for example_id, annotation in annotations.items():
        try:
            # Load corresponding output data
            output_file_path = output_dir / f"{example_id}_output.json"
            if not output_file_path.exists():
                logger.error(f"Output file not found for {example_id}")
                results[example_id] = {"error": "Output file not found"}
                continue

            with open(output_file_path, "r") as f:
                output_data = json.load(f)

            # Optionally load input data for context
            input_data = None
            input_file_path = input_dir / f"{example_id}_input.json"
            if input_file_path.exists():
                with open(input_file_path, "r") as f:
                    input_data = json.load(f)

            # Score the annotation
            scoring_result = scorer.score_annotation(
                annotation, output_data, input_data
            )

            # Store results
            results[example_id] = scoring_result.to_dict()

            # Accumulate scores
            total_scores["overall"].append(scoring_result.overall_score)
            total_scores["retrieval"].append(scoring_result.retrieval_score)
            total_scores["identification"].append(scoring_result.identification_score)
            total_scores["task_selection"].append(scoring_result.task_selection_score)

            logger.info(
                f"Scored {example_id}: Overall={scoring_result.overall_score:.2%}"
            )

        except Exception as e:
            logger.error(f"Error scoring {example_id}: {e}")
            results[example_id] = {"error": str(e)}

    # Calculate aggregate statistics
    def calculate_stats(scores: List[float]) -> Dict[str, float]:
        if not scores:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        import numpy as np

        return {
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "count": len(scores),
        }

    # Prepare output
    output_data = {
        "scoring_results": results,
        "aggregate_statistics": {
            "overall": calculate_stats(total_scores["overall"]),
            "retrieval": calculate_stats(total_scores["retrieval"]),
            "identification": calculate_stats(total_scores["identification"]),
            "task_selection": calculate_stats(total_scores["task_selection"]),
        },
        "metadata": {
            "annotation_directory": str(annotation_dir),
            "input_directory": str(input_dir),
            "output_directory": str(output_dir),
            "annotation_type": annotation_type,
            "total_examples": len(annotations),
            "successful_scores": len([r for r in results.values() if "error" not in r]),
        },
    }

    # Save if output file specified
    if output_file:
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)
        logger.info(f"Scoring results saved to: {output_file}")

    return output_data


if __name__ == "__main__":
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description="Score PersonaSim evaluations")
    parser.add_argument("annotation_dir", help="Directory with annotations")
    parser.add_argument("input_dir", help="Directory with input files")
    parser.add_argument("output_dir", help="Directory with output files")
    parser.add_argument("-o", "--output", help="Output file for scores")
    parser.add_argument(
        "-t",
        "--type",
        choices=["annotation", "prediction"],
        default="annotation",
        help="Type of annotations",
    )

    args = parser.parse_args()

    scores = score_batch_annotations(
        Path(args.annotation_dir),
        Path(args.input_dir),
        Path(args.output_dir),
        Path(args.output) if args.output else None,
        annotation_type=args.type,
    )

    if "aggregate_statistics" in scores:
        print("\nAggregate Statistics:")
        print(json.dumps(scores["aggregate_statistics"], indent=2))
    else:
        print(json.dumps(scores, indent=2))
