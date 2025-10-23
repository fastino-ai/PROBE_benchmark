#!/usr/bin/env python3
"""
Generate LLM scoring outputs for human annotation validation.

This script runs LLM scoring on a folder of model predictions and saves
only the LLM scoring outputs for bottleneck identification and task parameters.
These outputs are later compared against human annotations to measure agreement.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import argparse
from dotenv import load_dotenv

from data_generation.utils.clients.openai_client import get_openai_client
from evaluation.scoring import ChecklistScorer
from annotation.annotation_format import Annotation, ActionSelection

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

load_dotenv()


def get_llm_function(model_name: str):
    """Create LLM function for scoring."""
    client = get_openai_client()

    def llm_generate_func(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that evaluates agent performance.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    return llm_generate_func


def extract_llm_scoring_outputs(
    scoring_result_details: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract only the LLM scoring components we want humans to validate.

    Returns dict with bottleneck identification and task parameter judgments.
    """
    llm_outputs = {}

    # Extract bottleneck identification judgment
    identification_details = scoring_result_details.get("identification_details", {})
    if identification_details.get("method") == "llm_judge":
        llm_outputs["bottleneck_identification"] = {
            "judgment": identification_details.get("judgment", "INCORRECT"),
            "reasoning": identification_details.get("reasoning", ""),
            "identified": identification_details.get("identified", ""),
            "expected": identification_details.get("expected", ""),
        }

    # Extract task parameter judgment
    task_details = scoring_result_details.get("task_selection_details", {})
    if task_details.get("parameter_details", {}).get("method") == "llm_judge":
        param_details = task_details.get("parameter_details", {})
        llm_outputs["task_parameters"] = {
            "judgment": param_details.get("judgment", "INCORRECT"),
            "reasoning": param_details.get("reasoning", ""),
            "selected": param_details.get("selected", {}),
            "expected": param_details.get("expected", {}),
        }

    # Add action selection info for context (but no LLM judgment needed)
    llm_outputs["action_selection"] = {
        "selected_action": task_details.get("selected_action", ""),
        "expected_action": task_details.get("expected_action", ""),
        "action_correct": task_details.get("action_correct", False),
    }

    return llm_outputs


def load_prediction_data(filepath: Path) -> Tuple[Optional[Annotation], Optional[str]]:
    """Load and convert various prediction formats to Annotation."""
    try:
        with open(filepath, "r") as f:
            pred_data = json.load(f)

        # Handle different prediction formats
        if "retrieved_document_ids" in pred_data:  # Standard format
            annotation = Annotation.from_dict(pred_data)
        elif "retrieved_documents" in pred_data:  # baseline_agent format
            action_data = pred_data.get("action", {})
            annotation = Annotation(
                retrieved_document_ids=pred_data.get("retrieved_documents", []),
                bottleneck_description=pred_data.get("bottleneck", ""),
                action_selection=ActionSelection(
                    name=action_data.get("function_name", ""),
                    schema=action_data.get("parameters", {}),
                ),
            )
        else:
            # Try other formats
            annotation = Annotation(
                retrieved_document_ids=pred_data.get(
                    "selected_ids", pred_data.get("retrieved_ids", [])
                ),
                bottleneck_description=pred_data.get(
                    "bottleneck_description", pred_data.get("bottleneck", "")
                ),
                action_selection=ActionSelection(
                    name=pred_data.get("action", {}).get(
                        "name", pred_data.get("action", {}).get("function_name", "")
                    ),
                    schema=pred_data.get("action", {}).get(
                        "parameters", pred_data.get("action", {}).get("schema", {})
                    ),
                ),
            )

        return annotation, None
    except Exception as e:
        return None, str(e)


def generate_llm_scoring_for_directory(
    predictions_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    input_dir: Optional[Path] = None,
    model_name: str = "gpt-4o-mini",
    skip_errors: bool = True,
) -> Dict[str, Any]:
    """
    Generate LLM scoring outputs for all predictions in a directory.

    Args:
        predictions_dir: Directory containing model predictions
        labels_dir: Directory containing ground truth labels
        output_dir: Directory to save LLM scoring outputs
        input_dir: Optional directory with input files for context
        model_name: OpenAI model to use for scoring
        skip_errors: Whether to skip files with errors

    Returns:
        Summary statistics and results
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize LLM function and scorer
    llm_func = get_llm_function(model_name)
    scorer = ChecklistScorer(llm_generate_func=llm_func)

    # Get all prediction files
    pred_files = list(predictions_dir.glob("*.json"))
    if not pred_files:
        raise ValueError(f"No JSON files found in {predictions_dir}")

    # Extract example IDs and match with labels
    def extract_example_id(filename: str) -> str:
        """Extract example ID from various filename formats."""
        name = filename.replace(".json", "")
        for suffix in [
            "_annotation",
            "_output",
            "_input",
            "_prediction",
            "_pred",
            "_label",
            "_results",
        ]:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        return name

    pred_ids = {extract_example_id(f.name): f for f in pred_files}
    label_files = list(labels_dir.glob("*.json"))
    label_ids = {extract_example_id(f.name): f for f in label_files}

    # Find matched examples
    matched_ids = set(pred_ids.keys()) & set(label_ids.keys())
    logger.info(f"Found {len(matched_ids)} matched examples")

    if not matched_ids:
        raise ValueError("No matching prediction-label pairs found")

    results = {}
    errors = []
    processed = 0

    for example_id in sorted(matched_ids):
        try:
            # Load prediction
            pred_file = pred_ids[example_id]
            annotation, error = load_prediction_data(pred_file)
            if error:
                errors.append(f"Error loading prediction {example_id}: {error}")
                if not skip_errors:
                    break
                continue

            # Load label
            label_file = label_ids[example_id]
            with open(label_file, "r") as f:
                output_data = json.load(f)

            # Load input if available
            input_data = None
            if input_dir and input_dir.exists():
                input_file = input_dir / f"{example_id}_input.json"
                if input_file.exists():
                    with open(input_file, "r") as f:
                        input_data = json.load(f)

            # Score with LLM
            scoring_result = scorer.score_annotation(
                annotation, output_data, input_data
            )

            # Extract only LLM judgments
            llm_outputs = extract_llm_scoring_outputs(scoring_result.to_dict())

            if llm_outputs:  # Only save if there are LLM outputs
                # Save LLM scoring output
                output_file = output_dir / f"{example_id}_llm_scoring.json"
                with open(output_file, "w") as f:
                    json.dump(
                        {
                            "example_id": example_id,
                            "llm_scoring": llm_outputs,
                            "metadata": {
                                "model_name": model_name,
                                "timestamp": datetime.now().isoformat(),
                                "prediction_file": pred_file.name,
                                "label_file": label_file.name,
                            },
                        },
                        f,
                        indent=2,
                    )

                results[example_id] = llm_outputs
                processed += 1
                logger.info(f"Processed {example_id} ({processed}/{len(matched_ids)})")
            else:
                logger.warning(
                    f"No LLM outputs found for {example_id} (using exact matching?)"
                )

        except Exception as e:
            error_msg = f"Error processing {example_id}: {e}"
            errors.append(error_msg)
            logger.error(error_msg)
            if not skip_errors:
                break

    # Save summary
    summary = {
        "metadata": {
            "predictions_directory": str(predictions_dir),
            "labels_directory": str(labels_dir),
            "output_directory": str(output_dir),
            "input_directory": str(input_dir) if input_dir else None,
            "model_name": model_name,
            "timestamp": datetime.now().isoformat(),
            "total_matched": len(matched_ids),
            "processed": processed,
            "errors": len(errors),
        },
        "results": results,
        "errors": errors,
    }

    summary_file = output_dir / "llm_scoring_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Generated LLM scoring for {processed} examples")
    logger.info(f"Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate LLM scoring outputs for human annotation validation"
    )
    parser.add_argument(
        "predictions_dir", help="Directory containing model predictions"
    )
    parser.add_argument("labels_dir", help="Directory containing ground truth labels")
    parser.add_argument("output_dir", help="Directory to save LLM scoring outputs")
    parser.add_argument(
        "--input-dir", help="Optional directory with input files for context"
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        choices=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        help="OpenAI model to use for scoring",
    )
    parser.add_argument(
        "--no-skip-errors",
        action="store_true",
        help="Stop on first error instead of skipping",
    )

    args = parser.parse_args()

    try:
        summary = generate_llm_scoring_for_directory(
            Path(args.predictions_dir),
            Path(args.labels_dir),
            Path(args.output_dir),
            Path(args.input_dir) if args.input_dir else None,
            args.model,
            not args.no_skip_errors,
        )

        print("\nLLM Scoring Generation Complete!")
        print(f"Processed: {summary['metadata']['processed']}")
        print(f"Errors: {summary['metadata']['errors']}")
        if summary["errors"]:
            print("Errors:")
            for error in summary["errors"]:
                print(f"  - {error}")

    except Exception as e:
        logger.error(f"Failed to generate LLM scoring: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
