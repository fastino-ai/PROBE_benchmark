#!/usr/bin/env python3
"""
Prepare generated examples for annotation by splitting them into inputs and outputs.

This script processes batch example directories and creates:
1. inputs/ directory - Contains world model and shuffled data points
2. outputs/ directory - Contains ground truth (world model, bottleneck, checklist, true positive IDs)

Usage:
    python prepare_for_annotation.py <batch_directory> [--input-dir inputs] [--output-dir outputs]
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def clean_corpus_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean a corpus item by removing metadata that reveals if it's a true positive or distractor.

    Args:
        item: The corpus item to clean

    Returns:
        Cleaned corpus item with only essential fields
    """
    # Create a clean copy with only essential fields
    clean_item = {
        "id": item.get("id"),
        "type": item.get("type"),
        "payload": item.get("payload", item.get("content", {})),
    }

    # Remove any fields that might reveal the nature of the item
    fields_to_remove = [
        "metadata",
        "created_at",
        "updated_at",
        "tags",
        "bottleneck_evidence",
    ]

    # Clean payload if it exists
    if isinstance(clean_item.get("payload"), dict):
        payload = clean_item["payload"].copy()
        for field in fields_to_remove:
            payload.pop(field, None)
        clean_item["payload"] = payload

    return clean_item


def prepare_single_example(
    example_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Split a single example into input and output components.

    Args:
        example_data: The complete example data

    Returns:
        Tuple of (input_data, output_data)
    """
    # Prepare input data
    input_data = {"world_model": example_data.get("world_model", {}), "data_points": []}

    # Collect and clean all data points
    true_positives = example_data.get("true_positives", [])
    distractors = example_data.get("distractors", [])

    # Track true positive IDs for output
    true_positive_ids = [tp.get("id") for tp in true_positives]
    distractor_ids = [d.get("id") for d in distractors]

    # Clean and combine all items
    all_items = []

    # Add cleaned true positives
    for tp in true_positives:
        clean_item = clean_corpus_item(tp)
        all_items.append(clean_item)

    # Add cleaned distractors
    for distractor in distractors:
        clean_item = clean_corpus_item(distractor)
        all_items.append(clean_item)

    # Shuffle all items
    random.shuffle(all_items)

    # Add to input data
    input_data["data_points"] = all_items

    # Add metadata about the annotation task
    input_data["annotation_instructions"] = {
        "task": "Review the data points and identify which ones are relevant to addressing potential workflow bottlenecks for this persona. Then identify the bottleneck and select the appropriate action.",
        "total_data_points": len(all_items),
    }

    # Prepare output data (ground truth)
    output_data = {
        "world_model": example_data.get("world_model", {}),
        "bottleneck": example_data.get("bottleneck", {}),
        "checklist": example_data.get("checklist", {}),
        "true_positive_ids": true_positive_ids,
        "distractor_ids": distractor_ids,
        "metadata": example_data.get("metadata", {}),
    }

    return input_data, output_data


def process_batch_directory(
    batch_dir: Path, input_dir: Path, output_dir: Path, seed: int = 42
) -> Dict[str, Any]:
    """
    Process all examples in a batch directory.

    Args:
        batch_dir: Directory containing batch example JSON files
        input_dir: Directory to save input files
        output_dir: Directory to save output files
        seed: Random seed for shuffling

    Returns:
        Summary of processing results
    """
    # Set random seed for reproducibility
    random.seed(seed)

    # Create output directories
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all JSON files in batch directory
    json_files = list(batch_dir.glob("*.json"))

    # Filter out summary files
    example_files = [f for f in json_files if "summary" not in f.name.lower()]

    if not example_files:
        logger.error(f"No example files found in {batch_dir}")
        return {"error": "No example files found"}

    logger.info(f"Found {len(example_files)} example files to process")

    # Process each file
    summary = {
        "batch_directory": str(batch_dir),
        "input_directory": str(input_dir),
        "output_directory": str(output_dir),
        "total_files": len(example_files),
        "processed_files": 0,
        "failed_files": 0,
        "processing_details": [],
    }

    for example_file in sorted(example_files):
        try:
            logger.info(f"Processing: {example_file.name}")

            # Load the example data
            with open(example_file, "r", encoding="utf-8") as f:
                example_data = json.load(f)

            # Split into input and output
            input_data, output_data = prepare_single_example(example_data)

            # Create filenames
            base_name = example_file.stem
            input_file = input_dir / f"{base_name}_input.json"
            output_file = output_dir / f"{base_name}_output.json"

            # Save input file
            with open(input_file, "w", encoding="utf-8") as f:
                json.dump(input_data, f, indent=2, ensure_ascii=False)

            # Save output file
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            # Update summary
            summary["processed_files"] += 1
            summary["processing_details"].append(
                {
                    "original_file": example_file.name,
                    "input_file": input_file.name,
                    "output_file": output_file.name,
                    "total_data_points": len(input_data["data_points"]),
                    "true_positives": len(output_data["true_positive_ids"]),
                    "distractors": len(output_data["distractor_ids"]),
                }
            )

            logger.info(f"  ✓ Created {input_file.name} and {output_file.name}")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {example_file.name}: {str(e)}")
            summary["failed_files"] += 1
            summary["processing_details"].append(
                {"original_file": example_file.name, "error": str(e)}
            )

    # Save processing summary
    summary_file = (
        input_dir.parent
        / f"preparation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("\nProcessing complete!")
    logger.info(f"  - Processed: {summary['processed_files']} files")
    logger.info(f"  - Failed: {summary['failed_files']} files")
    logger.info(f"  - Input files saved to: {input_dir}")
    logger.info(f"  - Output files saved to: {output_dir}")
    logger.info(f"  - Summary saved to: {summary_file}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Prepare batch examples for annotation by splitting into inputs and outputs"
    )
    parser.add_argument(
        "batch_directory",
        type=str,
        help="Path to the batch directory containing example JSON files",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="inputs",
        help="Directory name for input files (default: inputs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory name for output files (default: outputs)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for shuffling (default: 42)"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        help="Base directory for inputs/outputs (defaults to parent of batch directory)",
    )

    args = parser.parse_args()

    # Resolve paths
    batch_dir = Path(args.batch_directory)

    if not batch_dir.exists():
        logger.error(f"Batch directory does not exist: {batch_dir}")
        return

    if not batch_dir.is_dir():
        logger.error(f"Path is not a directory: {batch_dir}")
        return

    # Determine base directory for inputs/outputs
    if args.base_dir:
        base_dir = Path(args.base_dir)
    else:
        base_dir = batch_dir.parent

    # Create input and output directories
    input_dir = base_dir / args.input_dir / batch_dir.name
    output_dir = base_dir / args.output_dir / batch_dir.name

    # Process the batch
    summary = process_batch_directory(batch_dir, input_dir, output_dir, args.seed)

    # Exit with error code if any files failed
    if summary.get("failed_files", 0) > 0:
        exit(1)


if __name__ == "__main__":
    main()
