#!/usr/bin/env python3
"""
Script to extract failing instances from model evaluation results.
Identifies two types of failures:
1. Bottleneck identification failures
2. Action parameter failures (given successful bottleneck identification)

Extracts up to 25 examples per model per failure type with input-output pairs.
"""

import json
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict


class FailureExtractor:
    """Extracts and analyzes failure cases from model evaluation results."""

    def __init__(self, base_path: str):
        """Initialize with base project path."""
        self.base_path = Path(base_path)
        self.final_results_path = self.base_path / "final_results"
        self.consolidated_scores_path = (
            self.final_results_path / "consolidated_scores.json"
        )
        self.ground_truth_path = (
            self.base_path / "data" / "sept_23_1000_outputs_20250923_131956"
        )

        # Models to exclude from analysis
        self.excluded_models = {
            "batch_openai_gpt-4.1-mini",
            "batch_together_openai_gpt-oss-120b",
            "batch_together_openai_gpt-oss-20b",
        }

    def load_consolidated_scores(self) -> Dict[str, Any]:
        """Load the consolidated scores JSON file."""
        with open(self.consolidated_scores_path, "r") as f:
            return json.load(f)

    def get_model_directories(self) -> List[str]:
        """Get list of model result directories."""
        model_dirs = []
        for item in self.final_results_path.iterdir():
            if item.is_dir() and not item.name.startswith("."):
                model_dirs.append(item.name)
        return sorted(model_dirs)

    def should_include_instance(self, instance_data: Dict[str, Any]) -> bool:
        """
        Determine if instance should be included in analysis.
        Include if identification OR parameters have failures.
        """
        identification_judgment = instance_data.get("identification", {}).get(
            "judgment", ""
        )
        parameters_judgment = instance_data.get("parameters", {}).get("judgment", "")

        # Include if either identification or parameters failed
        id_failed = identification_judgment in ["INCORRECT", "PARTIALLY_CORRECT"]
        param_failed = parameters_judgment in ["INCORRECT", "PARTIALLY_CORRECT"]

        return id_failed or param_failed

    def get_failure_pools(self, instance_data: Dict[str, Any]) -> List[str]:
        """
        Get the failure pool(s) this instance should be included in using overlapping logic.

        Overlapping pools structure:
        1. Identification pool: identification_judgment ∈ {INCORRECT, PARTIALLY_CORRECT}
        2. Parameter pool: parameters_judgment ∈ {INCORRECT, PARTIALLY_CORRECT} ∧
                          identification_judgment ∈ {PARTIALLY_CORRECT, CORRECT}
        3. Instances can appear in BOTH pools when both conditions are met
        """
        identification_judgment = instance_data.get("identification", {}).get(
            "judgment", ""
        )
        parameters_judgment = instance_data.get("parameters", {}).get("judgment", "")

        pools = []

        # Check for identification pool inclusion
        if identification_judgment in ["INCORRECT", "PARTIALLY_CORRECT"]:
            pools.append("identification_failure")

        # Check for parameter pool inclusion
        if parameters_judgment in [
            "INCORRECT",
            "PARTIALLY_CORRECT",
        ] and identification_judgment in ["PARTIALLY_CORRECT", "CORRECT"]:
            pools.append("parameter_failure")

        return pools

    def load_model_output(self, model_name: str, instance_name: str) -> Dict[str, Any]:
        """Load the model's actual response for a given instance."""
        output_file = (
            self.final_results_path / model_name / f"{instance_name}_results.json"
        )
        if output_file.exists():
            with open(output_file, "r") as f:
                return json.load(f)
        return {}

    def load_ground_truth(self, instance_name: str) -> Dict[str, Any]:
        """Load the ground truth data for a given instance."""
        ground_truth_file = self.ground_truth_path / f"{instance_name}_output.json"
        if ground_truth_file.exists():
            with open(ground_truth_file, "r") as f:
                return json.load(f)
        return {}

    def calculate_retrieval_accuracy(
        self, retrieved_docs: List[str], required_docs: List[str]
    ) -> float:
        """
        Calculate retrieval accuracy as percentage of required documents retrieved.

        Args:
            retrieved_docs: List of document IDs retrieved by the model
            required_docs: List of required document IDs from ground truth

        Returns:
            Float between 0.0 and 1.0 representing retrieval accuracy
        """
        if not required_docs:
            return 1.0  # No documents required, perfect accuracy

        retrieved_set = set(retrieved_docs)
        required_set = set(required_docs)

        correctly_retrieved = len(required_set.intersection(retrieved_set))
        total_required = len(required_set)

        return correctly_retrieved / total_required

    def extract_failures_by_model(
        self, max_per_type: int = None
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Extract failure cases by model and failure type using overlapping pools.

        Args:
            max_per_type: Maximum number of examples per failure type per model

        Returns:
            Dict with structure: {model_name: {failure_type: [failure_instances]}}
        """
        consolidated_scores = self.load_consolidated_scores()
        model_failures = defaultdict(lambda: defaultdict(list))

        for model_name, model_data in consolidated_scores.items():
            # Skip excluded models
            if model_name in self.excluded_models:
                print(f"Skipping excluded model: {model_name}")
                continue

            print(f"Processing model: {model_name}")

            # Count failures for tracking limits per category
            failure_counts = defaultdict(int)

            for instance_name, instance_data in model_data.items():
                # Skip instances with no failures
                if not self.should_include_instance(instance_data):
                    continue

                # Get the pool(s) this instance belongs to
                pools = self.get_failure_pools(instance_data)

                # Skip if no failure pools
                if not pools:
                    continue

                # Load model output
                model_output = self.load_model_output(model_name, instance_name)
                if not model_output:
                    continue

                # Load ground truth and calculate retrieval accuracy
                ground_truth = self.load_ground_truth(instance_name)
                retrieval_accuracy = 0.0
                required_docs = []

                if ground_truth:
                    # Get required documents from ground truth checklist
                    checklist = ground_truth.get("checklist", {})
                    for step in checklist.get("required_steps", []):
                        if step.get("step_type") == "retrieval":
                            required_docs = step.get("evidence_required", [])
                            break

                    # Get retrieved documents from model output
                    retrieved_docs = model_output.get("retrieved_documents", [])

                    # Calculate retrieval accuracy
                    retrieval_accuracy = self.calculate_retrieval_accuracy(
                        retrieved_docs, required_docs
                    )

                # Add instance to each pool it belongs to (overlapping)
                for pool in pools:
                    # Check if we've hit the limit for this pool type
                    if (
                        max_per_type is not None
                        and failure_counts[pool] >= max_per_type
                    ):
                        continue

                    # Create failure record
                    failure_record = {
                        "instance_name": instance_name,
                        "failure_pools": pools,  # List of all pools this instance belongs to
                        "model_output": model_output,
                        "scores": instance_data,
                        "identification_judgment": instance_data.get(
                            "identification", {}
                        ).get("judgment", ""),
                        "parameters_judgment": instance_data.get("parameters", {}).get(
                            "judgment", ""
                        ),
                        "identification_reasoning": instance_data.get(
                            "identification", {}
                        ).get("reasoning", ""),
                        "parameters_reasoning": instance_data.get("parameters", {}).get(
                            "reasoning", ""
                        ),
                        "retrieval_accuracy": retrieval_accuracy,
                        "ground_truth_required_docs": required_docs,
                        "model_retrieved_docs": model_output.get(
                            "retrieved_documents", []
                        ),
                    }

                    model_failures[model_name][pool].append(failure_record)
                    failure_counts[pool] += 1

            print(
                f"  - Identification failures: {len(model_failures[model_name]['identification_failure'])}"
            )
            print(
                f"  - Parameter failures: {len(model_failures[model_name]['parameter_failure'])}"
            )
            overlapping = sum(
                1
                for pool in ["identification_failure", "parameter_failure"]
                for record in model_failures[model_name][pool]
                if len(record["failure_pools"]) > 1
            )
            print(
                f"  - Overlapping instances: {overlapping // 2}"
            )  # Divide by 2 since each overlapping instance is counted twice

        return dict(model_failures)

    def generate_summary_report(
        self, failures: Dict[str, Dict[str, List[Dict[str, Any]]]]
    ) -> Dict[str, Any]:
        """Generate a summary report of failures across all models."""
        summary = {
            "total_models": len(failures),
            "models": {},
            "failure_type_totals": defaultdict(int),
        }

        for model_name, model_failures in failures.items():
            model_summary = {
                "identification_failures": len(
                    model_failures.get("identification_failure", [])
                ),
                "parameter_failures": len(model_failures.get("parameter_failure", [])),
                "total_failures": len(model_failures.get("identification_failure", []))
                + len(model_failures.get("parameter_failure", [])),
            }
            summary["models"][model_name] = model_summary

            summary["failure_type_totals"]["identification_failure"] += model_summary[
                "identification_failures"
            ]
            summary["failure_type_totals"]["parameter_failure"] += model_summary[
                "parameter_failures"
            ]

        return summary

    def save_results(
        self, failures: Dict[str, Dict[str, List[Dict[str, Any]]]], output_file: str
    ):
        """Save the extracted failures to a JSON file."""
        # Add summary to the output
        summary = self.generate_summary_report(failures)

        output_data = {
            "summary": summary,
            "failures_by_model": failures,
            "extraction_metadata": {
                "base_path": str(self.base_path),
                "total_models_processed": len(failures),
                "max_examples_per_type": "no_limit",
                "approach": "overlapping_pools",
                "description": "Instances can appear in both identification and parameter pools when criteria are met",
            },
        }

        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=2)

        print(f"Results saved to: {output_file}")


def main():
    """Main execution function with argparse support.

    Command-line Arguments:
        --base-path: Base path to project directory containing final_results/
        --output-file: Path to save extracted failures JSON
        --max-per-type: Maximum number of examples per failure type per model

    Examples:
        >>> # Extract all failures
        >>> python extract_failures.py --base-path /path/to/project

        >>> # Limit to 25 examples per type
        >>> python extract_failures.py --base-path /path/to/project --max-per-type 25

        >>> # Custom output file
        >>> python extract_failures.py --base-path . --output-file my_failures.json
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract failure cases from model evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract all failures from current directory
    python extract_failures.py --base-path .
    
    # Extract up to 25 failures per type per model
    python extract_failures.py --base-path . --max-per-type 25
    
    # Custom output location
    python extract_failures.py --base-path /path/to/results --output-file failures.json
        """,
    )

    parser.add_argument(
        "--base-path",
        type=str,
        required=True,
        help="Base path to project directory containing final_results/ subdirectory",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path (default: <base-path>/extracted_failures.json)",
    )

    parser.add_argument(
        "--max-per-type",
        type=int,
        default=None,
        help="Maximum number of examples per failure type per model (default: no limit)",
    )

    args = parser.parse_args()

    # Initialize extractor
    base_path = args.base_path
    extractor = FailureExtractor(base_path)

    # Check if required files exist
    if not extractor.consolidated_scores_path.exists():
        print(
            f"Error: Consolidated scores file not found at {extractor.consolidated_scores_path}"
        )
        print(
            f"Expected location: {extractor.final_results_path}/consolidated_scores.json"
        )
        return 1

    # Extract failures
    print("Extracting failure cases...")
    print(f"Base path: {base_path}")
    if args.max_per_type:
        print(f"Max per type: {args.max_per_type}")
    else:
        print("Max per type: no limit")

    failures = extractor.extract_failures_by_model(max_per_type=args.max_per_type)

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = str(Path(base_path) / "extracted_failures.json")

    # Save results
    extractor.save_results(failures, output_file)

    # Print summary
    summary = extractor.generate_summary_report(failures)
    print("\n=== EXTRACTION SUMMARY ===")
    print(f"Total models processed: {summary['total_models']}")
    print(
        f"Total identification failures: {summary['failure_type_totals']['identification_failure']}"
    )
    print(
        f"Total parameter failures: {summary['failure_type_totals']['parameter_failure']}"
    )
    print("\nPer-model breakdown:")
    for model_name, model_stats in summary["models"].items():
        print(f"  {model_name}:")
        print(
            f"    - Identification failures: {model_stats['identification_failures']}"
        )
        print(f"    - Parameter failures: {model_stats['parameter_failures']}")
        print(f"    - Total: {model_stats['total_failures']}")

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
