#!/usr/bin/env python3
"""
Script to analyze failure modes from extracted failures and generate statistical tables.
Creates tabular output showing percentage of each failure mode per model.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


class FailureModeAnalyzer:
    """Analyzes failure modes and generates statistical tables."""

    def __init__(self, extracted_failures_path: str):
        """Initialize with path to extracted failures JSON."""
        self.failures_path = Path(extracted_failures_path)

    def load_failures(self) -> Dict:
        """Load the extracted failures data."""
        with open(self.failures_path, "r") as f:
            return json.load(f)

    def categorize_identification_failure(self, scores: Dict) -> List[str]:
        """
        Categorize identification failure based on scoring details.
        Returns list of failure modes for this instance.
        """
        categories = []
        essential = scores.get("identification", {}).get(
            "essential_details_analysis", {}
        )
        non_essential = scores.get("identification", {}).get(
            "non_essential_details_analysis", {}
        )

        # Essential detail failures
        # Combine wrong blocker and wrong blocked into single interpersonal category
        if (
            essential.get("who_blocked") == "incorrect"
            or essential.get("who_blocker") == "incorrect"
        ):
            categories.append("Incorrect interpersonal")

        if essential.get("why_root_cause") == "incorrect":
            categories.append("Incorrect root cause")

        # Non-essential detail failures
        if non_essential.get("when_deadline") in ["incorrect", "missing"]:
            categories.append("Missing/wrong deadline")

        # If no specific categories found but it's marked as failure, use generic
        if not categories:
            categories.append("General identification error")

        return categories

    def is_correct_function_selected(self, scores: Dict) -> bool:
        """
        Determine if the correct function was selected based on scoring.
        Uses 'would_resolve_bottleneck' as primary indicator.
        """
        param_analysis = scores.get("parameters", {}).get("parameter_analysis", {})
        would_resolve = param_analysis.get("would_resolve_bottleneck", "")

        # Consider function correct if it would at least partially resolve the bottleneck
        return would_resolve in ["yes", "partially"]

    def categorize_parameter_failure(self, scores: Dict) -> List[str]:
        """
        Categorize parameter failure based on scoring details.
        Returns list of failure modes for this instance.
        NOTE: This should only be called for instances where correct function was selected.
        """
        categories = []
        param_analysis = scores.get("parameters", {}).get("parameter_analysis", {})

        # Analyze incorrect and missing parameters
        incorrect_params = param_analysis.get("incorrect_parameters", [])
        missing_params = param_analysis.get("missing_parameters", [])

        # If there are any missing parameters, categorize as missing critical parameters
        if missing_params:
            categories.append("Critical parameters missing")

        # If there are any incorrect parameters, categorize as incorrectly filled
        if incorrect_params:
            categories.append("Incorrectly filled")

        # If no specific categories found but it's marked as failure, use incorrectly filled as default
        if not categories:
            categories.append("Incorrectly filled")

        return categories

    def analyze_failures(self) -> Tuple[Dict, List[str]]:
        """
        Analyze all failures and return categorized statistics.
        Returns: (statistics_dict, model_names_list)
        """
        data = self.load_failures()
        failures_by_model = data["failures_by_model"]

        # Structure: {task_type: {failure_mode: {model: count}}}
        stats = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        # Track total instances per task_type per model for percentage calculation
        totals = defaultdict(lambda: defaultdict(int))

        # Get all model names
        model_names = list(failures_by_model.keys())

        for model_name, model_failures in failures_by_model.items():

            # Process identification failures
            id_failures = model_failures.get("identification_failure", [])
            totals["identification"][model_name] = len(id_failures)

            for failure in id_failures:
                categories = self.categorize_identification_failure(failure["scores"])
                for category in categories:
                    stats["identification"][category][model_name] += 1

            # Process parameter failures (already properly conditioned in extraction)
            param_failures = model_failures.get("parameter_failure", [])
            totals["parameter"][model_name] = len(param_failures)

            # Separate parameter failures by function selection correctness
            correct_function_failures = []
            wrong_function_failures = []

            for failure in param_failures:
                if self.is_correct_function_selected(failure["scores"]):
                    correct_function_failures.append(failure)
                else:
                    wrong_function_failures.append(failure)

            # Update totals for analysis breakdown
            totals["function_selection"][model_name] = len(wrong_function_failures)
            totals["parameter_given_correct_function"][model_name] = len(
                correct_function_failures
            )

            # Analyze function selection failures
            for failure in wrong_function_failures:
                stats["function_selection"]["Wrong function selected"][model_name] += 1

            # Analyze parameter failures (only for correct function selection)
            for failure in correct_function_failures:
                categories = self.categorize_parameter_failure(failure["scores"])
                for category in categories:
                    stats["parameter_given_correct_function"][category][model_name] += 1

        # Convert counts to percentages with correct denominators
        percentage_stats = defaultdict(lambda: defaultdict(dict))

        # Map stats categories to their correct totals
        totals_mapping = {
            "identification": "identification",
            "function_selection": "parameter",  # out of all parameter failures
            "parameter_given_correct_function": "parameter_given_correct_function",
        }

        for task_type in stats:
            total_key = totals_mapping.get(task_type, task_type)
            for failure_mode in stats[task_type]:
                for model in stats[task_type][failure_mode]:
                    total = totals[total_key][model]
                    count = stats[task_type][failure_mode][model]
                    if total > 0:
                        percentage = (count / total) * 100
                        percentage_stats[task_type][failure_mode][model] = percentage
                    else:
                        percentage_stats[task_type][failure_mode][model] = 0.0

        return dict(percentage_stats), model_names

    def analyze_retrieval_identification_relationship(self) -> Tuple[Dict, Dict]:
        """
        Analyze conditional probability relationship between retrieval accuracy and identification correctness.
        Uses 3 categories based on percentage of correctly retrieved documents:
        - 0%: No documents retrieved correctly
        - >25% but <75%: Partial retrieval
        - >=75%: Most/all documents retrieved correctly

        Returns:
            Tuple of (conditional_stats, raw_counts) where:
            - conditional_stats: P(identification_correct | retrieval_quality) for each model
            - raw_counts: Raw count data for transparency
        """
        data = self.load_failures()
        failures_by_model = data["failures_by_model"]

        conditional_stats = {}
        raw_counts = {}

        for model_name, model_failures in failures_by_model.items():
            # Initialize counters for 3 categories
            zero_pct_correct_id = 0
            zero_pct_total = 0
            partial_pct_correct_id = 0
            partial_pct_total = 0
            high_pct_correct_id = 0
            high_pct_total = 0

            # Process all failures (both identification and parameter failures)
            all_failures = model_failures.get(
                "identification_failure", []
            ) + model_failures.get("parameter_failure", [])

            for failure in all_failures:
                # Use the retrieval_accuracy directly (already calculated as percentage)
                retrieval_accuracy = failure.get("retrieval_accuracy", 0.0)

                # Categorize retrieval quality based on percentage
                if retrieval_accuracy == 0.0:
                    category = "zero_pct"
                elif retrieval_accuracy >= 0.75:  # >= 75%
                    category = "high_pct"
                elif retrieval_accuracy > 0.25:  # > 25% but < 75%
                    category = "partial_pct"
                else:
                    continue  # Skip cases between 0% and 25% for cleaner categories

                # Check if identification was correct
                id_judgment = failure.get("identification_judgment", "")
                id_correct = id_judgment in ["PARTIALLY_CORRECT", "CORRECT"]

                # Update counters
                if category == "zero_pct":
                    zero_pct_total += 1
                    if id_correct:
                        zero_pct_correct_id += 1
                elif category == "partial_pct":
                    partial_pct_total += 1
                    if id_correct:
                        partial_pct_correct_id += 1
                elif category == "high_pct":
                    high_pct_total += 1
                    if id_correct:
                        high_pct_correct_id += 1

            # Calculate conditional probabilities
            zero_pct_prob = (
                (zero_pct_correct_id / zero_pct_total) if zero_pct_total > 0 else 0.0
            )
            partial_pct_prob = (
                (partial_pct_correct_id / partial_pct_total)
                if partial_pct_total > 0
                else 0.0
            )
            high_pct_prob = (
                (high_pct_correct_id / high_pct_total) if high_pct_total > 0 else 0.0
            )

            conditional_stats[model_name] = {
                "zero_pct_correct_identification_rate": zero_pct_prob,
                "partial_pct_correct_identification_rate": partial_pct_prob,
                "high_pct_correct_identification_rate": high_pct_prob,
            }

            raw_counts[model_name] = {
                "zero_pct_correct_id": zero_pct_correct_id,
                "zero_pct_total": zero_pct_total,
                "partial_pct_correct_id": partial_pct_correct_id,
                "partial_pct_total": partial_pct_total,
                "high_pct_correct_id": high_pct_correct_id,
                "high_pct_total": high_pct_total,
            }

        return conditional_stats, raw_counts

    def analyze_identification_failure_attribution(self) -> Dict:
        """
        Analyze what percentage of identification failures are due to retrieval issues vs understanding issues.
        Uses 3 categories based on percentage of correctly retrieved documents:
        - 0%: No documents retrieved correctly (pure retrieval issue)
        - >25% but <75%: Partial retrieval (mixed issue)
        - >=75%: Most/all documents retrieved correctly (understanding issue)

        Returns:
            Dict with attribution analysis for each model
        """
        data = self.load_failures()
        failures_by_model = data["failures_by_model"]

        attribution_stats = {}

        for model_name, model_failures in failures_by_model.items():
            # Get identification failures only
            id_failures = model_failures.get("identification_failure", [])

            if not id_failures:
                attribution_stats[model_name] = {
                    "total_identification_failures": 0,
                    "failures_zero_pct": 0,
                    "failures_partial_pct": 0,
                    "failures_high_pct": 0,
                    "zero_pct_percentage": 0.0,
                    "partial_pct_percentage": 0.0,
                    "high_pct_percentage": 0.0,
                }
                continue

            zero_pct_failures = 0
            partial_pct_failures = 0
            high_pct_failures = 0
            processed_failures = 0

            for failure in id_failures:
                # Use the retrieval_accuracy directly (already calculated as percentage)
                retrieval_accuracy = failure.get("retrieval_accuracy", 0.0)

                # Categorize by retrieval quality based on percentage
                if retrieval_accuracy == 0.0:
                    zero_pct_failures += 1
                    processed_failures += 1
                elif retrieval_accuracy >= 0.75:  # >= 75%
                    high_pct_failures += (
                        1  # Understanding issue - had good retrieval but still failed
                    )
                    processed_failures += 1
                elif retrieval_accuracy > 0.25:  # > 25% but < 75%
                    partial_pct_failures += 1
                    processed_failures += 1
                # Skip cases between 0% and 25% for cleaner categories

            attribution_stats[model_name] = {
                "total_identification_failures": processed_failures,
                "failures_zero_pct": zero_pct_failures,
                "failures_partial_pct": partial_pct_failures,
                "failures_high_pct": high_pct_failures,
                "zero_pct_percentage": (
                    (zero_pct_failures / processed_failures) * 100
                    if processed_failures > 0
                    else 0.0
                ),
                "partial_pct_percentage": (
                    (partial_pct_failures / processed_failures) * 100
                    if processed_failures > 0
                    else 0.0
                ),
                "high_pct_percentage": (
                    (high_pct_failures / processed_failures) * 100
                    if processed_failures > 0
                    else 0.0
                ),
            }

        return attribution_stats

    def generate_table(self, stats: Dict, model_names: List[str]) -> str:
        """Generate formatted table string."""

        # Clean up model names for display (remove long prefixes)
        clean_model_names = []
        for name in model_names:
            if "anthropic" in name:
                clean_name = name.split("_")[-1]  # e.g., claude-opus-4-1-20250805
            elif "openai" in name and "batch_openai" in name:
                clean_name = name.replace("batch_openai_", "")  # e.g., gpt-4.1
            elif "together" in name:
                parts = name.split("_")
                if len(parts) >= 3:
                    clean_name = "_".join(parts[2:])  # e.g., deepseek-ai_DeepSeek-R1
                else:
                    clean_name = name
            else:
                clean_name = name
            clean_model_names.append(clean_name)

        # Build table
        lines = []

        # Header
        header_parts = ["Task Type", "Failure Mode"] + clean_model_names
        lines.append(" | ".join(f"{part:<25}" for part in header_parts))
        lines.append("-" * (27 * len(header_parts) + (len(header_parts) - 1) * 3))

        # Identification failures
        if "identification" in stats:
            for failure_mode in sorted(stats["identification"].keys()):
                row_parts = ["identification", failure_mode]
                for model in model_names:
                    percentage = stats["identification"][failure_mode].get(model, 0.0)
                    row_parts.append(f"{percentage:.1f}%")
                lines.append(" | ".join(f"{part:<25}" for part in row_parts))

        # Function selection failures
        if "function_selection" in stats:
            for failure_mode in sorted(stats["function_selection"].keys()):
                row_parts = ["function_selection", failure_mode]
                for model in model_names:
                    percentage = stats["function_selection"][failure_mode].get(
                        model, 0.0
                    )
                    row_parts.append(f"{percentage:.1f}%")
                lines.append(" | ".join(f"{part:<25}" for part in row_parts))

        # Parameter failures (only for cases with correct function selection)
        if "parameter_given_correct_function" in stats:
            for failure_mode in sorted(
                stats["parameter_given_correct_function"].keys()
            ):
                row_parts = ["parameter", failure_mode]
                for model in model_names:
                    percentage = stats["parameter_given_correct_function"][
                        failure_mode
                    ].get(model, 0.0)
                    row_parts.append(f"{percentage:.1f}%")
                lines.append(" | ".join(f"{part:<25}" for part in row_parts))

        return "\n".join(lines)

    def generate_summary_stats(self, stats: Dict, model_names: List[str]) -> str:
        """Generate summary statistics."""
        lines = []
        lines.append("=== SUMMARY STATISTICS ===\n")

        lines.append("=== CONDITIONING EXPLANATION ===")
        lines.append("Mathematical Conditional Specifications:")
        lines.append("")
        lines.append("• Identification rows:")
        lines.append(
            "  P(specific_issue | identification_judgment ∈ {INCORRECT, PARTIALLY_CORRECT})"
        )
        lines.append("  Example: P(wrong_blocked_person | identification_failed)")
        lines.append("")
        lines.append("• Function Selection row:")
        lines.append(
            "  P(wrong_function | parameter_failed ∧ identification_judgment ∈ {PARTIALLY_CORRECT, CORRECT})"
        )
        lines.append("  Example: P(would_resolve_bottleneck = 'no' | parameter_failed)")
        lines.append("")
        lines.append("• Parameter rows:")
        lines.append(
            "  P(specific_param_issue | parameter_failed ∧ identification_judgment ∈ {PARTIALLY_CORRECT, CORRECT} ∧ correct_function)"
        )
        lines.append(
            "  Example: P(missing_due_date | parameter_failed ∧ would_resolve_bottleneck ∈ {'yes', 'partially'})"
        )
        lines.append("")
        lines.append("Where:")
        lines.append(
            "  - identification_failed = judgment ∈ {INCORRECT, PARTIALLY_CORRECT}"
        )
        lines.append("  - parameter_failed = judgment ∈ {INCORRECT, PARTIALLY_CORRECT}")
        lines.append(
            "  - correct_function = would_resolve_bottleneck ∈ {'yes', 'partially'}"
        )
        lines.append("  - wrong_function = would_resolve_bottleneck = 'no'")
        lines.append("")
        lines.append("Hierarchical Structure:")
        lines.append("  1. If identification = INCORRECT → identification_failure pool")
        lines.append(
            "  2. If identification ∈ {PARTIALLY_CORRECT, CORRECT} ∧ parameters failed → parameter_failure pool"
        )
        lines.append(
            "  3. If identification = PARTIALLY_CORRECT ∧ parameters = CORRECT → identification_failure pool"
        )
        lines.append("")

        # Function selection failures
        lines.append("Function Selection Failures (average across models):")
        if "function_selection" in stats:
            func_averages = {}
            for failure_mode in stats["function_selection"]:
                values = [
                    stats["function_selection"][failure_mode].get(model, 0.0)
                    for model in model_names
                ]
                func_averages[failure_mode] = sum(values) / len(values)

            for failure_mode, avg in sorted(
                func_averages.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {failure_mode}: {avg:.1f}%")

        # Most common failure modes across all models
        lines.append("\nMost Common Identification Failures (average across models):")
        if "identification" in stats:
            id_averages = {}
            for failure_mode in stats["identification"]:
                values = [
                    stats["identification"][failure_mode].get(model, 0.0)
                    for model in model_names
                ]
                id_averages[failure_mode] = sum(values) / len(values)

            for failure_mode, avg in sorted(
                id_averages.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {failure_mode}: {avg:.1f}%")

        lines.append(
            "\nMost Common Parameter Failures (given correct function, average across models):"
        )
        if "parameter_given_correct_function" in stats:
            param_averages = {}
            for failure_mode in stats["parameter_given_correct_function"]:
                values = [
                    stats["parameter_given_correct_function"][failure_mode].get(
                        model, 0.0
                    )
                    for model in model_names
                ]
                param_averages[failure_mode] = sum(values) / len(values)

            for failure_mode, avg in sorted(
                param_averages.items(), key=lambda x: x[1], reverse=True
            ):
                lines.append(f"  {failure_mode}: {avg:.1f}%")

        return "\n".join(lines)

    def generate_retrieval_analysis_tables(
        self, conditional_stats: Dict, attribution_stats: Dict, model_names: List[str]
    ) -> str:
        """Generate formatted tables for retrieval-identification analysis."""
        lines = []

        # Clean up model names for display
        clean_model_names = []
        for name in model_names:
            if "anthropic" in name:
                clean_name = name.split("_")[-1]
            elif "openai" in name and "batch_openai" in name:
                clean_name = name.replace("batch_openai_", "")
            elif "together" in name:
                parts = name.split("_")
                if len(parts) >= 3:
                    clean_name = "_".join(parts[2:])
                else:
                    clean_name = name
            else:
                clean_name = name
            clean_model_names.append(clean_name)

        lines.append("=== RETRIEVAL-IDENTIFICATION CONDITIONAL ANALYSIS ===\n")

        # Conditional probability table with documentation
        lines.append(
            "Conditional Probabilities: P(Correct Identification | Retrieval Quality)"
        )
        lines.append("")
        lines.append("Mathematical Formulation:")
        lines.append("• P(correct_identification | retrieval_accuracy = 0%)")
        lines.append("• P(correct_identification | 25% < retrieval_accuracy < 75%)")
        lines.append("• P(correct_identification | retrieval_accuracy ≥ 75%)")
        lines.append(
            "• Improvement = P(correct_id | ≥75% retrieval) - P(correct_id | 0% retrieval)"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append(
            "• Positive improvement: Model benefits from better document retrieval"
        )
        lines.append(
            "• Negative improvement: Model gets confused by more documents (information overload)"
        )
        lines.append(
            "• Near-zero improvement: Retrieval quality doesn't significantly affect identification"
        )
        lines.append("")

        header_parts = [
            "Model",
            "0% Retrieved",
            ">25% Retrieved",
            "≥75% Retrieved",
            "Improvement",
        ]
        lines.append(" | ".join(f"{part:<20}" for part in header_parts))
        lines.append("-" * (22 * len(header_parts) + (len(header_parts) - 1) * 3))

        for i, model in enumerate(model_names):
            if model in conditional_stats:
                zero_pct_rate = (
                    conditional_stats[model]["zero_pct_correct_identification_rate"]
                    * 100
                )
                partial_pct_rate = (
                    conditional_stats[model]["partial_pct_correct_identification_rate"]
                    * 100
                )
                high_pct_rate = (
                    conditional_stats[model]["high_pct_correct_identification_rate"]
                    * 100
                )
                improvement = high_pct_rate - zero_pct_rate

                row_parts = [
                    clean_model_names[i],
                    f"{zero_pct_rate:.1f}%",
                    f"{partial_pct_rate:.1f}%",
                    f"{high_pct_rate:.1f}%",
                    f"{improvement:+.1f}%",
                ]
                lines.append(" | ".join(f"{part:<20}" for part in row_parts))

        lines.append("\n" + "=" * 80 + "\n")

        # Attribution analysis table with documentation
        lines.append("Identification Failure Attribution Analysis")
        lines.append("")
        lines.append("Mathematical Formulation:")
        lines.append(
            "• P(identification_failed | retrieval_accuracy = 0%) = failures_at_0% / total_failures_at_0%"
        )
        lines.append(
            "• P(identification_failed | 25% < retrieval_accuracy < 75%) = failures_partial / total_partial"
        )
        lines.append(
            "• P(identification_failed | retrieval_accuracy ≥ 75%) = failures_at_high / total_at_high"
        )
        lines.append("")
        lines.append("Interpretation:")
        lines.append("• 0% Retrieved: Pure retrieval failure cases")
        lines.append("• >25% Retrieved: Mixed retrieval/understanding issues")
        lines.append(
            "• ≥75% Retrieved: Pure understanding failure cases (good docs, wrong interpretation)"
        )
        lines.append(
            "• Understanding %: Percentage of ID failures that are understanding issues"
        )
        lines.append(
            "  (i.e., model had ≥75% correct documents but still failed identification)"
        )
        lines.append("")

        header_parts = [
            "Model",
            "Total ID Failures",
            "0% Retrieved",
            ">25% Retrieved",
            "≥75% Retrieved",
            "Understanding %",
        ]
        lines.append(" | ".join(f"{part:<20}" for part in header_parts))
        lines.append("-" * (22 * len(header_parts) + (len(header_parts) - 1) * 3))

        for i, model in enumerate(model_names):
            if model in attribution_stats:
                stats = attribution_stats[model]
                row_parts = [
                    clean_model_names[i],
                    str(stats["total_identification_failures"]),
                    str(stats["failures_zero_pct"]),
                    str(stats["failures_partial_pct"]),
                    str(stats["failures_high_pct"]),
                    f"{stats['high_pct_percentage']:.1f}%",
                ]
                lines.append(" | ".join(f"{part:<20}" for part in row_parts))

        lines.append("")
        lines.append("Key Insights:")
        lines.append(
            "• High 'Understanding %' = model good at retrieval but poor at document comprehension"
        )
        lines.append(
            "• Low 'Understanding %' = model failures primarily due to poor document retrieval"
        )
        lines.append(
            "• Models with many '≥75% Retrieved' failures struggle with information synthesis"
        )

        return "\n".join(lines)


def main():
    """Main execution function with argparse support.

    Command-line Arguments:
        --failures-file: Path to extracted_failures.json file
        --output-file: Path to save analysis results (default: failure_mode_analysis.txt)

    Examples:
        >>> # Analyze failures from current directory
        >>> python analyze_failure_modes.py --failures-file extracted_failures.json

        >>> # Custom output location
        >>> python analyze_failure_modes.py --failures-file failures.json --output-file analysis.txt
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze failure modes from extracted failures and generate statistical tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Analyze with default output
    python analyze_failure_modes.py --failures-file extracted_failures.json
    
    # Custom output file
    python analyze_failure_modes.py --failures-file failures.json --output-file my_analysis.txt
    
    # Full path example
    python analyze_failure_modes.py \\
        --failures-file /path/to/extracted_failures.json \\
        --output-file /path/to/failure_mode_analysis.txt
        """,
    )

    parser.add_argument(
        "--failures-file",
        type=str,
        required=True,
        help="Path to extracted_failures.json file (generated by extract_failures.py)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file path for analysis results (default: failure_mode_analysis.txt in same directory as failures file)",
    )

    args = parser.parse_args()

    # Check if failures file exists
    failures_path = Path(args.failures_file)
    if not failures_path.exists():
        print(f"Error: Failures file not found: {args.failures_file}")
        print("Run extract_failures.py first to generate the failures file.")
        return 1

    # Determine output file
    if args.output_file:
        output_file = args.output_file
    else:
        # Default to same directory as failures file
        output_file = str(failures_path.parent / "failure_mode_analysis.txt")

    print(f"Analyzing failures from: {args.failures_file}")
    print(f"Output will be saved to: {output_file}")

    analyzer = FailureModeAnalyzer(args.failures_file)

    print("\nAnalyzing failure modes...")
    stats, model_names = analyzer.analyze_failures()

    print("Generating table...")
    table = analyzer.generate_table(stats, model_names)

    print("Generating summary...")
    summary = analyzer.generate_summary_stats(stats, model_names)

    print("Analyzing retrieval-identification relationship...")
    conditional_stats, raw_counts = (
        analyzer.analyze_retrieval_identification_relationship()
    )

    print("Analyzing identification failure attribution...")
    attribution_stats = analyzer.analyze_identification_failure_attribution()

    print("Generating retrieval analysis tables...")
    retrieval_tables = analyzer.generate_retrieval_analysis_tables(
        conditional_stats, attribution_stats, model_names
    )

    # Output results
    output = f"""
=== FAILURE MODE ANALYSIS TABLE ===

{table}

{summary}

{retrieval_tables}
"""

    # Save to file
    with open(output_file, "w") as f:
        f.write(output)

    print(f"\n✅ Results saved to: {output_file}")
    print(output)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
