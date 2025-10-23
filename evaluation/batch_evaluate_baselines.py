#!/usr/bin/env python3
"""
Batch Evaluation Script for PROBE Baselines

This script evaluates baseline model predictions against ground truth labels.

Usage:
    python evaluation/batch_evaluate_baselines.py \\
      --predictions-dir results/my_experiment/inputs \\
      --labels-dir generated_data/TIMESTAMP_batch/outputs \\
      --use-llm-judge

It processes:
- Ground truth labels from labels-dir/ (*_output.json files)
- Model predictions from predictions-dir/[model_name]/ (*_results.json files)
- Input files (optional) from inputs-dir/ (*_input.json files)

Outputs comprehensive evaluation metrics for each model.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

from evaluation.scoring import ChecklistScorer, ScoringResult
from annotation.annotation_format import Annotation, ActionSelection

from data_generation.utils.clients.openai_client import get_openai_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class BaselineEvaluator:
    """Evaluates baseline models using the same logic as the UI."""

    def __init__(
        self,
        predictions_dir: Path,
        labels_dir: Path,
        inputs_dir: Optional[Path] = None,
        use_llm_judge: bool = False,
    ):
        """
        Initialize the evaluator.

        Args:
            predictions_dir: Path to directory containing model predictions (subdirs per model)
            labels_dir: Path to directory containing ground truth labels
            inputs_dir: Optional path to directory containing input files
            use_llm_judge: Whether to use LLM for judgment (requires OpenAI API key)
        """
        self.predicted_outputs_dir = Path(predictions_dir)
        self.gold_outputs_dir = Path(labels_dir)
        self.inputs_dir = Path(inputs_dir) if inputs_dir else self.gold_outputs_dir.parent / "inputs"

        # Initialize scorer
        llm_func = None
        if use_llm_judge and os.getenv("OPENAI_API_KEY"):
            client = get_openai_client()

            def llm_generate_func(prompt: str) -> str:
                response = client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are an expert evaluator. Always respond with valid JSON.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                )
                content = response.choices[0].message.content
                if content is None:
                    logger.warning("LLM returned no content, returning empty JSON object")
                    return "{}"
                return content

            llm_func = llm_generate_func
            logger.info("✅ LLM judge enabled")
        else:
            logger.info("⚠️ LLM judge disabled (exact matching only)")

        self.scorer = ChecklistScorer(llm_generate_func=llm_func)

    def get_model_names(self) -> List[str]:
        """Get all available model names from predicted_outputs."""
        model_dirs = [d for d in self.predicted_outputs_dir.iterdir() if d.is_dir()]
        return [d.name for d in model_dirs]

    def get_example_ids(self) -> List[str]:
        """Get all available example IDs from inputs directory."""
        input_files = list(self.inputs_dir.glob("*_input.json"))
        return [f.stem.replace("_input", "") for f in input_files]

    def load_input_data(self, example_id: str) -> Optional[Dict[str, Any]]:
        """Load input data for an example."""
        input_file = self.inputs_dir / f"{example_id}_input.json"
        if not input_file.exists():
            return None
        try:
            with open(input_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading input {example_id}: {e}")
            return None

    def load_gold_output(self, example_id: str) -> Optional[Dict[str, Any]]:
        """Load gold/ground truth output for an example."""
        gold_file = self.gold_outputs_dir / f"{example_id}_output.json"
        if not gold_file.exists():
            return None
        try:
            with open(gold_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading gold output {example_id}: {e}")
            return None

    def load_predicted_output(
        self, model_name: str, example_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load predicted output for a model and example."""
        # Try different possible file naming conventions
        possible_files = [
            self.predicted_outputs_dir / model_name / f"{example_id}_results.json",
            self.predicted_outputs_dir / model_name / f"{example_id}_output.json",
            self.predicted_outputs_dir / model_name / f"{example_id}.json",
        ]

        for pred_file in possible_files:
            if pred_file.exists():
                try:
                    with open(pred_file, "r") as f:
                        data = json.load(f)
                        # Skip error files
                        if "error" in data and "inference_time" in data:
                            return None
                        return data
                except Exception as e:
                    logger.debug(f"Error loading {pred_file}: {e}")
                    continue
        
        # If individual files not found, check for JSONL batch files (e.g., Anthropic/OpenAI batch outputs)
        model_dir = self.predicted_outputs_dir / model_name
        if model_dir.exists():
            jsonl_files = list(model_dir.glob("*.jsonl"))
            for jsonl_file in jsonl_files:
                prediction = self._extract_from_jsonl_batch(jsonl_file, example_id)
                if prediction:
                    return prediction
        
        return None
    
    def _extract_from_jsonl_batch(self, jsonl_file: Path, example_id: str) -> Optional[Dict[str, Any]]:
        """Extract prediction for a specific example from a JSONL batch file."""
        try:
            with open(jsonl_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    
                    try:
                        entry = json.loads(line)
                        custom_id = entry.get("custom_id", "")
                        
                        # Match custom_id to example_id (handle various formats)
                        # e.g., "bottleneck_001_01_adriano_lujan__mpa_results" matches "bottleneck_001_01_adriano_lujan__mpa"
                        if (example_id in custom_id or 
                            custom_id.replace("_results", "") == example_id or
                            custom_id == f"{example_id}_results"):
                            
                            # Extract prediction based on format
                            # Anthropic format: {"response": {"content": [{"text": "...```json\n{...}\n```"}]}}
                            if "response" in entry and "content" in entry.get("response", {}):
                                return self._extract_from_anthropic_batch_entry(entry)
                            
                            # OpenAI/Together format: {"response": {"body": {"choices": [{"message": {"content": "..."}}]}}}
                            elif "response" in entry and "body" in entry.get("response", {}):
                                return self._extract_from_openai_batch_entry(entry)
                            
                            # Already in standard format
                            elif "retrieved_documents" in entry or "bottleneck" in entry:
                                return entry
                                
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.debug(f"Error reading JSONL file {jsonl_file}: {e}")
        
        return None
    
    def _extract_from_anthropic_batch_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract prediction from Anthropic batch API entry."""
        try:
            import re
            content_list = entry.get("response", {}).get("content", [])
            if not content_list:
                return None
            
            text_content = content_list[0].get("text", "")
            if not text_content:
                return None
            
            # Extract JSON from markdown code blocks
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                return json.loads(json_str)
            
            # Fallback: find JSON without markdown
            first_brace = text_content.find("{")
            last_brace = text_content.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_str = text_content[first_brace:last_brace + 1]
                return json.loads(json_str)
                
        except Exception as e:
            logger.debug(f"Error extracting from Anthropic entry: {e}")
        
        return None
    
    def _extract_from_openai_batch_entry(self, entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract prediction from OpenAI/Together batch API entry."""
        try:
            choices = entry.get("response", {}).get("body", {}).get("choices", [])
            if not choices:
                return None
            
            content = choices[0].get("message", {}).get("content", "")
            if not content:
                return None
            
            # Remove markdown if present
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Extract JSON
            first_brace = content.find("{")
            last_brace = content.rfind("}")
            if first_brace != -1 and last_brace != -1:
                json_str = content[first_brace:last_brace + 1]
                return json.loads(json_str)
                
        except Exception as e:
            logger.debug(f"Error extracting from OpenAI/Together entry: {e}")
        
        return None

    def create_annotation_from_prediction(
        self, prediction: Dict[str, Any]
    ) -> Optional[Annotation]:
        """Convert a model's prediction to an Annotation object for scoring."""
        try:
            # Extract required fields
            retrieved_docs = prediction.get("retrieved_documents", [])
            bottleneck_desc = prediction.get("bottleneck", "")
            action_data = prediction.get("action", {})

            # Parse parameters if they're a JSON string
            parameters = action_data.get("parameters", "")
            if isinstance(parameters, str) and parameters:
                try:
                    parameters_dict = json.loads(parameters)
                except json.JSONDecodeError:
                    logger.debug(f"Failed to parse parameters as JSON: {parameters}")
                    parameters_dict = {"raw_parameters": parameters}
            elif isinstance(parameters, dict):
                parameters_dict = parameters
            else:
                parameters_dict = {}

            # Create action selection with correct format
            action_selection = ActionSelection(
                name=action_data.get("function_name", ""),
                schema=parameters_dict,  # Use the parsed parameters directly
            )

            # Create annotation
            annotation = Annotation(
                retrieved_document_ids=retrieved_docs,
                bottleneck_description=bottleneck_desc,
                action_selection=action_selection,
            )

            return annotation

        except Exception as e:
            logger.error(f"Error creating annotation from prediction: {e}")
            return None

    def evaluate_single_example(
        self, model_name: str, example_id: str
    ) -> Optional[Tuple[ScoringResult, Dict[str, Any]]]:
        """
        Evaluate a single example using the EXACT same logic as the UI.

        Returns:
            Tuple of (ScoringResult, metadata) or None if evaluation failed
        """
        # Load all required data
        input_data = self.load_input_data(example_id)
        gold_output = self.load_gold_output(example_id)
        prediction = self.load_predicted_output(model_name, example_id)

        if not input_data:
            logger.warning(f"Missing input data for {example_id}")
            return None

        if not gold_output:
            logger.warning(f"Missing gold output for {example_id}")
            return None

        if not prediction:
            logger.debug(f"Missing/invalid prediction for {model_name}/{example_id}")
            return None

        # Convert prediction to annotation format
        annotation = self.create_annotation_from_prediction(prediction)
        if not annotation:
            logger.warning(
                f"Could not convert prediction to annotation for {model_name}/{example_id}"
            )
            return None

        try:
            # Use the EXACT same scoring logic as the UI
            scoring_result = self.scorer.score_annotation(
                annotation=annotation, output_data=gold_output, input_data=input_data
            )

            metadata = {
                "model_name": model_name,
                "example_id": example_id,
                "has_prediction": True,
                "prediction_keys": list(prediction.keys()),
            }

            return scoring_result, metadata

        except Exception as e:
            logger.error(f"Error scoring {model_name}/{example_id}: {e}")
            return None

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all examples for a single model."""
        logger.info(f"🎯 Evaluating model: {model_name}")

        example_ids = self.get_example_ids()
        results = []
        successful_evaluations = 0

        for example_id in example_ids:
            result = self.evaluate_single_example(model_name, example_id)
            if result:
                scoring_result, metadata = result
                results.append(
                    {
                        "example_id": example_id,
                        "scores": scoring_result.to_dict(),
                        "metadata": metadata,
                    }
                )
                successful_evaluations += 1
            else:
                # Track failed evaluations
                results.append(
                    {
                        "example_id": example_id,
                        "scores": None,
                        "metadata": {
                            "model_name": model_name,
                            "example_id": example_id,
                            "has_prediction": False,
                        },
                    }
                )

        # Calculate aggregate statistics (same logic as UI)
        valid_results = [r for r in results if r["scores"] is not None]

        if valid_results:
            # Extract scores for aggregation
            retrieval_scores = [r["scores"]["retrieval_score"] for r in valid_results]
            identification_scores = [
                r["scores"]["identification_score"] for r in valid_results
            ]
            task_selection_scores = [
                r["scores"]["task_selection_score"] for r in valid_results
            ]
            overall_scores = [r["scores"]["overall_score"] for r in valid_results]

            # Calculate statistics
            def calc_stats(scores: List[float]) -> Dict[str, float]:
                if not scores:
                    return {
                        "mean": 0.0,
                        "median": 0.0,
                        "std": 0.0,
                        "min": 0.0,
                        "max": 0.0,
                    }

                scores_series = pd.Series(scores)
                return {
                    "mean": float(scores_series.mean()),
                    "median": float(scores_series.median()),
                    "std": float(scores_series.std()),
                    "min": float(scores_series.min()),
                    "max": float(scores_series.max()),
                }

            aggregate_stats = {
                "retrieval": calc_stats(retrieval_scores),
                "identification": calc_stats(identification_scores),
                "task_selection": calc_stats(task_selection_scores),
                "overall": calc_stats(overall_scores),
            }
        else:
            aggregate_stats = {
                "retrieval": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "identification": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "task_selection": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
                "overall": {
                    "mean": 0.0,
                    "median": 0.0,
                    "std": 0.0,
                    "min": 0.0,
                    "max": 0.0,
                },
            }

        model_results = {
            "model_name": model_name,
            "total_examples": len(example_ids),
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / len(example_ids) if example_ids else 0.0
            ),
            "aggregate_stats": aggregate_stats,
            "individual_results": results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ {model_name}: {successful_evaluations}/{len(example_ids)} examples evaluated"
        )
        logger.info(
            f"📊 Overall Score: {aggregate_stats['overall']['mean']:.3f} ± {aggregate_stats['overall']['std']:.3f}"
        )

        return model_results

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all available models."""
        logger.info("🚀 Starting evaluation of all baseline models...")

        model_names = self.get_model_names()
        logger.info(f"Found {len(model_names)} models: {model_names}")

        all_results = {}
        summary_stats = []

        for model_name in model_names:
            model_results = self.evaluate_model(model_name)
            all_results[model_name] = model_results

            # Add to summary
            summary_stats.append(
                {
                    "model_name": model_name,
                    "successful_evaluations": model_results["successful_evaluations"],
                    "success_rate": model_results["success_rate"],
                    "overall_mean": model_results["aggregate_stats"]["overall"]["mean"],
                    "overall_std": model_results["aggregate_stats"]["overall"]["std"],
                    "retrieval_mean": model_results["aggregate_stats"]["retrieval"][
                        "mean"
                    ],
                    "identification_mean": model_results["aggregate_stats"][
                        "identification"
                    ]["mean"],
                    "task_selection_mean": model_results["aggregate_stats"][
                        "task_selection"
                    ]["mean"],
                }
            )

        # Create summary DataFrame and display
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values("overall_mean", ascending=False)

        final_results = {
            "summary": summary_df.to_dict("records"),
            "detailed_results": all_results,
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(model_names),
                "evaluator_version": "ui_compatible_v1.0",
                "scoring_weights": {
                    "retrieval": 0.334,
                    "identification": 0.333,
                    "task_selection": 0.333,
                },
            },
        }

        logger.info("🎉 Evaluation complete!")
        logger.info("\n📊 SUMMARY LEADERBOARD:")
        logger.info(
            f"{'Model':<50} {'Success Rate':<12} {'Overall':<10} {'Retrieval':<10} {'Identification':<13} {'Task Selection':<13}"
        )
        logger.info("-" * 120)

        for row in summary_df.itertuples():
            logger.info(
                f"{row.model_name:<50} {row.success_rate:<12.1%} {row.overall_mean:<10.3f} {row.retrieval_mean:<10.3f} {row.identification_mean:<13.3f} {row.task_selection_mean:<13.3f}"
            )

        return final_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate model predictions against ground truth labels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python evaluation/batch_evaluate_baselines.py \\
    --predictions-dir results/my_experiment/inputs \\
    --labels-dir generated_data/20251014_151439_batch/outputs

  # With LLM judge for semantic matching
  python evaluation/batch_evaluate_baselines.py \\
    --predictions-dir results/my_experiment/inputs \\
    --labels-dir generated_data/20251014_151439_batch/outputs \\
    --use-llm-judge \\
    --output-file results/evaluation.json

  # Evaluate specific models only
  python evaluation/batch_evaluate_baselines.py \\
    --predictions-dir results/my_experiment/inputs \\
    --labels-dir generated_data/20251014_151439_batch/outputs \\
    --models react_agent_gpt-4.1 baseline_agent_gpt-5
        """,
    )
    parser.add_argument(
        "--predictions-dir",
        type=str,
        required=True,
        help="Directory containing model predictions (with subdirs per model)",
    )
    parser.add_argument(
        "--labels-dir",
        type=str,
        required=True,
        help="Directory containing ground truth labels (*_output.json files)",
    )
    parser.add_argument(
        "--inputs-dir",
        type=str,
        default=None,
        help="Optional directory containing input files (*_input.json). If not specified, will look in labels-dir/../inputs",
    )
    parser.add_argument(
        "--use-llm-judge",
        action="store_true",
        help="Use LLM for judgment (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file for results (default: auto-generated)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Specific models to evaluate (default: all models in predictions-dir)",
    )

    args = parser.parse_args()

    # Initialize evaluator
    evaluator = BaselineEvaluator(
        predictions_dir=args.predictions_dir,
        labels_dir=args.labels_dir,
        inputs_dir=args.inputs_dir,
        use_llm_judge=args.use_llm_judge,
    )

    # Run evaluation
    if args.models:
        logger.info(f"Evaluating specific models: {args.models}")
        all_results = {"detailed_results": {}, "summary": []}
        for model in args.models:
            if model in evaluator.get_model_names():
                result = evaluator.evaluate_model(model)
                all_results["detailed_results"][model] = result
            else:
                logger.error(f"Model {model} not found!")
    else:
        all_results = evaluator.evaluate_all_models()

    # Save results
    if args.output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"baseline_evaluation_results_{timestamp}.json"
    else:
        output_file = args.output_file

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"💾 Results saved to: {output_file}")

    # Also save summary CSV
    if "summary" in all_results:
        csv_file = output_file.replace(".json", "_summary.csv")
        summary_df = pd.DataFrame(all_results["summary"])
        summary_df.to_csv(csv_file, index=False)
        logger.info(f"📊 Summary saved to: {csv_file}")


if __name__ == "__main__":
    main()
