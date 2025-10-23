#!/usr/bin/env python3
"""
Robust Batch Evaluation Script for PersonaSim Baselines

This script provides a bulletproof evaluation system that:
1. Handles all formatting variations across baseline models
2. Uses LLM for semantic parameter comparison (not just exact matching)
3. Gracefully handles malformed JSON, connection errors, and edge cases
4. Provides detailed error reporting and recovery strategies
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from evaluation.scoring import ChecklistScorer, ScoringResult
from annotation.annotation_format import Annotation, ActionSelection
from evaluation.llm_json_postprocessor import create_robust_json_parser
from data_generation.utils.clients.openai_client import get_openai_client

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

load_dotenv()


class RobustParameterParser:
    """Handles various parameter formats and attempts intelligent parsing."""

    @staticmethod
    def parse_parameters(raw_params: Any) -> Dict[str, Any]:
        """
        Robustly parse parameters from various formats.

        Args:
            raw_params: Raw parameter data (could be string, dict, etc.)

        Returns:
            Parsed parameters dictionary
        """
        if isinstance(raw_params, dict):
            return raw_params

        if not isinstance(raw_params, str) or not raw_params.strip():
            return {}

        # Try direct JSON parsing first
        try:
            return json.loads(raw_params)
        except json.JSONDecodeError:
            pass

        # Apply common fixes
        fixed_params = raw_params
        fixes_applied = []

        # Fix 1: Remove trailing commas
        if re.search(r",(\s*[}\]])", fixed_params):
            fixed_params = re.sub(r",(\s*[}\]])", r"\1", fixed_params)
            fixes_applied.append("trailing_commas")

        # Fix 2: Fix unescaped quotes in strings
        # Look for patterns like: "text with "quotes" inside"
        def fix_quotes(match):
            content = match.group(1)
            # Escape internal quotes
            content = content.replace('"', '\\"')
            return f'"{content}"'

        if '".*".*".*"' in fixed_params:
            # This is a complex fix - be conservative
            try:
                # Try to find and fix unescaped quotes
                fixed_params = re.sub(r'"([^"]*"[^"]*)"', fix_quotes, fixed_params)
                fixes_applied.append("escaped_quotes")
            except Exception:
                pass

        # Fix 3: Try to complete truncated JSON
        if fixed_params.count("{") > fixed_params.count("}"):
            missing_braces = fixed_params.count("{") - fixed_params.count("}")
            fixed_params += "}" * missing_braces
            fixes_applied.append("added_closing_braces")

        if fixed_params.count("[") > fixed_params.count("]"):
            missing_brackets = fixed_params.count("[") - fixed_params.count("]")
            fixed_params += "]" * missing_brackets
            fixes_applied.append("added_closing_brackets")

        # Fix 4: Handle malformed strings at the end
        if fixed_params.endswith('"') and fixed_params.count('"') % 2 == 1:
            # Find the last complete string and truncate there
            last_complete = fixed_params.rfind('","')
            if last_complete != -1:
                fixed_params = fixed_params[: last_complete + 1] + "}"
                fixes_applied.append("truncated_incomplete_string")

        # Try parsing the fixed version
        try:
            result = json.loads(fixed_params)
            if fixes_applied:
                logger.debug(
                    f"Successfully parsed parameters with fixes: {fixes_applied}"
                )
            return result
        except json.JSONDecodeError as e:
            logger.debug(
                f"Could not parse parameters even after fixes {fixes_applied}: {e}"
            )

            # Last resort: extract key-value pairs using regex
            try:
                kv_pairs = {}
                # Look for "key": "value" or "key": ["value1", "value2"]
                for match in re.finditer(
                    r'"([^"]+)":\s*(?:"([^"]+)"|(\[[^\]]+\]))', fixed_params
                ):
                    key = match.group(1)
                    value = match.group(2) or match.group(3)
                    if value.startswith("["):
                        try:
                            value = json.loads(value)
                        except Exception:
                            pass
                    kv_pairs[key] = value

                if kv_pairs:
                    logger.debug(
                        f"Extracted {len(kv_pairs)} key-value pairs using regex"
                    )
                    return kv_pairs

            except Exception as regex_error:
                logger.debug(f"Regex extraction also failed: {regex_error}")

            # Ultimate fallback: return raw string wrapped
            return {"raw_parameters": raw_params, "parse_error": str(e)}


class RobustBaselineEvaluator:
    """Enhanced evaluator that handles all formatting variations and uses LLM scoring."""

    def __init__(
        self, batch_samples_dir: Path, use_llm_judge: bool = True, max_workers: int = 8
    ):
        """
        Initialize the robust evaluator.

        Args:
            batch_samples_dir: Path to batch_samples directory
            use_llm_judge: Whether to use LLM for judgment (strongly recommended)
            max_workers: Maximum number of parallel workers for evaluation
        """
        self.batch_samples_dir = Path(batch_samples_dir)
        self.inputs_dir = self.batch_samples_dir / "inputs"
        self.gold_outputs_dir = self.batch_samples_dir / "gold_outputs"
        self.predicted_outputs_dir = self.batch_samples_dir / "predicted_outputs"
        self.parser = RobustParameterParser()
        self.max_workers = max_workers
        self.use_llm_judge = use_llm_judge

        # Thread-safe statistics tracking
        self.stats_lock = threading.Lock()
        self.stats = {
            "total_files": 0,
            "connection_errors": 0,
            "json_parse_errors": 0,
            "parameter_parse_errors": 0,
            "successful_parses": 0,
            "llm_judge_calls": 0,
            "llm_judge_errors": 0,
            "llm_json_fixes": 0,
            "llm_json_failures": 0,
            "llm_correction_calls": 0,
        }

        # Initialize scorer with LLM support and robust JSON post-processing
        self.scorer = None  # Will be created per-thread to avoid sharing issues

    def _create_scorer(self) -> ChecklistScorer:
        """Create a scorer instance for the current thread."""
        llm_func = None
        if self.use_llm_judge and os.getenv("OPENAI_API_KEY"):
            client = get_openai_client()

            # Create JSON post-processor for LLM response fixing
            def llm_generate_func_raw(prompt: str) -> str:
                """Raw LLM function without JSON post-processing."""
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": 'You are an expert evaluator. Always respond with valid JSON containing \'judgment\' and \'reasoning\' fields. Format: {"judgment": "CORRECT|PARTIALLY_CORRECT|INCORRECT", "reasoning": "explanation"}',
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.1,
                        response_format={"type": "json_object"},  # Force JSON mode
                    )
                    with self.stats_lock:
                        self.stats["llm_judge_calls"] += 1
                    return response.choices[0].message.content
                except Exception as e:
                    with self.stats_lock:
                        self.stats["llm_judge_errors"] += 1
                    logger.error(f"LLM judge error: {e}")
                    raise

            # LLM correction function (separate from evaluation)
            def llm_correction_func(prompt: str) -> str:
                """LLM function specifically for JSON correction."""
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a JSON formatting specialist. Fix malformed JSON and return only valid JSON.",
                            },
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.0,  # Deterministic for consistent formatting
                        response_format={"type": "json_object"},
                    )
                    with self.stats_lock:
                        self.stats["llm_correction_calls"] += 1
                    return response.choices[0].message.content
                except Exception as e:
                    logger.error(f"LLM JSON correction error: {e}")
                    raise

            # Create robust JSON post-processor with dedicated LLM correction
            json_processor = create_robust_json_parser(llm_correction_func)

            def llm_generate_func(prompt: str) -> str:
                """Enhanced LLM function with robust JSON post-processing."""
                try:
                    # Get raw response
                    raw_response = llm_generate_func_raw(prompt)

                    # Try robust JSON parsing
                    parse_result = json_processor.parse_json_response(
                        raw_response, expected_fields=["judgment", "reasoning"]
                    )

                    if parse_result.success:
                        if parse_result.method_used == "llm_correction":
                            with self.stats_lock:
                                self.stats["llm_json_fixes"] += 1
                            logger.debug("✅ JSON fixed using LLM correction")
                        elif parse_result.method_used != "direct_parse":
                            logger.debug(
                                f"JSON processed using: {parse_result.method_used}"
                            )

                        # Return properly formatted JSON string
                        return json.dumps(parse_result.data)
                    else:
                        with self.stats_lock:
                            self.stats["llm_json_failures"] += 1
                        logger.warning(
                            f"❌ LLM JSON post-processing failed: {parse_result.error}"
                        )

                        # Consistent fallback response
                        fallback_response = {
                            "judgment": "INCORRECT",
                            "reasoning": f"Evaluation failed due to JSON formatting issues. Method attempted: {parse_result.method_used}. Error: {parse_result.error}",
                        }
                        return json.dumps(fallback_response)

                except Exception as e:
                    with self.stats_lock:
                        self.stats["llm_judge_errors"] += 1
                    logger.error(f"Enhanced LLM judge error: {e}")
                    # Create error response in valid JSON format
                    error_response = {
                        "judgment": "INCORRECT",
                        "reasoning": f"LLM evaluation failed: {str(e)}",
                    }
                    return json.dumps(error_response)

            llm_func = llm_generate_func
            logger.info("✅ LLM judge enabled with robust JSON post-processing")
        else:
            logger.warning("⚠️ LLM judge disabled - using exact matching only")

        return ChecklistScorer(llm_generate_func=llm_func)

    def load_predicted_output(
        self, model_name: str, example_id: str
    ) -> Optional[Dict[str, Any]]:
        """Robustly load predicted output for a model and example."""
        # Try different possible file naming conventions
        possible_files = [
            self.predicted_outputs_dir / model_name / f"{example_id}_results.json",
            self.predicted_outputs_dir / model_name / f"{example_id}_output.json",
            self.predicted_outputs_dir / model_name / f"{example_id}.json",
        ]

        for pred_file in possible_files:
            if pred_file.exists():
                try:
                    with self.stats_lock:
                        self.stats["total_files"] += 1
                    with open(pred_file, "r") as f:
                        data = json.load(f)

                    # Handle various error formats
                    if "error" in data:
                        error_msg = data.get("error", "unknown")
                        if any(
                            x in error_msg.lower()
                            for x in ["connection", "timeout", "network"]
                        ):
                            with self.stats_lock:
                                self.stats["connection_errors"] += 1
                        logger.debug(f"Skipping {pred_file.name}: {error_msg}")
                        return None

                    with self.stats_lock:
                        self.stats["successful_parses"] += 1
                    return data

                except json.JSONDecodeError as e:
                    with self.stats_lock:
                        self.stats["json_parse_errors"] += 1
                    logger.debug(f"JSON parse error in {pred_file}: {e}")
                    continue
                except Exception as e:
                    logger.debug(f"Error loading {pred_file}: {e}")
                    continue

        return None

    def create_annotation_from_prediction(
        self, prediction: Dict[str, Any]
    ) -> Optional[Annotation]:
        """Robustly convert a model's prediction to an Annotation object."""
        try:
            # Extract required fields with defaults
            retrieved_docs = prediction.get("retrieved_documents", [])
            bottleneck_desc = prediction.get("bottleneck", "")
            action_data = prediction.get("action", {})

            # Robustly parse parameters
            raw_parameters = action_data.get("parameters", "")
            try:
                parameters_dict = self.parser.parse_parameters(raw_parameters)
                if "parse_error" in parameters_dict:
                    with self.stats_lock:
                        self.stats["parameter_parse_errors"] += 1
            except Exception as e:
                logger.debug(f"Parameter parsing failed: {e}")
                parameters_dict = {
                    "raw_parameters": raw_parameters,
                    "parse_error": str(e),
                }
                with self.stats_lock:
                    self.stats["parameter_parse_errors"] += 1

            # Create action selection with robust format
            action_selection = ActionSelection(
                name=action_data.get("function_name", ""), schema=parameters_dict
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

    def get_model_names(self) -> List[str]:
        """Get all available model names from predicted_outputs."""
        model_dirs = [d for d in self.predicted_outputs_dir.iterdir() if d.is_dir()]
        return [d.name for d in model_dirs]

    def get_example_ids(self) -> List[str]:
        """Get all available example IDs from inputs directory."""
        input_files = list(self.inputs_dir.glob("*_input.json"))
        return [f.stem.replace("_input", "") for f in input_files]

    def evaluate_single_example(
        self, model_name: str, example_id: str
    ) -> Optional[Tuple[ScoringResult, Dict[str, Any]]]:
        """
        Evaluate a single example using robust parsing and LLM scoring.

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
            # Create scorer for this thread (thread-safe)
            scorer = self._create_scorer()

            # Use the enhanced scoring logic with LLM support
            scoring_result = scorer.score_annotation(
                annotation=annotation, output_data=gold_output, input_data=input_data
            )

            metadata = {
                "model_name": model_name,
                "example_id": example_id,
                "has_prediction": True,
                "prediction_keys": list(prediction.keys()),
                "action_name": annotation.action_selection.name,
                "parameters_parsed": "parse_error"
                not in annotation.action_selection.schema,
            }

            return scoring_result, metadata

        except Exception as e:
            logger.error(f"Error scoring {model_name}/{example_id}: {e}")
            return None

    def _evaluate_example_worker(
        self, args: Tuple[str, str]
    ) -> Optional[Tuple[str, Optional[Tuple]]]:
        """Worker function for parallel example evaluation."""
        model_name, example_id = args
        result = self.evaluate_single_example(model_name, example_id)
        return example_id, result

    def evaluate_model_parallel(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all examples for a single model using parallel processing."""
        logger.info(f"🎯 Evaluating model: {model_name} (PARALLEL)")

        example_ids = self.get_example_ids()
        results = []
        successful_evaluations = 0

        # Reset model-specific stats
        model_stats = {
            "connection_errors": 0,
            "parse_errors": 0,
            "successful": 0,
            "action_matches": 0,
            "parameter_scores": [],
        }

        # Prepare arguments for parallel processing
        worker_args = [(model_name, example_id) for example_id in example_ids]

        # Use ThreadPoolExecutor for I/O-bound LLM calls
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_example = {
                executor.submit(self._evaluate_example_worker, args): args[1]
                for args in worker_args
            }

            # Collect results as they complete
            for future in as_completed(future_to_example):
                example_id = future_to_example[future]
                try:
                    example_id_result, result = future.result()

                    if result:
                        scoring_result, metadata = result
                        results.append(
                            {
                                "example_id": example_id_result,
                                "scores": scoring_result.to_dict(),
                                "metadata": metadata,
                            }
                        )
                        successful_evaluations += 1
                        model_stats["successful"] += 1

                        # Track action and parameter performance
                        task_details = scoring_result.task_selection_details
                        if task_details.get("action_correct"):
                            model_stats["action_matches"] += 1
                        model_stats["parameter_scores"].append(
                            task_details.get("parameter_score", 0.0)
                        )

                        # Progress logging
                        if successful_evaluations % 5 == 0:
                            logger.info(
                                f"   ✅ {successful_evaluations}/{len(example_ids)} examples completed for {model_name}"
                            )

                    else:
                        # Track failed evaluations with reason
                        results.append(
                            {
                                "example_id": example_id_result,
                                "scores": None,
                                "metadata": {
                                    "model_name": model_name,
                                    "example_id": example_id_result,
                                    "has_prediction": False,
                                    "failure_reason": "connection_error_or_parse_failure",
                                },
                            }
                        )

                except Exception as exc:
                    logger.error(f"Error evaluating {model_name}/{example_id}: {exc}")
                    results.append(
                        {
                            "example_id": example_id,
                            "scores": None,
                            "metadata": {
                                "model_name": model_name,
                                "example_id": example_id,
                                "has_prediction": False,
                                "failure_reason": f"parallel_processing_error: {exc}",
                            },
                        }
                    )

        # Calculate aggregate statistics (same as before)
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

        # Enhanced reporting
        action_accuracy = model_stats["action_matches"] / max(
            1, model_stats["successful"]
        )
        avg_param_score = sum(model_stats["parameter_scores"]) / max(
            1, len(model_stats["parameter_scores"])
        )

        model_results = {
            "model_name": model_name,
            "total_examples": len(example_ids),
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / len(example_ids) if example_ids else 0.0
            ),
            "aggregate_stats": aggregate_stats,
            "action_accuracy": action_accuracy,
            "average_parameter_score": avg_param_score,
            "individual_results": results,
            "timestamp": datetime.now().isoformat(),
            "parallel_processing": True,
            "max_workers": self.max_workers,
        }

        logger.info(
            f"✅ {model_name}: {successful_evaluations}/{len(example_ids)} examples evaluated (PARALLEL)"
        )
        logger.info(
            f"📊 Overall: {aggregate_stats['overall']['mean']:.3f} | Action Accuracy: {action_accuracy:.1%} | Param Score: {avg_param_score:.3f}"
        )

        return model_results

    def evaluate_model(self, model_name: str) -> Dict[str, Any]:
        """Evaluate all examples for a single model with robust error handling."""
        logger.info(f"🎯 Evaluating model: {model_name}")

        example_ids = self.get_example_ids()
        results = []
        successful_evaluations = 0

        # Reset model-specific stats
        model_stats = {
            "connection_errors": 0,
            "parse_errors": 0,
            "successful": 0,
            "action_matches": 0,
            "parameter_scores": [],
        }

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
                model_stats["successful"] += 1

                # Track action and parameter performance
                task_details = scoring_result.task_selection_details
                if task_details.get("action_correct"):
                    model_stats["action_matches"] += 1
                model_stats["parameter_scores"].append(
                    task_details.get("parameter_score", 0.0)
                )

            else:
                # Track failed evaluations with reason
                results.append(
                    {
                        "example_id": example_id,
                        "scores": None,
                        "metadata": {
                            "model_name": model_name,
                            "example_id": example_id,
                            "has_prediction": False,
                            "failure_reason": "connection_error_or_parse_failure",
                        },
                    }
                )

        # Calculate aggregate statistics
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

        # Enhanced reporting
        action_accuracy = model_stats["action_matches"] / max(
            1, model_stats["successful"]
        )
        avg_param_score = sum(model_stats["parameter_scores"]) / max(
            1, len(model_stats["parameter_scores"])
        )

        model_results = {
            "model_name": model_name,
            "total_examples": len(example_ids),
            "successful_evaluations": successful_evaluations,
            "success_rate": (
                successful_evaluations / len(example_ids) if example_ids else 0.0
            ),
            "aggregate_stats": aggregate_stats,
            "action_accuracy": action_accuracy,
            "average_parameter_score": avg_param_score,
            "individual_results": results,
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(
            f"✅ {model_name}: {successful_evaluations}/{len(example_ids)} examples evaluated"
        )
        logger.info(
            f"📊 Overall: {aggregate_stats['overall']['mean']:.3f} | Action Accuracy: {action_accuracy:.1%} | Param Score: {avg_param_score:.3f}"
        )

        return model_results

    def evaluate_all_models(self) -> Dict[str, Any]:
        """Evaluate all available models with comprehensive error handling."""
        logger.info("🚀 Starting ROBUST evaluation of all baseline models...")

        model_names = self.get_model_names()
        logger.info(f"Found {len(model_names)} models: {model_names}")

        all_results = {}
        summary_stats = []

        for i, model_name in enumerate(model_names, 1):
            logger.info(f"\n{'=' * 80}")
            logger.info(f"🎯 EVALUATING MODEL {i}/{len(model_names)}: {model_name}")
            logger.info(f"{'=' * 80}")

            # Use parallel evaluation for faster processing
            model_results = self.evaluate_model_parallel(model_name)
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
                    "action_accuracy": model_results["action_accuracy"],
                    "parameter_score": model_results["average_parameter_score"],
                }
            )

            # 🔥 IMMEDIATE RESULTS OUTPUT FOR THIS MODEL
            logger.info(f"\n🎉 COMPLETED MODEL: {model_name}")
            logger.info("📊 RESULTS SUMMARY:")
            logger.info(
                f"   Overall Score:     {model_results['aggregate_stats']['overall']['mean']:.3f}"
            )
            logger.info(
                f"   Retrieval Score:   {model_results['aggregate_stats']['retrieval']['mean']:.3f}"
            )
            logger.info(
                f"   Identification:    {model_results['aggregate_stats']['identification']['mean']:.3f}"
            )
            logger.info(
                f"   Task Selection:    {model_results['aggregate_stats']['task_selection']['mean']:.3f}"
            )
            logger.info(f"   Action Accuracy:   {model_results['action_accuracy']:.1%}")
            logger.info(
                f"   Parameter Score:   {model_results['average_parameter_score']:.3f}"
            )
            logger.info(f"   Success Rate:      {model_results['success_rate']:.1%}")

            # Save individual model result immediately
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            individual_file = f"model_result_{model_name}_{timestamp}.json"
            with open(individual_file, "w") as f:
                json.dump({model_name: model_results}, f, indent=2)
            logger.info(f"💾 Individual result saved: {individual_file}")

            # Save cumulative results so far
            cumulative_file = f"cumulative_results_{timestamp}.json"
            cumulative_results = {
                "summary": summary_stats,
                "detailed_results": all_results,
                "evaluation_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "completed_models": i,
                    "total_models": len(model_names),
                    "evaluator_version": "robust_v2.0_with_llm_judge",
                    "scoring_weights": {
                        "retrieval": 0.334,
                        "identification": 0.333,
                        "task_selection": 0.333,
                    },
                    "parsing_stats": self.stats,
                },
            }
            with open(cumulative_file, "w") as f:
                json.dump(cumulative_results, f, indent=2)
            logger.info(f"💾 Cumulative results saved: {cumulative_file}")

            logger.info(f"\n🔄 Progress: {i}/{len(model_names)} models completed")
            if i < len(model_names):
                logger.info(f"⏭️  Next: {model_names[i]}")
                logger.info(
                    f"⏱️  Estimated remaining: ~{(len(model_names) - i) * 2} minutes"
                )
            logger.info(f"{'=' * 80}\n")

        # Create summary DataFrame and display
        summary_df = pd.DataFrame(summary_stats)
        summary_df = summary_df.sort_values("overall_mean", ascending=False)

        final_results = {
            "summary": summary_df.to_dict("records"),
            "detailed_results": all_results,
            "evaluation_metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_models": len(model_names),
                "evaluator_version": "robust_v2.0_with_llm_judge",
                "scoring_weights": {
                    "retrieval": 0.334,
                    "identification": 0.333,
                    "task_selection": 0.333,
                },
                "parsing_stats": self.stats,
            },
        }

        logger.info("🎉 ROBUST Evaluation complete!")
        logger.info("\n📊 PARSING STATISTICS:")
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Connection errors: {self.stats['connection_errors']}")
        logger.info(f"JSON parse errors: {self.stats['json_parse_errors']}")
        logger.info(f"Parameter parse errors: {self.stats['parameter_parse_errors']}")
        logger.info(f"Successful parses: {self.stats['successful_parses']}")
        logger.info(f"LLM judge calls: {self.stats['llm_judge_calls']}")
        logger.info(f"LLM judge errors: {self.stats['llm_judge_errors']}")
        logger.info(f"LLM JSON correction calls: {self.stats['llm_correction_calls']}")
        logger.info(f"LLM JSON fixes applied: {self.stats['llm_json_fixes']}")
        logger.info(f"LLM JSON fix failures: {self.stats['llm_json_failures']}")

        logger.info("\n📊 ENHANCED LEADERBOARD:")
        logger.info(
            f"{'Model':<50} {'Success':<8} {'Overall':<8} {'Retrieval':<10} {'Task Sel':<8} {'Action Acc':<10} {'Param Score':<10}"
        )
        logger.info("-" * 130)

        for row in summary_df.itertuples():
            logger.info(
                f"{row.model_name:<50} {row.success_rate:<8.1%} {row.overall_mean:<8.3f} {row.retrieval_mean:<10.3f} {row.task_selection_mean:<8.3f} {row.action_accuracy:<10.1%} {row.parameter_score:<10.3f}"
            )

        return final_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Robust batch evaluate PersonaSim baseline models with LLM JSON post-processing"
    )
    parser.add_argument(
        "--batch-samples-dir",
        type=str,
        default="/Users/dheeraj/Projects/agent-mono/data_generation/batch_samples",
        help="Path to batch_samples directory",
    )
    parser.add_argument(
        "--disable-llm-judge",
        action="store_true",
        help="Disable LLM judge (not recommended - will use exact matching)",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Use faster but less accurate evaluation (exact matching only)",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Only evaluate first N examples per model (for quick testing)",
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
        help="Specific models to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=8,
        help="Maximum number of parallel workers for evaluation (default: 8)",
    )

    args = parser.parse_args()

    # Initialize robust evaluator
    evaluator = RobustBaselineEvaluator(
        batch_samples_dir=args.batch_samples_dir,
        use_llm_judge=not args.disable_llm_judge,
        max_workers=args.max_workers,
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
        output_file = f"robust_baseline_evaluation_{timestamp}.json"
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
