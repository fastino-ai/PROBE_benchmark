#!/usr/bin/env python3
"""
Inference orchestrator for baseline testing against benchmark dataset.

This script loads test samples, instantiates agents from baselines/,
and collects outputs to save in results files.
"""

import os
import sys
import json
import argparse
import importlib.util
import time
import random
import threading
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dataclasses import dataclass
from collections import defaultdict

logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting per model provider."""

    requests_per_minute: int = 60
    requests_per_second: float = 1.0
    max_concurrent: int = 5
    backoff_base: float = 1.0
    backoff_max: float = 300.0


class ModelRateLimiter:
    """Rate limiter that tracks requests per model/provider."""

    def __init__(self):
        self.model_configs = defaultdict(lambda: RateLimitConfig())
        self.request_times = defaultdict(list)
        self.active_requests = defaultdict(int)
        self.locks = defaultdict(threading.Lock)

        # Set specific rate limits for different providers
        self.setup_provider_limits()

    def setup_provider_limits(self):
        """Configure rate limits for different providers."""
        # OpenAI models - Allow more concurrent requests
        openai_config = RateLimitConfig(
            requests_per_minute=60,
            requests_per_second=10.0,  # Allow more requests per second
            max_concurrent=20,  # Allow more concurrent requests
        )

        # Anthropic models
        anthropic_config = RateLimitConfig(
            requests_per_minute=50, requests_per_second=2.0, max_concurrent=5
        )

        # Google/Gemini models
        google_config = RateLimitConfig(
            requests_per_minute=60, requests_per_second=2.0, max_concurrent=8
        )

        # Together AI models
        together_config = RateLimitConfig(
            requests_per_minute=100, requests_per_second=5.0, max_concurrent=15
        )

        # Apply configs based on model patterns
        self.provider_configs = {
            "anthropic": anthropic_config,
            "google": google_config,
            "gemini": google_config,
            "together_ai": together_config,
            "openai": openai_config,
            "gpt": openai_config,  # For string model names like 'gpt-4o'
        }

    def get_provider_from_model(self, model: Union[str, Any]) -> str:
        """Extract provider name from model."""
        if isinstance(model, str):
            model_lower = model.lower()
            if "gpt" in model_lower or "openai" in model_lower:
                return "openai"
            elif "claude" in model_lower or "anthropic" in model_lower:
                return "anthropic"
            elif "gemini" in model_lower or "google" in model_lower:
                return "google"
            else:
                return "unknown"
        else:
            # For LitellmModel instances
            model_str = str(model).lower()
            if "anthropic" in model_str:
                return "anthropic"
            elif "google" in model_str or "gemini" in model_str:
                return "google"
            elif "together_ai" in model_str:
                return "together_ai"
            else:
                return "unknown"

    def get_config(self, model: Union[str, Any]) -> RateLimitConfig:
        """Get rate limit config for a model."""
        provider = self.get_provider_from_model(model)
        return self.provider_configs.get(provider, RateLimitConfig())

    def can_make_request(self, model: Union[str, Any]) -> bool:
        """Check if we can make a request for this model."""
        model_key = str(model)
        config = self.get_config(model)

        with self.locks[model_key]:
            now = time.time()

            # Clean old request times (older than 1 minute)
            cutoff = now - 60
            self.request_times[model_key] = [
                t for t in self.request_times[model_key] if t > cutoff
            ]

            # Check concurrent requests
            if self.active_requests[model_key] >= config.max_concurrent:
                return False

            # Check requests per minute
            if len(self.request_times[model_key]) >= config.requests_per_minute:
                return False

            # Check requests per second
            recent_cutoff = now - 1.0
            recent_requests = [
                t for t in self.request_times[model_key] if t > recent_cutoff
            ]
            if len(recent_requests) >= config.requests_per_second:
                return False

            return True

    def record_request_start(self, model: Union[str, Any]):
        """Record that a request is starting."""
        model_key = str(model)
        with self.locks[model_key]:
            self.request_times[model_key].append(time.time())
            self.active_requests[model_key] += 1

    def record_request_end(self, model: Union[str, Any]):
        """Record that a request has ended."""
        model_key = str(model)
        with self.locks[model_key]:
            self.active_requests[model_key] = max(
                0, self.active_requests[model_key] - 1
            )

    def wait_for_slot(
        self,
        model: Union[str, Any],
        timeout: float = 60.0,
        disable_rate_limiting: bool = False,
    ):
        """Wait until we can make a request for this model."""
        if disable_rate_limiting:
            return True  # Skip rate limiting entirely

        start_time = time.time()
        config = self.get_config(model)

        while time.time() - start_time < timeout:
            if self.can_make_request(model):
                return True

            # Calculate sleep time based on rate limits
            sleep_time = min(1.0 / config.requests_per_second, 1.0)
            time.sleep(sleep_time)

        return False


# Global rate limiter instance
rate_limiter = ModelRateLimiter()


def exponential_backoff_with_jitter(
    attempt: int, base: float = 1.0, max_delay: float = 300.0
) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(base * (2**attempt), max_delay)
    jitter = random.uniform(0.1, 0.3) * delay
    return delay + jitter


def clean_filename_stem(file_stem: str) -> str:
    """Remove '_input' suffix from filename stem if present."""
    if file_stem.endswith("_input"):
        return file_stem[:-6]  # Remove '_input' (6 characters)
    return file_stem


def load_sample(file_path: str) -> Dict[str, Any]:
    """Load a test sample from file."""
    with open(file_path, "r") as f:
        return json.load(f)


def save_results(results: Dict[str, Any], output_path: str):
    """Save inference results to file."""
    try:
        print(f"Attempting to save results to: {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"✓ Successfully saved results to: {output_path}")
    except Exception as e:
        print(f"✗ Error saving results to {output_path}: {str(e)}")
        raise


def load_agent(agent_name: str):
    """Load an agent class from the baselines/agentic directory.

    Args:
        agent_name: Name of the agent (e.g., 'baseline_agent', 'react_agent', 'reflexion_agent', 'rewoo_agent')
                   Can also use short names: 'baseline', 'react', 'reflexion', 'rewoo'
                   Can also include path like 'agentic/baseline_agent' which will be normalized

    Returns:
        Agent class from the specified module

    Raises:
        FileNotFoundError: If agent file doesn't exist
        AttributeError: If Agent class not found in module
        ImportError: If agent cannot be imported
    """
    # Remove any path components and .py extension
    agent_name_clean = (
        agent_name.replace("/", "_").replace("\\", "_").replace(".py", "")
    )

    # Remove 'agentic' prefix if present (handles cases like 'agentic/rewoo_agent')
    if agent_name_clean.startswith("agentic_"):
        agent_name_clean = agent_name_clean[8:]  # Remove 'agentic_' prefix

    # Map short names to full agent file names
    agent_name_map = {
        "baseline": "baseline_agent",
        "react": "react_agent",
        "reflexion": "reflexion_agent",
        "rewoo": "rewoo_agent",
    }

    # Normalize agent name
    normalized_name = agent_name_map.get(agent_name_clean, agent_name_clean)

    # Ensure it has _agent suffix (but don't duplicate if already present)
    if not normalized_name.endswith("_agent"):
        normalized_name = f"{normalized_name}_agent"

    # Ensure project root is in sys.path for imports to work
    project_root = Path(__file__).parent.parent.parent.absolute()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Use regular import instead of importlib for better error handling
    try:
        module_path = f"baselines.agentic.{normalized_name}"
        module = __import__(module_path, fromlist=["Agent"])

        if hasattr(module, "Agent"):
            return module.Agent
        else:
            raise AttributeError(f"No 'Agent' class found in {module_path}")
    except ImportError as e:
        # Fallback to file-based loading if import fails
        baselines_dir = Path("baselines") / "agentic"
        agent_file = baselines_dir / f"{normalized_name}.py"

        if not agent_file.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_file}") from e

        raise ImportError(f"Could not import {module_path}: {e}") from e


def filter_memory_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter memory docs to only include id, type, payload fields, clean up IDs, and shuffle."""
    filtered_docs = []
    for doc in docs:
        filtered_doc = {
            "id": doc.get("id", ""),
            "type": doc.get("type", ""),
            "payload": doc.get("payload", {}),
        }

        # Remove "distractor_" prefix from ID if present
        if filtered_doc["id"].startswith("distractor_"):
            filtered_doc["id"] = filtered_doc["id"][len("distractor_") :]

        filtered_docs.append(filtered_doc)

    # Shuffle the documents to randomize order
    random.shuffle(filtered_docs)

    return filtered_docs


def run_inference_with_retry(
    agent_class,
    sample_data: Dict[str, Any],
    model: Union[str, Any] = "gpt-4o",
    agent_instance=None,
    max_retries: int = 3,
    disable_rate_limiting: bool = False,
) -> Dict[str, Any]:
    """Run inference with exponential backoff and rate limiting.

    Args:
        agent_class: The agent class to instantiate
        sample_data: Test sample data
        model: Either a string model name or a LitellmModel instance
        agent_instance: Optional pre-instantiated agent
        max_retries: Maximum number of retry attempts
        disable_rate_limiting: Whether to disable rate limiting for testing
    """
    # Wait for rate limit slot
    if not rate_limiter.wait_for_slot(
        model, disable_rate_limiting=disable_rate_limiting
    ):
        raise RuntimeError(f"Rate limit timeout for model {model}")

    # Record request start (unless rate limiting is disabled)
    if not disable_rate_limiting:
        rate_limiter.record_request_start(model)

    try:
        for attempt in range(max_retries + 1):
            try:
                return run_inference(agent_class, sample_data, model, agent_instance)
            except Exception as e:
                if attempt == max_retries:
                    # Last attempt failed, re-raise the exception
                    raise e

                # Check if it's a rate limit or transient error
                error_str = str(e).lower()
                if any(
                    keyword in error_str
                    for keyword in [
                        "rate limit",
                        "too many requests",
                        "quota",
                        "timeout",
                        "429",
                        "503",
                        "502",
                    ]
                ):
                    # Calculate exponential backoff delay
                    delay = exponential_backoff_with_jitter(attempt)
                    print(f"Retrying after {delay:.1f}s due to: {str(e)}")
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    raise e

    finally:
        # Always record request end (unless rate limiting is disabled)
        if not disable_rate_limiting:
            rate_limiter.record_request_end(model)


def run_inference(
    agent_class,
    sample_data: Dict[str, Any],
    model: Union[str, Any] = "gpt-4o",
    agent_instance=None,
) -> Dict[str, Any]:
    """Run inference using the specified agent on a test sample.

    Args:
        agent_class: The agent class to instantiate
        sample_data: Test sample data
        model: Either a string model name or a LitellmModel instance
        agent_instance: Optional pre-instantiated agent
    """
    # Prepare inputs for the agent
    if "data_points" in sample_data:
        # New format: single data_points array
        raw_memory_docs = sample_data.get("data_points", [])
    else:
        # Old format: separate true_positives and distractors arrays
        raw_memory_docs = sample_data.get("true_positives", []) + sample_data.get(
            "distractors", []
        )

    memory_docs = filter_memory_docs(raw_memory_docs)
    world_model = sample_data.get("world_model", {})
    persona = sample_data.get("persona", {})

    # Use provided agent instance or create new one
    if agent_instance is None:
        agent = agent_class(model=model)
    else:
        agent = agent_instance

    # Run agent inference
    start_time = time.time()
    results = agent.run(memory=memory_docs, world_model=world_model, persona=persona)
    end_time = time.time()

    # Add timing info
    results["inference_time"] = end_time - start_time

    return results


def check_existing_results(
    sample_files: List[Path],
    output_dir: str,
    agent_name: str,
    model: Union[str, Any],
    data_dir_name: str = None,
) -> List[Path]:
    """Check which sample files already have results and return list of files to process."""
    # Handle both string models and LitellmModel instances
    if isinstance(model, str):
        model_str = model.replace("/", "_")
    else:
        # For LitellmModel instances, get the model string
        model_str = (
            str(model).replace("/", "_").replace("LitellmModel(", "").replace(")", "")
        )

    agent_model_dir = f"{agent_name}_{model_str}"
    if data_dir_name:
        results_dir = Path(output_dir) / data_dir_name / agent_model_dir
    else:
        results_dir = Path(output_dir) / agent_model_dir

    files_to_process = []
    skipped_files = []

    for sample_file in sample_files:
        cleaned_stem = clean_filename_stem(sample_file.stem)
        expected_result_file = results_dir / f"{cleaned_stem}_results.json"
        if expected_result_file.exists():
            skipped_files.append(sample_file)
            print(f"Skipping {sample_file.name} (results already exist)")
        else:
            files_to_process.append(sample_file)

    if skipped_files:
        print(
            f"Found existing results for {len(skipped_files)} files, processing remaining {len(files_to_process)} files"
        )

    return files_to_process


def run_multi_model_inference(
    agent_class,
    samples: List[Dict[str, Any]],
    models: List[Union[str, Any]],
    max_workers: int = 8,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run inference on multiple samples with multiple models concurrently.

    Args:
        agent_class: The agent class to use
        samples: List of sample data
        models: List of models to test
        max_workers: Maximum number of concurrent workers across all models

    Returns:
        Dictionary mapping model names to their results
    """
    results_by_model = {}

    # Calculate workers per model to balance load
    workers_per_model = max(1, max_workers // len(models))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all model/sample combinations
        future_to_info = {}

        for model in models:
            model_key = str(model)
            results_by_model[model_key] = []

            # Create agent instances for this model
            model_agents = []
            for _ in range(workers_per_model):
                agent = agent_class(model=model)
                model_agents.append(agent)

            # Submit tasks for this model
            for i, sample in enumerate(samples):
                agent = model_agents[i % len(model_agents)]
                future = executor.submit(
                    run_inference_with_retry, agent_class, sample, model, agent
                )
                future_to_info[future] = (model_key, i, sample)

        # Collect results as they complete
        for future in as_completed(future_to_info):
            model_key, i, sample = future_to_info[future]
            try:
                result = future.result()
                results_by_model[model_key].append((i, result))
                print(f"✓ Completed {model_key} sample {i+1}/{len(samples)}")
            except Exception as e:
                print(f"✗ Failed {model_key} sample {i+1}: {str(e)}")
                # Add error result
                error_result = {
                    "error": str(e),
                    "inference_time": 0,
                    "timestamp": time.time(),
                }
                results_by_model[model_key].append((i, error_result))

    # Sort results by original order for each model
    for model_key in results_by_model:
        results_by_model[model_key].sort(key=lambda x: x[0])
        results_by_model[model_key] = [
            result for _, result in results_by_model[model_key]
        ]

    return results_by_model


def run_batch_inference(
    agent_class,
    samples: List[Dict[str, Any]],
    model: Union[str, Any] = "gpt-4o",
    max_workers: int = 4,
    disable_rate_limiting: bool = False,
    agent_name: str = "agent",
    output_dir: str = "results",
    data_dir_name: str = None,
) -> List[Dict[str, Any]]:
    """Run inference on multiple samples concurrently with incremental saving.

    Args:
        agent_class: The agent class to use
        samples: List of sample data
        model: Either a string model name or a LitellmModel instance
        max_workers: Maximum number of concurrent workers
        agent_name: Name of the agent for output directory
        output_dir: Base output directory
        data_dir_name: Nested data directory name
    """
    print(f"Starting batch inference with {max_workers} concurrent workers")

    # Create agent instances for reuse
    agents = []
    for _ in range(max_workers):
        agent = agent_class(model=model)
        agents.append(agent)

    # Setup output directory structure
    if isinstance(model, str):
        model_str = model.replace("/", "_")
    else:
        model_str = (
            str(model).replace("/", "_").replace("LitellmModel(", "").replace(")", "")
        )

    agent_model_dir = f"{agent_name}_{model_str}"

    results = []
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_sample = {}
        for i, sample in enumerate(samples):
            agent = agents[i % len(agents)]  # Round-robin agent assignment
            future = executor.submit(
                run_inference_with_retry,
                agent_class,
                sample,
                model,
                agent,
                3,
                disable_rate_limiting,
            )
            future_to_sample[future] = (i, sample)

        print(f"Submitted {len(samples)} tasks to thread pool")

        # Collect results as they complete and save immediately
        completed_count = 0
        for future in as_completed(future_to_sample):
            i, sample = future_to_sample[future]
            try:
                result = future.result()
                results.append((i, result))
                completed_count += 1
                elapsed = time.time() - start_time
                print(
                    f"✓ Completed sample {i+1}/{len(samples)} (worker {completed_count}, elapsed: {elapsed:.1f}s)"
                )

                # Save result immediately
                file_stem = sample["_file_stem"]
                if data_dir_name:
                    output_path = os.path.join(
                        output_dir,
                        data_dir_name,
                        agent_model_dir,
                        f"{file_stem}_results.json",
                    )
                else:
                    output_path = os.path.join(
                        output_dir, agent_model_dir, f"{file_stem}_results.json"
                    )
                save_results(result, output_path)

            except Exception as e:
                print(f"✗ Failed sample {i+1}: {str(e)}")
                # Add error result and save it too
                error_result = {
                    "error": str(e),
                    "inference_time": 0,
                    "timestamp": time.time(),
                }
                results.append((i, error_result))
                completed_count += 1

                # Save error result
                file_stem = sample["_file_stem"]
                if data_dir_name:
                    output_path = os.path.join(
                        output_dir,
                        data_dir_name,
                        agent_model_dir,
                        f"{file_stem}_results.json",
                    )
                else:
                    output_path = os.path.join(
                        output_dir, agent_model_dir, f"{file_stem}_results.json"
                    )
                save_results(error_result, output_path)

    total_time = time.time() - start_time
    print(f"Batch processing completed in {total_time:.2f}s with {max_workers} workers")

    # Sort results by original order
    results.sort(key=lambda x: x[0])
    return [result for _, result in results]


def main():
    parser = argparse.ArgumentParser(description="Run inference on test samples")
    parser.add_argument(
        "--data_dir", default="data", help="Directory containing test samples"
    )
    parser.add_argument(
        "--output_dir", default="results", help="Directory to save results"
    )
    parser.add_argument("--sample", help="Process specific sample file")
    parser.add_argument(
        "--agent",
        default="baseline_agent",
        help="Agent to use from baselines/agentic/ directory (e.g., baseline_agent, react_agent, reflexion_agent, rewoo_agent)",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help='LLM model to use (default: gpt-4o) or model path like "MODELS.ANTHROPIC.CLAUDE_3_5_SONNET"',
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Multiple models to test simultaneously (e.g., --models gpt-4o MODELS.ANTHROPIC.CLAUDE_4_SONNET)",
    )
    parser.add_argument(
        "--batch", action="store_true", help="Use batch processing for faster inference"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=8,
        help="Maximum concurrent workers for batch processing",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip files that already have results (default: enabled)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing all files, ignoring existing results",
    )
    parser.add_argument(
        "--rate_limit_override",
        help="Override rate limits in format provider:rpm:rps:concurrent (e.g., openai:100:5:10)",
    )
    parser.add_argument(
        "--multi_model",
        action="store_true",
        help="Run multiple models simultaneously (use with --models)",
    )
    parser.add_argument(
        "--disable_rate_limiting",
        action="store_true",
        help="Disable rate limiting entirely for maximum concurrency (use with caution)",
    )

    args = parser.parse_args()

    # Handle rate limit overrides
    if args.rate_limit_override:
        try:
            provider, rpm, rps, concurrent = args.rate_limit_override.split(":")
            override_config = RateLimitConfig(
                requests_per_minute=int(rpm),
                requests_per_second=float(rps),
                max_concurrent=int(concurrent),
            )
            rate_limiter.provider_configs[provider] = override_config
            print(
                f"Override rate limits for {provider}: {rpm} RPM, {rps} RPS, {concurrent} concurrent"
            )
        except ValueError:
            print(
                "Invalid rate limit override format. Use: provider:rpm:rps:concurrent"
            )
            return

    def parse_model(model_str: str):
        """Parse a model string that could be a MODELS path."""
        if model_str.startswith("MODELS."):
            try:
                # Import the models module and get the model
                from baselines.agentic.models import MODELS

                model_parts = model_str.split(".")[1:]  # Remove 'MODELS' prefix
                model_obj = MODELS
                for part in model_parts:
                    model_obj = getattr(model_obj, part)
                print(f"Using model from MODELS: {model_obj}")
                return model_obj
            except (ImportError, AttributeError) as e:
                print(f"Error loading model {model_str}: {e}")
                print("Falling back to string model name")
                return model_str
        return model_str

    # Parse models (single model or multiple models)
    if args.multi_model and args.models:
        models = [parse_model(m) for m in args.models]
        print(
            f"Using multi-model mode with {len(models)} models: {[str(m) for m in models]}"
        )
    elif args.models:
        # Multiple models but not multi-model mode - run sequentially
        models = [parse_model(m) for m in args.models]
        print(f"Using sequential multi-model mode with {len(models)} models")
    else:
        # Single model mode
        model = parse_model(args.model)
        models = [model]
        print(f"Using single model: {model}")

    # Load the specified agent
    agent_class = load_agent(args.agent)
    print(f"Loaded agent: {args.agent}")

    # Extract data directory name for nested results structure
    data_dir_path = Path(args.data_dir)
    data_dir_name = data_dir_path.name if data_dir_path.name != "data" else None

    if args.sample:
        # Process single sample
        sample_path = args.sample
        sample_name = clean_filename_stem(Path(sample_path).stem)

        print(f"Processing sample: {sample_path}")
        sample_data = load_sample(sample_path)

        # Process with each model
        for model in models:
            print(f"Running {model} on {sample_path}")
            results = run_inference_with_retry(
                agent_class, sample_data, model, max_retries=args.max_retries
            )

            # Create nested folder structure: output_dir/agent_model/data_dir/sample_results.json
            if isinstance(model, str):
                model_str = model.replace("/", "_")
            else:
                model_str = (
                    str(model)
                    .replace("/", "_")
                    .replace("LitellmModel(", "")
                    .replace(")", "")
                )

            agent_model_dir = f"{args.agent}_{model_str}"
            if data_dir_name:
                output_path = os.path.join(
                    args.output_dir,
                    data_dir_name,
                    agent_model_dir,
                    f"{sample_name}_results.json",
                )
            else:
                output_path = os.path.join(
                    args.output_dir, agent_model_dir, f"{sample_name}_results.json"
                )
            save_results(results, output_path)
            print(f"Results saved to: {output_path}")
            print(f"Inference time: {results.get('inference_time', 0):.2f}s")

    else:
        # Process all samples in data directory
        data_dir = Path(args.data_dir)
        sample_files = list(data_dir.glob("*.txt")) + list(data_dir.glob("*.json"))
        sample_files.sort()  # Ensure alphabetical order

        print(f"Found {len(sample_files)} sample files")

        if args.multi_model and len(models) > 1:
            # Multi-model simultaneous processing
            print(
                f"Using multi-model processing with {len(models)} models and {args.max_workers} workers"
            )

            # Load all samples
            samples = []
            for sample_file in sample_files:
                sample_data = load_sample(str(sample_file))
                sample_data["_file_stem"] = clean_filename_stem(sample_file.stem)
                samples.append(sample_data)

            start_time = time.time()
            results_by_model = run_multi_model_inference(
                agent_class, samples, models, args.max_workers
            )
            total_time = time.time() - start_time

            # Save results for each model
            for model in models:
                model_key = str(model)
                if isinstance(model, str):
                    model_str = model.replace("/", "_")
                else:
                    model_str = (
                        str(model)
                        .replace("/", "_")
                        .replace("LitellmModel(", "")
                        .replace(")", "")
                    )

                agent_model_dir = f"{args.agent}_{model_str}"
                for i, results in enumerate(results_by_model[model_key]):
                    file_stem = samples[i]["_file_stem"]
                    if data_dir_name:
                        output_path = os.path.join(
                            args.output_dir,
                            data_dir_name,
                            agent_model_dir,
                            f"{file_stem}_results.json",
                        )
                    else:
                        output_path = os.path.join(
                            args.output_dir,
                            agent_model_dir,
                            f"{file_stem}_results.json",
                        )
                    save_results(results, output_path)

            print(f"Multi-model processing completed in {total_time:.2f}s")
            print(
                f"Average time per sample per model: {total_time/(len(samples)*len(models)):.2f}s"
            )

        elif args.batch and len(sample_files) > 1:
            # Batch processing (sequential for multiple models)
            for model in models:
                # Check for existing results and filter out already processed files (unless --force is used)
                current_sample_files = sample_files
                if not args.force:
                    current_sample_files = check_existing_results(
                        sample_files, args.output_dir, args.agent, model, data_dir_name
                    )

                    if len(current_sample_files) == 0:
                        print(
                            f"All files for {model} have already been processed! Use --force to reprocess."
                        )
                        continue

                print(
                    f"Using batch processing for {model} with {args.max_workers} workers"
                )
                samples = []
                for sample_file in current_sample_files:
                    sample_data = load_sample(str(sample_file))
                    sample_data["_file_stem"] = clean_filename_stem(sample_file.stem)
                    samples.append(sample_data)

                start_time = time.time()
                results_list = run_batch_inference(
                    agent_class,
                    samples,
                    model,
                    args.max_workers,
                    args.disable_rate_limiting,
                    args.agent,
                    args.output_dir,
                    data_dir_name,
                )
                total_time = time.time() - start_time

                # Results are already saved incrementally, just report completion
                print(
                    f"All {len(results_list)} results saved incrementally during batch processing"
                )

                print(f"Batch processing for {model} completed in {total_time:.2f}s")
                print(f"Average time per sample: {total_time/len(samples):.2f}s")

        else:
            # Sequential processing (for multiple models)
            for model in models:
                # Check for existing results and filter out already processed files (unless --force is used)
                current_sample_files = sample_files
                if not args.force:
                    current_sample_files = check_existing_results(
                        sample_files, args.output_dir, args.agent, model, data_dir_name
                    )

                    if len(current_sample_files) == 0:
                        print(
                            f"All files for {model} have already been processed! Use --force to reprocess."
                        )
                        continue

                total_time = 0
                if isinstance(model, str):
                    model_str = model.replace("/", "_")
                else:
                    model_str = (
                        str(model)
                        .replace("/", "_")
                        .replace("LitellmModel(", "")
                        .replace(")", "")
                    )

                agent_model_dir = f"{args.agent}_{model_str}"
                print(f"Sequential processing for {model}")
                for sample_file in current_sample_files:
                    print(f"Processing: {sample_file}")
                    sample_data = load_sample(str(sample_file))
                    results = run_inference_with_retry(
                        agent_class, sample_data, model, max_retries=args.max_retries
                    )

                    cleaned_stem = clean_filename_stem(sample_file.stem)
                    if data_dir_name:
                        output_path = os.path.join(
                            args.output_dir,
                            data_dir_name,
                            agent_model_dir,
                            f"{cleaned_stem}_results.json",
                        )
                    else:
                        output_path = os.path.join(
                            args.output_dir,
                            agent_model_dir,
                            f"{cleaned_stem}_results.json",
                        )
                    save_results(results, output_path)
                    print(f"Results saved to: {output_path}")

                    inference_time = results.get("inference_time", 0)
                    total_time += inference_time
                    print(f"Inference time: {inference_time:.2f}s")

                print(f"Total processing time for {model}: {total_time:.2f}s")


if __name__ == "__main__":
    main()
