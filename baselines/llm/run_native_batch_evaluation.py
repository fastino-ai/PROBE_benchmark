#!/usr/bin/env python3
"""
Run native batch evaluation using actual batch APIs from OpenAI, Gemini, and Anthropic.
Each provider gets ONE batch job containing ALL requests.
Uses existing agent prompt generation logic with CLI arguments like proactive_prep.sh.
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

from baselines.llm.llm_prompt_template import generate_llm_prompt
from baselines.agentic.models import MODELS
from baselines.agentic.litellm_model import LitellmModel

# Load environment variables from .env file
load_dotenv()

# Suppress verbose HTTP debug logging
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)
logging.getLogger("together").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)


def sanitize_custom_id(custom_id: str) -> str:
    """
    Sanitize custom_id to comply with Anthropic's pattern ^[a-zA-Z0-9_-]{1,64}$.
    Replace dots and other invalid characters with underscores, keep hyphens.
    """
    # Replace dots and other invalid characters with underscores, keep hyphens
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", custom_id)

    # Ensure it's within the 64 character limit
    if len(sanitized) > 64:
        sanitized = sanitized[:64]

    # Ensure it's not empty
    if not sanitized:
        sanitized = "req_0000"

    return sanitized


def map_model_to_provider_and_name(model) -> tuple[str, str]:
    """Map MODELS.* constants to provider and actual model name for batch APIs."""

    # Handle string models
    if isinstance(model, str):
        # Detect provider from model name
        model_lower = model.lower()
        
        # Anthropic models
        if "claude" in model_lower or "anthropic" in model_lower:
            # Remove "anthropic/" prefix if present
            return "anthropic", model.replace("anthropic/", "")
        
        # Google models
        elif "gemini" in model_lower or "google" in model_lower:
            # Remove "gemini/" prefix if present
            return "google", model.replace("gemini/", "")
        
        # Together.ai models
        elif "together" in model_lower or "deepseek" in model_lower:
            # Remove "together_ai/" prefix if present
            return "together", model.replace("together_ai/", "")
        
        # Default to OpenAI for gpt models or unknown strings
        else:
            return "openai", model

    # Handle LitellmModel instances
    elif isinstance(model, LitellmModel):
        model_name = model.model

        # Anthropic models
        if "anthropic/" in model_name:
            return "anthropic", model_name.replace("anthropic/", "")

        # Google models
        elif "gemini/" in model_name:
            return "google", model_name.replace("gemini/", "")

        # Together.ai models
        elif "together_ai/" in model_name:
            return "together", model_name.replace("together_ai/", "")

        else:
            raise ValueError(f"Unknown LitellmModel provider in: {model_name}")

    else:
        raise ValueError(
            f"Unknown model type: {type(model)}. Expected str or LitellmModel."
        )


def load_input_data(data_dir: str, max_files: int = None) -> List[Dict[str, Any]]:
    """Load input data from directory."""
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Get all JSON files
    all_json_files = list(data_path.glob("*.json"))

    if not all_json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    # Apply max_files limit if specified
    if max_files is not None:
        json_files = all_json_files[:max_files]
        print(
            f"Loading {len(json_files)} files (limited from {len(all_json_files)}) from {data_dir}"
        )
    else:
        json_files = all_json_files
        print(f"Loading all {len(json_files)} files from {data_dir}")

    datasets = []
    for json_file in json_files:
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                data["source_file"] = json_file.name
                datasets.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return datasets


def prepare_sample_for_inference(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert sample to the format expected by existing agents."""
    world_model = sample.get("world_model", {})
    data_points = sample.get("data_points", [])

    # Create memory from data_points (matches existing structure)
    memory = data_points

    # Extract persona from world_model (following existing pattern)
    persona = {
        "name": world_model.get("persona_full_name", "Unknown"),
        "occupation": world_model.get("persona_occupation", ""),
        "about": world_model.get("persona_about", ""),
        "relationships": world_model.get("relationships", []),
    }

    return {"memory": memory, "world_model": world_model, "persona": persona}


def generate_baseline_prompt(
    memory: List[Dict[str, Any]], world_model: Dict[str, Any], persona: Dict[str, Any]
) -> str:
    """Generate prompt using LLM prompt template for pure LLM calls."""
    prompt = generate_llm_prompt(
        persona=json.dumps(persona, indent=2),
        world_model=json.dumps(world_model, indent=2),
        data_sources=json.dumps(memory, indent=2),
        available_actions=json.dumps(
            world_model.get("available_actions", []), indent=2
        ),
    )
    return prompt


async def submit_openai_batch(datasets: List[Dict[str, Any]], model: str) -> List[str]:
    """Submit TEN batch jobs to OpenAI (splitting requests into 10 groups to avoid file size limits)."""
    from openai import OpenAI
    import math

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"🔨 Preparing OpenAI batches for {model}...")
    print(
        f"📊 Splitting {len(datasets)} requests into 10 groups to avoid file size limits"
    )

    # Split datasets into 10 groups
    group_size = math.ceil(len(datasets) / 10)
    dataset_groups = []
    for i in range(0, len(datasets), group_size):
        dataset_groups.append(datasets[i : i + group_size])

    print(f"📋 Group sizes: {[len(group) for group in dataset_groups]}")

    batch_ids = []

    # Submit each group as a separate batch
    for group_idx, group_datasets in enumerate(dataset_groups, 1):
        print(
            f"🔨 Preparing OpenAI batch {group_idx}/10 ({len(group_datasets)} requests)..."
        )

        # Create JSONL data for this group
        jsonl_data = []
        for i, sample in enumerate(group_datasets):
            try:
                prepared = prepare_sample_for_inference(sample)
                prompt = generate_baseline_prompt(
                    prepared["memory"], prepared["world_model"], prepared["persona"]
                )

                # Convert input filename to results filename for custom_id
                # Preserve meaningful filenames: "bottleneck_004_02_harish_bhat_output.json" -> "bottleneck_004_02_harish_bhat_results.json"
                source_file = sample["source_file"]
                if source_file.endswith("_input.json"):
                    custom_id = source_file.replace("_input.json", "_results.json")
                elif source_file.endswith("_output.json"):
                    custom_id = source_file.replace("_output.json", "_results.json")
                elif source_file.endswith(".json"):
                    # For files ending with .json, insert "_results" before the extension
                    custom_id = source_file.replace(".json", "_results.json")
                else:
                    # Fallback for unexpected formats
                    custom_id = f"{source_file}_results.json"
                batch_request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                jsonl_data.append(json.dumps(batch_request))

            except Exception as e:
                print(
                    f"❌ Error preparing OpenAI request {i} in group {group_idx}: {e}"
                )
                continue

        # Write JSONL file for this group
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_data))
            jsonl_path = f.name

        try:
            # Upload file to OpenAI
            with open(jsonl_path, "rb") as f:
                file_response = client.files.create(file=f, purpose="batch")

            # Create batch job for this group
            batch = client.batches.create(
                input_file_id=file_response.id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            batch_ids.append(batch.id)
            print(f"✅ OpenAI batch {group_idx}/10 created: {batch.id}")

        except Exception as e:
            print(f"❌ Error creating OpenAI batch {group_idx}/10: {e}")
            continue
        finally:
            # Cleanup temp file
            Path(jsonl_path).unlink(missing_ok=True)

    print(f"✅ Created {len(batch_ids)} OpenAI batches total")
    return batch_ids


async def submit_gemini_batch(
    datasets: List[Dict[str, Any]], model: str = "gemini-2.5-flash"
) -> List[str]:
    """Submit TEN batch jobs to Gemini (splitting requests into 10 groups to avoid size limits)."""
    from google import genai
    import math

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"🔨 Preparing Gemini batches for {model}...")
    print(f"📊 Splitting {len(datasets)} requests into 10 groups to avoid size limits")

    # Split datasets into 10 groups
    group_size = math.ceil(len(datasets) / 10)
    dataset_groups = []
    for i in range(0, len(datasets), group_size):
        dataset_groups.append(datasets[i : i + group_size])

    print(f"📋 Group sizes: {[len(group) for group in dataset_groups]}")

    batch_names = []

    # Submit each group as a separate batch
    for group_idx, group_datasets in enumerate(dataset_groups, 1):
        print(
            f"🔨 Preparing Gemini batch {group_idx}/10 ({len(group_datasets)} requests)..."
        )

        # Prepare inline requests for this group (no generation_config allowed for inline method)
        inline_requests = []
        for i, sample in enumerate(group_datasets):
            try:
                prepared = prepare_sample_for_inference(sample)
                prompt = generate_baseline_prompt(
                    prepared["memory"], prepared["world_model"], prepared["persona"]
                )

                # Simple format for inline requests - NO generation_config
                gemini_request = {
                    "contents": [{"parts": [{"text": prompt}], "role": "user"}]
                }

                inline_requests.append(gemini_request)

            except Exception as e:
                print(
                    f"❌ Error preparing Gemini request {i} in group {group_idx}: {e}"
                )
                continue

        try:
            # Create batch job for this group with inline requests (exactly as docs show)
            batch_job = client.batches.create(
                model=f"models/{model}",
                src=inline_requests,  # Direct list of requests
                config={
                    "display_name": f"batch_evaluation_group_{group_idx}_{int(time.time())}"
                },
            )

            batch_names.append(batch_job.name)  # Store full name for status checking
            batch_id = batch_job.name.split("/")[-1]
            print(f"✅ Gemini batch {group_idx}/10 created: {batch_id}")
            print(f"📋 Full batch name: {batch_job.name}")

        except Exception as e:
            print(f"❌ Error creating Gemini batch {group_idx}/10: {e}")
            continue

    print(f"✅ Created {len(batch_names)} Gemini batches total")
    return batch_names  # Return list of full names for status checking


async def submit_together_batch(
    datasets: List[Dict[str, Any]], model: str = "deepseek-ai/DeepSeek-R1"
) -> List[str]:
    """Submit TEN batch jobs to Together.ai (splitting requests into 10 groups to avoid file size limits)."""
    from together import Together
    import math

    client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

    print(f"🔨 Preparing Together.ai batches for {model}...")
    print(
        f"📊 Splitting {len(datasets)} requests into 10 groups to avoid file size limits"
    )

    # Split datasets into 10 groups
    group_size = math.ceil(len(datasets) / 10)
    dataset_groups = []
    for i in range(0, len(datasets), group_size):
        dataset_groups.append(datasets[i : i + group_size])

    print(f"📋 Group sizes: {[len(group) for group in dataset_groups]}")

    batch_ids = []

    # Submit each group as a separate batch
    for group_idx, group_datasets in enumerate(dataset_groups, 1):
        print(
            f"🔨 Preparing Together.ai batch {group_idx}/10 ({len(group_datasets)} requests)..."
        )

        # Create JSONL data for this group (Together.ai format)
        jsonl_data = []
        for i, sample in enumerate(group_datasets):
            try:
                prepared = prepare_sample_for_inference(sample)
                prompt = generate_baseline_prompt(
                    prepared["memory"], prepared["world_model"], prepared["persona"]
                )

                # Convert input filename to results filename for custom_id
                # Preserve meaningful filenames: "bottleneck_004_02_harish_bhat_output.json" -> "bottleneck_004_02_harish_bhat_results.json"
                source_file = sample["source_file"]
                if source_file.endswith("_input.json"):
                    custom_id = source_file.replace("_input.json", "_results.json")
                elif source_file.endswith("_output.json"):
                    custom_id = source_file.replace("_output.json", "_results.json")
                elif source_file.endswith(".json"):
                    # For files ending with .json, insert "_results" before the extension
                    custom_id = source_file.replace(".json", "_results.json")
                else:
                    # Fallback for unexpected formats
                    custom_id = f"{source_file}_results.json"
                batch_request = {
                    "custom_id": custom_id,
                    "body": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                    },
                }
                jsonl_data.append(json.dumps(batch_request))

            except Exception as e:
                print(
                    f"❌ Error preparing Together.ai request {i} in group {group_idx}: {e}"
                )
                continue

        # Write JSONL file for this group
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(jsonl_data))
            jsonl_path = f.name

        try:
            # Upload file to Together.ai
            with open(jsonl_path, "rb") as f:
                file_response = client.files.upload(
                    file=jsonl_path, purpose="batch-api"
                )

            # Create batch job for this group
            batch = client.batches.create_batch(
                file_id=file_response.id, endpoint="/v1/chat/completions"
            )

            batch_ids.append(batch.id)
            print(f"✅ Together.ai batch {group_idx}/10 created: {batch.id}")

        except Exception as e:
            print(f"❌ Error creating Together.ai batch {group_idx}/10: {e}")
            continue
        finally:
            # Cleanup temp file
            Path(jsonl_path).unlink(missing_ok=True)

    print(f"✅ Created {len(batch_ids)} Together.ai batches total")
    return batch_ids


async def submit_anthropic_batch(
    datasets: List[Dict[str, Any]], model: str
) -> List[str]:
    """Submit TEN batch jobs to Anthropic (splitting requests into 10 groups to avoid size limits)."""
    import anthropic
    import math

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"🔨 Preparing Anthropic batches for {model}...")
    print(f"📊 Splitting {len(datasets)} requests into 10 groups to avoid size limits")

    # Split datasets into 10 groups
    group_size = math.ceil(len(datasets) / 10)
    dataset_groups = []
    for i in range(0, len(datasets), group_size):
        dataset_groups.append(datasets[i : i + group_size])

    print(f"📋 Group sizes: {[len(group) for group in dataset_groups]}")

    batch_ids = []

    # Submit each group as a separate batch
    for group_idx, group_datasets in enumerate(dataset_groups, 1):
        print(
            f"🔨 Preparing Anthropic batch {group_idx}/10 ({len(group_datasets)} requests)..."
        )

        # Prepare requests for this group
        anthropic_requests = []

        for i, sample in enumerate(group_datasets):
            try:
                prepared = prepare_sample_for_inference(sample)
                prompt = generate_baseline_prompt(
                    prepared["memory"], prepared["world_model"], prepared["persona"]
                )

                # Convert input filename to results filename
                # For Anthropic: generate custom_id WITHOUT .json extension (batch_post_processor will add it back)
                # "bottleneck_004_02_harish_bhat_output.json" -> "bottleneck_004_02_harish_bhat_results"
                source_file = sample["source_file"]
                if source_file.endswith("_input.json"):
                    custom_id = source_file.replace("_input.json", "_results")
                elif source_file.endswith("_output.json"):
                    custom_id = source_file.replace("_output.json", "_results")
                elif source_file.endswith(".json"):
                    # For files ending with .json, insert "_results" before the extension and remove extension
                    custom_id = source_file.replace(".json", "_results")
                else:
                    # Fallback for unexpected formats
                    custom_id = f"{source_file}_results"

                # Sanitize custom_id for Anthropic API compliance (dots to underscores, keep hyphens)
                sanitized_custom_id = sanitize_custom_id(custom_id)

                anthropic_request = {
                    "custom_id": sanitized_custom_id,
                    "params": {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 8192,  # High limit for Anthropic batch requests
                    },
                }

                anthropic_requests.append(anthropic_request)

            except Exception as e:
                print(
                    f"❌ Error preparing Anthropic request {i} in group {group_idx}: {e}"
                )
                continue

        # Create message batch for this group
        try:
            batch = client.messages.batches.create(requests=anthropic_requests)
            batch_ids.append(batch.id)
            print(f"✅ Anthropic batch {group_idx}/10 created: {batch.id}")
        except Exception as e:
            print(f"❌ Error creating Anthropic batch {group_idx}/10: {e}")
            continue

    print(f"✅ Created {len(batch_ids)} Anthropic batches total")
    return batch_ids


async def wait_for_openai_batch(
    batch_ids: List[str], model: str, data_dir: str, timeout: int = 86400
):
    """Wait for multiple OpenAI batches to complete and return combined results."""
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    print(f"⏳ Waiting for {len(batch_ids)} OpenAI batches to complete...")

    # Set up results tracking
    completed_batches = {}
    failed_batches = set()

    # Prepare combined results file
    data_dir_name = Path(data_dir).name
    results_dir = f"results/{data_dir_name}/batch_openai_{model}"
    os.makedirs(results_dir, exist_ok=True)
    combined_output_file = f"{results_dir}/openai_combined_batches.jsonl"

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check status of all pending batches
        pending_batch_ids = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]

        if not pending_batch_ids:
            print("✅ All OpenAI batches completed!")
            break

        for batch_id in pending_batch_ids:
            try:
                batch_status = client.batches.retrieve(batch_id)

                if batch_status.status == "completed":
                    print(f"✅ OpenAI batch {batch_id} completed!")

                    # Check if output file exists
                    if not batch_status.output_file_id:
                        print(
                            f"❌ OpenAI batch {batch_id} completed but no output file available"
                        )
                        failed_batches.add(batch_id)
                    else:
                        completed_batches[batch_id] = batch_status

                elif batch_status.status in ["failed", "expired", "cancelled"]:
                    print(f"❌ OpenAI batch {batch_id} failed: {batch_status.status}")
                    failed_batches.add(batch_id)

                else:
                    # Show progress for in-progress batches
                    if (
                        hasattr(batch_status, "request_counts")
                        and batch_status.request_counts
                    ):
                        counts = batch_status.request_counts
                        total = counts.total if hasattr(counts, "total") else 0
                        completed = (
                            counts.completed if hasattr(counts, "completed") else 0
                        )
                        failed = counts.failed if hasattr(counts, "failed") else 0

                        if total > 0:
                            progress_pct = (completed + failed) / total * 100
                            print(
                                f"⏳ OpenAI batch {batch_id}: {batch_status.status} - {completed}/{total} completed ({progress_pct:.1f}%), {failed} failed"
                            )

            except Exception as e:
                print(f"⚠️  Error checking OpenAI batch {batch_id}: {e}")

        # Only sleep if we have pending batches
        if pending_batch_ids:
            await asyncio.sleep(30)

    # Handle timeout case
    if time.time() - start_time >= timeout:
        remaining_batches = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]
        if remaining_batches:
            print(
                f"⏰ {len(remaining_batches)} OpenAI batches timed out after {timeout//3600} hours"
            )
            return []

    # Combine results from all completed batches
    if completed_batches:
        print(
            f"📝 Combining results from {len(completed_batches)} completed OpenAI batches..."
        )

        with open(combined_output_file, "w") as f:
            total_results = 0
            for batch_id, batch_status in completed_batches.items():
                print(f"📥 Downloading results from batch {batch_id}...")
                try:
                    result_file = client.files.content(batch_status.output_file_id)
                    for line in result_file.text.strip().split("\n"):
                        if line.strip():  # Skip empty lines
                            # Add batch_id to each result for tracking
                            result_data = json.loads(line)
                            result_data["batch_id"] = batch_id
                            f.write(json.dumps(result_data) + "\n")
                            total_results += 1
                except Exception as e:
                    print(f"❌ Error downloading results from batch {batch_id}: {e}")

        print(f"💾 Combined OpenAI batch results saved to: {combined_output_file}")
        print(
            f"📊 Total results: {total_results} from {len(completed_batches)} batches"
        )

        # Return combined results info
        return {
            "results_file": combined_output_file,
            "batch_ids": list(completed_batches.keys()),
            "completed_count": len(completed_batches),
            "failed_count": len(failed_batches),
            "total_results": total_results,
        }

    else:
        print("❌ No OpenAI batches completed successfully")
        return []


async def wait_for_gemini_batch(
    batch_names: List[str], model: str, data_dir: str, timeout: int = 86400
):
    """Wait for multiple Gemini batches to complete and return combined results."""
    from google import genai

    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    print(f"⏳ Waiting for {len(batch_names)} Gemini batches to complete...")

    # Set up results tracking
    completed_batches = {}
    failed_batches = set()

    # Prepare combined results file
    data_dir_name = Path(data_dir).name
    results_dir = f"results/{data_dir_name}/batch_gemini_{model}"
    os.makedirs(results_dir, exist_ok=True)
    combined_output_file = f"{results_dir}/gemini_combined_batches.jsonl"

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check status of all pending batches
        pending_batch_names = [
            name
            for name in batch_names
            if name not in completed_batches and name not in failed_batches
        ]

        if not pending_batch_names:
            print("✅ All Gemini batches completed!")
            break

        for batch_name in pending_batch_names:
            try:
                batch_job = client.batches.get(name=batch_name)

                if batch_job.done and batch_job.metadata.state == "JOB_STATE_SUCCEEDED":
                    print(f"✅ Gemini batch {batch_name.split('/')[-1]} completed!")
                    completed_batches[batch_name] = batch_job

                elif batch_job.done and batch_job.metadata.state in [
                    "JOB_STATE_FAILED",
                    "JOB_STATE_CANCELLED",
                ]:
                    print(
                        f"❌ Gemini batch {batch_name.split('/')[-1]} failed: {batch_job.metadata.state}"
                    )
                    failed_batches.add(batch_name)

                else:
                    # Show progress for in-progress batches
                    batch_id = batch_name.split("/")[-1]
                    if hasattr(batch_job.metadata, "progress_percent"):
                        print(
                            f"⏳ Gemini batch {batch_id}: {batch_job.metadata.state} ({batch_job.metadata.progress_percent}%)"
                        )
                    else:
                        print(f"⏳ Gemini batch {batch_id}: {batch_job.metadata.state}")

            except Exception as e:
                print(
                    f"⚠️  Error checking Gemini batch {batch_name.split('/')[-1]}: {e}"
                )

        # Only sleep if we have pending batches
        if pending_batch_names:
            await asyncio.sleep(30)

    # Handle timeout case
    if time.time() - start_time >= timeout:
        remaining_batches = [
            name
            for name in batch_names
            if name not in completed_batches and name not in failed_batches
        ]
        if remaining_batches:
            print(
                f"⏰ {len(remaining_batches)} Gemini batches timed out after {timeout//3600} hours"
            )
            return []

    # Combine results from all completed batches
    if completed_batches:
        print(
            f"📝 Combining results from {len(completed_batches)} completed Gemini batches..."
        )

        with open(combined_output_file, "w") as f:
            total_results = 0
            for batch_name, batch_job in completed_batches.items():
                batch_id = batch_name.split("/")[-1]
                print(f"📥 Processing results from batch {batch_id}...")
                try:
                    # Handle inline responses from Gemini Batch API (using correct attribute as per docs)
                    if hasattr(batch_job, "dest") and hasattr(
                        batch_job.dest, "inlined_responses"
                    ):
                        for i, inline_response in enumerate(
                            batch_job.dest.inlined_responses
                        ):
                            if (
                                hasattr(inline_response, "response")
                                and inline_response.response
                            ):
                                # Extract text from Gemini response format
                                try:
                                    text_content = ""
                                    if hasattr(inline_response.response, "candidates"):
                                        for (
                                            candidate
                                        ) in inline_response.response.candidates:
                                            if hasattr(
                                                candidate, "content"
                                            ) and hasattr(candidate.content, "parts"):
                                                for part in candidate.content.parts:
                                                    if hasattr(part, "text"):
                                                        text_content += part.text
                                    elif hasattr(inline_response.response, "text"):
                                        # Fallback for direct text attribute
                                        text_content = inline_response.response.text

                                    result_data = {
                                        "custom_id": f"req_{total_results:04d}",  # Sequential numbering across all batches
                                        "response": {"content": text_content},
                                        "error": None,
                                        "batch_id": batch_id,  # Track which batch it came from
                                    }
                                    f.write(json.dumps(result_data) + "\n")
                                    total_results += 1
                                except Exception as e:
                                    result_data = {
                                        "custom_id": f"req_{total_results:04d}",
                                        "response": None,
                                        "error": f"Failed to parse response: {e}",
                                        "batch_id": batch_id,
                                    }
                                    f.write(json.dumps(result_data) + "\n")
                                    total_results += 1

                            elif hasattr(inline_response, "error"):
                                result_data = {
                                    "custom_id": f"req_{total_results:04d}",
                                    "response": None,
                                    "error": str(inline_response.error),
                                    "batch_id": batch_id,
                                }
                                f.write(json.dumps(result_data) + "\n")
                                total_results += 1

                    # Fallback: check if response structure is different
                    elif hasattr(batch_job, "response") and hasattr(
                        batch_job.response, "inlined_responses"
                    ):
                        print(
                            f"⚠️  Using fallback response structure for Gemini batch {batch_id}"
                        )
                        for i, inline_response in enumerate(
                            batch_job.response.inlined_responses
                        ):
                            if (
                                hasattr(inline_response, "response")
                                and inline_response.response
                            ):
                                try:
                                    text_content = ""
                                    if hasattr(inline_response.response, "candidates"):
                                        for (
                                            candidate
                                        ) in inline_response.response.candidates:
                                            if hasattr(
                                                candidate, "content"
                                            ) and hasattr(candidate.content, "parts"):
                                                for part in candidate.content.parts:
                                                    if hasattr(part, "text"):
                                                        text_content += part.text
                                    elif hasattr(inline_response.response, "text"):
                                        text_content = inline_response.response.text

                                    result_data = {
                                        "custom_id": f"req_{total_results:04d}",
                                        "response": {"content": text_content},
                                        "error": None,
                                        "batch_id": batch_id,
                                    }
                                    f.write(json.dumps(result_data) + "\n")
                                    total_results += 1
                                except Exception as e:
                                    result_data = {
                                        "custom_id": f"req_{total_results:04d}",
                                        "response": None,
                                        "error": f"Failed to parse response: {e}",
                                        "batch_id": batch_id,
                                    }
                                    f.write(json.dumps(result_data) + "\n")
                                    total_results += 1

                            elif hasattr(inline_response, "error"):
                                result_data = {
                                    "custom_id": f"req_{total_results:04d}",
                                    "response": None,
                                    "error": str(inline_response.error),
                                    "batch_id": batch_id,
                                }
                                f.write(json.dumps(result_data) + "\n")
                                total_results += 1

                    else:
                        print(
                            f"❌ Could not find inlined_responses in batch {batch_id}"
                        )
                        # Write a single error entry
                        result_data = {
                            "custom_id": f"req_{total_results:04d}",
                            "response": None,
                            "error": "No inlined_responses found in batch job response structure",
                            "batch_id": batch_id,
                        }
                        f.write(json.dumps(result_data) + "\n")
                        total_results += 1

                except Exception as e:
                    print(f"❌ Error processing results from batch {batch_id}: {e}")

        print(f"💾 Combined Gemini batch results saved to: {combined_output_file}")
        print(
            f"📊 Total results: {total_results} from {len(completed_batches)} batches"
        )

        # Return combined results info
        return {
            "results_file": combined_output_file,
            "batch_ids": [name.split("/")[-1] for name in completed_batches.keys()],
            "completed_count": len(completed_batches),
            "failed_count": len(failed_batches),
            "total_results": total_results,
        }

    else:
        print("❌ No Gemini batches completed successfully")
        return []


async def wait_for_together_batch(
    batch_ids: List[str], model: str, data_dir: str, timeout: int = 86400
):
    """Wait for multiple Together.ai batches to complete and return combined results."""
    from together import Together

    client = Together(api_key=os.getenv("TOGETHERAI_API_KEY"))

    print(f"⏳ Waiting for {len(batch_ids)} Together.ai batches to complete...")

    # Set up results tracking
    completed_batches = {}
    failed_batches = set()

    # Prepare combined results file
    data_dir_name = Path(data_dir).name
    results_dir = f"results/{data_dir_name}/batch_together_{model.replace('/', '_')}"
    os.makedirs(results_dir, exist_ok=True)
    combined_output_file = f"{results_dir}/together_combined_batches.jsonl"

    start_time = time.time()
    while time.time() - start_time < timeout:
        # Check status of all pending batches
        pending_batch_ids = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]

        if not pending_batch_ids:
            print("✅ All Together.ai batches completed!")
            break

        for batch_id in pending_batch_ids:
            try:
                batch_status = client.batches.get_batch(batch_id)

                if batch_status.status == "COMPLETED":
                    print(f"✅ Together.ai batch {batch_id} completed!")
                    completed_batches[batch_id] = batch_status

                elif batch_status.status in ["FAILED", "EXPIRED", "CANCELLED"]:
                    print(
                        f"❌ Together.ai batch {batch_id} failed: {batch_status.status}"
                    )
                    failed_batches.add(batch_id)

                else:
                    # Show progress for in-progress batches
                    if (
                        hasattr(batch_status, "request_counts")
                        and batch_status.request_counts
                    ):
                        counts = batch_status.request_counts
                        if hasattr(counts, "completed") and hasattr(counts, "total"):
                            print(
                                f"⏳ Together.ai batch {batch_id}: {batch_status.status} - {counts.completed}/{counts.total} completed"
                            )

            except Exception as e:
                print(f"⚠️  Error checking Together.ai batch {batch_id}: {e}")

        # Only sleep if we have pending batches
        if pending_batch_ids:
            await asyncio.sleep(30)

    # Handle timeout case
    if time.time() - start_time >= timeout:
        remaining_batches = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]
        if remaining_batches:
            print(
                f"⏰ {len(remaining_batches)} Together.ai batches timed out after {timeout//3600} hours"
            )
            return []

    # Combine results from all completed batches
    if completed_batches:
        print(
            f"📝 Combining results from {len(completed_batches)} completed Together.ai batches..."
        )

        with open(combined_output_file, "w") as f:
            total_results = 0
            for batch_id, batch_status in completed_batches.items():
                print(f"📥 Downloading results from batch {batch_id}...")
                try:
                    # Download to temporary file first, then read and combine
                    temp_output_file = f"{results_dir}/temp_{batch_id}.jsonl"
                    client.files.retrieve_content(
                        id=batch_status.output_file_id, output=temp_output_file
                    )

                    # Read and add to combined file
                    with open(temp_output_file, "r") as temp_f:
                        for line in temp_f:
                            if line.strip():  # Skip empty lines
                                # Add batch_id to each result for tracking
                                result_data = json.loads(line)
                                result_data["batch_id"] = batch_id
                                f.write(json.dumps(result_data) + "\n")
                                total_results += 1

                    # Clean up temp file
                    Path(temp_output_file).unlink(missing_ok=True)

                except Exception as e:
                    print(f"❌ Error downloading results from batch {batch_id}: {e}")

        print(f"💾 Combined Together.ai batch results saved to: {combined_output_file}")
        print(
            f"📊 Total results: {total_results} from {len(completed_batches)} batches"
        )

        # Return combined results info
        return {
            "results_file": combined_output_file,
            "batch_ids": list(completed_batches.keys()),
            "completed_count": len(completed_batches),
            "failed_count": len(failed_batches),
            "total_results": total_results,
        }

    else:
        print("❌ No Together.ai batches completed successfully")
        return []


async def wait_for_anthropic_batch(
    batch_ids: List[str], model: str, data_dir: str, timeout: int = 86400
):
    """Wait for multiple Anthropic batches to complete and return combined results."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    print(f"⏳ Waiting for {len(batch_ids)} Anthropic batches to complete...")

    # Set up results tracking
    completed_batches = {}
    failed_batches = set()
    start_time = time.time()

    # Prepare combined results file
    data_dir_name = Path(data_dir).name
    results_dir = f"results/{data_dir_name}/batch_anthropic_{model}"
    os.makedirs(results_dir, exist_ok=True)
    combined_output_file = f"{results_dir}/anthropic_combined_batches.jsonl"

    while time.time() - start_time < timeout:
        # Check status of all pending batches
        pending_batch_ids = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]

        if not pending_batch_ids:
            print("✅ All Anthropic batches completed!")
            break

        for batch_id in pending_batch_ids:
            try:
                batch_status = client.messages.batches.retrieve(batch_id)

                if batch_status.processing_status == "ended":
                    print(f"✅ Anthropic batch {batch_id} completed!")
                    completed_batches[batch_id] = batch_status

                elif batch_status.processing_status in ["canceled", "errored"]:
                    print(
                        f"❌ Anthropic batch {batch_id} failed: {batch_status.processing_status}"
                    )
                    failed_batches.add(batch_id)

                else:
                    # Show progress for in-progress batches
                    if (
                        hasattr(batch_status, "request_counts")
                        and batch_status.request_counts
                    ):
                        counts = batch_status.request_counts
                        if (
                            hasattr(counts, "processing")
                            and hasattr(counts, "succeeded")
                            and hasattr(counts, "errored")
                        ):
                            completed = (
                                counts.succeeded + counts.errored
                                if hasattr(counts, "succeeded")
                                else 0
                            )
                            total = (
                                counts.processing + completed
                                if hasattr(counts, "processing")
                                else completed
                            )
                            if total > 0:
                                print(
                                    f"⏳ Anthropic batch {batch_id}: {batch_status.processing_status} - {completed}/{total} completed"
                                )

            except Exception as e:
                print(f"⚠️  Error checking Anthropic batch {batch_id}: {e}")

        # Only sleep if we have pending batches
        if pending_batch_ids:
            await asyncio.sleep(30)

    # Handle timeout case
    if time.time() - start_time >= timeout:
        remaining_batches = [
            bid
            for bid in batch_ids
            if bid not in completed_batches and bid not in failed_batches
        ]
        if remaining_batches:
            print(
                f"⏰ {len(remaining_batches)} Anthropic batches timed out after {timeout//3600} hours"
            )
            return []

    # Combine results from all completed batches
    if completed_batches:
        print(
            f"📝 Combining results from {len(completed_batches)} completed Anthropic batches..."
        )

        with open(combined_output_file, "w") as f:
            total_results = 0
            for batch_id, batch_status in completed_batches.items():
                print(f"📥 Downloading results from batch {batch_id}...")
                try:
                    for result in client.messages.batches.results(batch_id):
                        # Use the sanitized custom_id directly (dots replaced with underscores)
                        result_data = {
                            "custom_id": result.custom_id,
                            "response": (
                                result.result.message.dict()
                                if result.result.type == "succeeded"
                                else None
                            ),
                            "error": (
                                str(result.result.error)
                                if result.result.type == "errored"
                                else None
                            ),
                            "batch_id": batch_id,  # Include batch_id for tracking
                        }
                        f.write(json.dumps(result_data) + "\n")
                        total_results += 1
                except Exception as e:
                    print(f"❌ Error downloading results from batch {batch_id}: {e}")

        print(f"💾 Combined Anthropic batch results saved to: {combined_output_file}")
        print(
            f"📊 Total results: {total_results} from {len(completed_batches)} batches"
        )

        # Return combined results info
        return {
            "results_file": combined_output_file,
            "batch_ids": list(completed_batches.keys()),
            "completed_count": len(completed_batches),
            "failed_count": len(failed_batches),
            "total_results": total_results,
        }

    else:
        print("❌ No Anthropic batches completed successfully")
        return []


async def run_native_batch_evaluation(
    data_dir: str = "data/sept_23_1000_inputs_20250923_131956", max_files: int = None
):
    """Run native batch evaluation with ONE batch job per provider."""
    print("🚀 Starting Native Batch Evaluation")
    print("=" * 60)
    print("Each provider gets ONE batch job containing ALL requests")

    # Check API keys
    required_keys = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "TOGETHERAI_API_KEY": "Together.ai",
    }

    # Check for Gemini API key (multiple possible names)
    gemini_key_found = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")

    available_providers = []
    for key, provider in required_keys.items():
        if os.getenv(key):
            available_providers.append(provider)
            print(f"✅ {provider} API key found")
        else:
            print(f"❌ {provider} API key missing ({key})")

    # Special handling for Gemini
    if gemini_key_found:
        available_providers.append("Gemini")
        key_name = "GOOGLE_API_KEY" if os.getenv("GOOGLE_API_KEY") else "GEMINI_API_KEY"
        print(f"✅ Gemini API key found ({key_name})")
    else:
        print("❌ Gemini API key missing (GOOGLE_API_KEY or GEMINI_API_KEY)")

    if not available_providers:
        print("🛑 No API keys found. Please set environment variables.")
        return

    # Load data
    try:
        datasets = load_input_data(data_dir, max_files=max_files)
        print(f"📁 Loaded {len(datasets)} datasets from {data_dir}")
    except FileNotFoundError as e:
        print(f"🛑 {e}")
        return

    # Submit batch jobs to all providers
    batch_jobs = {}
    batch_timings = {}  # Track start times for latency calculation

    if "OpenAI" in available_providers:
        try:
            start_time = time.time()
            provider, model_name = map_model_to_provider_and_name(MODELS.OPENAI.GPT_5)
            batch_ids = await submit_openai_batch(datasets, model_name)
            batch_jobs["openai"] = batch_ids
            batch_timings["openai"] = {"start_time": start_time, "model": model_name}
        except Exception as e:
            print(f"❌ Failed to submit OpenAI batch: {e}")

    if "Gemini" in available_providers:
        try:
            start_time = time.time()
            provider, model_name = map_model_to_provider_and_name(
                MODELS.GOOGLE.GEMINI_25_FLASH
            )
            batch_ids = await submit_gemini_batch(datasets, model_name)
            batch_jobs["gemini"] = batch_ids
            batch_timings["gemini"] = {"start_time": start_time, "model": model_name}
        except Exception as e:
            print(f"❌ Failed to submit Gemini batch: {e}")

    if "Anthropic" in available_providers:
        try:
            start_time = time.time()
            provider, model_name = map_model_to_provider_and_name(
                MODELS.ANTHROPIC.CLAUDE_4_SONNET
            )
            batch_ids = await submit_anthropic_batch(datasets, model_name)
            batch_jobs["anthropic"] = batch_ids
            batch_timings["anthropic"] = {"start_time": start_time, "model": model_name}
        except Exception as e:
            print(f"❌ Failed to submit Anthropic batch: {e}")

    if "Together.ai" in available_providers:
        try:
            start_time = time.time()
            provider, model_name = map_model_to_provider_and_name(
                MODELS.TOGETHER_AI.DEEPSEEK_R1
            )
            batch_ids = await submit_together_batch(datasets, model_name)
            batch_jobs["together"] = batch_ids
            batch_timings["together"] = {"start_time": start_time, "model": model_name}
        except Exception as e:
            print(f"❌ Failed to submit Together.ai batch: {e}")

    print(f"\n📊 Submitted {len(batch_jobs)} batch jobs")
    print(f"💰 Each batch processes {len(datasets)} requests at 50% cost savings")

    if not batch_jobs:
        print("🛑 No batch jobs submitted successfully")
        return

    print(
        "\n⏳ Waiting for batch completion... (10-30 minutes typically, will wait up to 24 hours)"
    )

    # Wait for all batches to complete
    all_results = {}

    # Use asyncio.gather to wait for all batches concurrently
    tasks = []

    if "openai" in batch_jobs:
        tasks.append(
            ("openai", wait_for_openai_batch(batch_jobs["openai"], "gpt-4o", data_dir))
        )
    if "gemini" in batch_jobs:
        tasks.append(
            (
                "gemini",
                wait_for_gemini_batch(
                    batch_jobs["gemini"], "gemini-2.5-flash", data_dir
                ),
            )
        )
    if "anthropic" in batch_jobs:
        tasks.append(
            (
                "anthropic",
                wait_for_anthropic_batch(
                    batch_jobs["anthropic"], "claude-4-sonnet-20250219", data_dir
                ),
            )
        )
    if "together" in batch_jobs:
        tasks.append(
            (
                "together",
                wait_for_together_batch(
                    batch_jobs["together"], "deepseek-ai/DeepSeek-R1", data_dir
                ),
            )
        )

    # Wait for all providers concurrently
    for provider, task in tasks:
        try:
            result_info = await task
            all_results[provider] = result_info

            # Calculate and print latency for this provider
            if provider in batch_timings and isinstance(result_info, dict):
                end_time = time.time()
                total_latency = end_time - batch_timings[provider]["start_time"]
                model_name = batch_timings[provider]["model"]

                if "error" in result_info:
                    print(
                        f"❌ {provider.upper()}: {result_info['error']} (batch_id: {result_info.get('batch_id', 'unknown')})"
                    )
                    print(
                        f"⏱️  {model_name} TOTAL LATENCY: {total_latency:.1f}s (trigger to error)"
                    )
                else:
                    print(
                        f"🎯 {provider.upper()}: Results saved to {result_info['results_file']}"
                    )
                    print(
                        f"⏱️  {model_name} TOTAL LATENCY: {total_latency:.1f}s (trigger to complete response)"
                    )
            elif isinstance(result_info, dict):
                if "error" in result_info:
                    print(
                        f"❌ {provider.upper()}: {result_info['error']} (batch_id: {result_info.get('batch_id', 'unknown')})"
                    )
                else:
                    print(
                        f"🎯 {provider.upper()}: Results saved to {result_info['results_file']}"
                    )
            else:
                print(f"❌ {provider.upper()}: No results received")

        except Exception as e:
            print(f"❌ Error getting {provider} results: {e}")
            all_results[provider] = None

    # Display results summary
    print("\n📊 BATCH RESULTS SUMMARY")
    print("=" * 40)

    successful_providers = 0
    failed_providers = 0

    for provider, result_info in all_results.items():
        if isinstance(result_info, dict) and "results_file" in result_info:
            successful_providers += 1
            timing_info = batch_timings[provider]
            model_name = timing_info["model"]
            print(
                f"✅ {provider.upper()} ({model_name}): {result_info['results_file']}"
            )
        else:
            failed_providers += 1
            print(f"❌ {provider.upper()}: Failed to complete")

    print(
        f"\n🎯 SUMMARY: {successful_providers}/{successful_providers + failed_providers} providers completed successfully"
    )

    # Display latency summary
    print("\n⏱️  LATENCY SUMMARY")
    print("=" * 40)
    for provider in batch_timings:
        if provider in all_results and isinstance(all_results[provider], dict):
            end_time = time.time()
            total_latency = end_time - batch_timings[provider]["start_time"]
            model_name = batch_timings[provider]["model"]
            print(f"📊 {model_name}: {total_latency:.1f}s (submission to completion)")

    print("\n✨ Native batch evaluation complete!")
    print(
        "🎯 Used actual batch APIs: OpenAI Files+Batch, Gemini Batch, Anthropic Message Batches, Together.ai Batch"
    )
    print(f"💰 Processed {len(datasets)} examples with 50% cost savings per provider")
    print("📁 All results saved as JSONL files in batch_results/ directory")


async def run_dynamic_batch_evaluation(
    models: List[str], data_dir: str, max_files: int = None
):
    """Run dynamic batch evaluation with specified models and data directory."""
    print("🚀 Starting Dynamic Native Batch Evaluation")
    print("=" * 60)
    print("Each provider gets ONE batch job containing ALL requests")

    # Parse and group models by provider
    provider_models = {}
    required_api_keys = set()

    for model_str in models:
        try:
            # Parse model string to get actual model object
            if model_str.startswith("MODELS."):
                # Handle format like "MODELS.OPENAI.GPT_5"
                model_obj = eval(model_str)  # Convert "MODELS.OPENAI.GPT_5" to actual object
            else:
                # Handle raw model name strings like "gpt-5", "claude-opus-4.1"
                model_obj = model_str

            provider, native_model = map_model_to_provider_and_name(model_obj)

            if provider not in provider_models:
                provider_models[provider] = []

            provider_models[provider].append(
                {"original": model_str, "model_obj": model_obj, "native": native_model}
            )

            # Track required API keys
            if provider == "openai":
                required_api_keys.add("OPENAI_API_KEY")
            elif provider == "google":
                required_api_keys.add(
                    "GOOGLE_API_KEY"
                )  # Will check both GOOGLE_API_KEY and GEMINI_API_KEY later
            elif provider == "anthropic":
                required_api_keys.add("ANTHROPIC_API_KEY")
            elif provider == "together":
                required_api_keys.add("TOGETHERAI_API_KEY")

        except Exception as e:
            print(f"❌ Error parsing model {model_str}: {e}")
            return

    print("📋 Models grouped by provider:")
    for provider, model_list in provider_models.items():
        print(f"   🔸 {provider.upper()}: {len(model_list)} models")
        for model_info in model_list:
            print(f"      - {model_info['original']} → {model_info['native']}")

    # Check required API keys
    missing_keys = []
    for key in required_api_keys:
        if key == "GOOGLE_API_KEY":
            # Special handling for Gemini - check both possible key names
            if not (os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")):
                missing_keys.append("GOOGLE_API_KEY or GEMINI_API_KEY")
        elif not os.getenv(key):
            missing_keys.append(key)

    if missing_keys:
        print(f"❌ Missing required API keys: {', '.join(missing_keys)}")
        return

    print(f"✅ All required API keys found")

    # Load data
    try:
        datasets = load_input_data(data_dir, max_files)
        print(f"📁 Loaded {len(datasets)} datasets from {data_dir}")
    except FileNotFoundError as e:
        print(f"🛑 {e}")
        return

    # Submit batch jobs for each model
    batch_jobs = {}
    batch_timings = {}

    for provider, model_list in provider_models.items():
        for model_info in model_list:
            model_key = f"{provider}_{model_info['original'].replace('.', '_').replace('/', '_')}"

            try:
                start_time = time.time()

                if provider == "openai":
                    batch_ids = await submit_openai_batch(
                        datasets, model_info["native"]
                    )
                    batch_id = batch_ids  # Keep as list for OpenAI
                elif provider == "google":
                    batch_ids = await submit_gemini_batch(
                        datasets, model_info["native"]
                    )
                    batch_id = batch_ids  # Keep as list for Gemini
                elif provider == "anthropic":
                    batch_ids = await submit_anthropic_batch(
                        datasets, model_info["native"]
                    )
                    batch_id = batch_ids  # Keep as list for Anthropic
                elif provider == "together":
                    batch_ids = await submit_together_batch(
                        datasets, model_info["native"]
                    )
                    batch_id = batch_ids  # Keep as list for Together.ai
                else:
                    print(f"⚠️ Unsupported provider: {provider}")
                    continue

                batch_jobs[model_key] = batch_id
                batch_timings[model_key] = {
                    "start_time": start_time,
                    "model": model_info["native"],
                    "original_model": model_info["original"],
                    "provider": provider,
                }

                print(f"✅ {model_info['original']} batch submitted: {batch_id}")

            except Exception as e:
                print(f"❌ Failed to submit {model_info['original']} batch: {e}")

    total_requests = len(datasets) * len(batch_jobs)
    print(f"\n📊 Submitted {len(batch_jobs)} batch jobs")
    print(
        f"📋 Total requests: {total_requests} ({len(datasets)} samples × {len(batch_jobs)} models)"
    )

    if not batch_jobs:
        print("🛑 No batch jobs submitted successfully")
        return

    print(
        "\n⏳ Waiting for batch completion... (10-30 minutes typically, will wait up to 24 hours)"
    )

    # Create wait tasks for each batch
    tasks = []

    for model_key, batch_id in batch_jobs.items():
        provider = batch_timings[model_key]["provider"]

        if provider == "openai":
            # batch_id is already a list for OpenAI
            task = wait_for_openai_batch(
                batch_id, batch_timings[model_key]["model"], data_dir
            )
        elif provider == "google":
            # batch_id is already a list for Gemini
            task = wait_for_gemini_batch(
                batch_id, batch_timings[model_key]["model"], data_dir
            )
        elif provider == "anthropic":
            # batch_id is already a list for Anthropic
            task = wait_for_anthropic_batch(
                batch_id, batch_timings[model_key]["model"], data_dir
            )
        elif provider == "together":
            # batch_id is already a list for Together.ai
            task = wait_for_together_batch(
                batch_id, batch_timings[model_key]["model"], data_dir
            )
        else:
            continue

        tasks.append((model_key, task))

    # Wait for all batches concurrently
    all_results = {}

    for model_key, task in tasks:
        try:
            result_info = await task
            all_results[model_key] = result_info

            # Calculate and print latency
            end_time = time.time()
            timing_info = batch_timings[model_key]
            total_latency = end_time - timing_info["start_time"]

            if isinstance(result_info, dict) and "error" in result_info:
                print(
                    f"❌ {timing_info['original_model']}: {result_info['error']} (batch_id: {result_info.get('batch_id', 'unknown')})"
                )
                print(
                    f"⏱️  {timing_info['model']} TOTAL LATENCY: {total_latency:.1f}s (trigger to error)"
                )
            elif isinstance(result_info, dict) and "results_file" in result_info:
                print(
                    f"🎯 {timing_info['original_model']}: Results saved to {result_info['results_file']}"
                )
                print(
                    f"⏱️  {timing_info['model']} TOTAL LATENCY: {total_latency:.1f}s (trigger to complete response)"
                )
            else:
                print(f"❌ {timing_info['original_model']}: No results received")

        except Exception as e:
            print(f"❌ Error getting {model_key} results: {e}")
            all_results[model_key] = None

    # Display results summary
    print("\n📊 BATCH RESULTS SUMMARY")
    print("=" * 50)

    successful_models = 0
    failed_models = 0

    for model_key, result_info in all_results.items():
        timing_info = batch_timings[model_key]

        if isinstance(result_info, dict) and "results_file" in result_info:
            successful_models += 1
            print(
                f"✅ {timing_info['original_model']} ({timing_info['model']}): {result_info['results_file']}"
            )
        else:
            failed_models += 1
            print(f"❌ {timing_info['original_model']}: Failed to complete")

    print(
        f"\n🎯 SUMMARY: {successful_models}/{successful_models + failed_models} models completed successfully"
    )

    # Display latency summary
    print("\n⏱️  LATENCY SUMMARY")
    print("=" * 50)
    for model_key in batch_timings:
        if model_key in all_results and isinstance(all_results[model_key], dict):
            timing_info = batch_timings[model_key]
            end_time = time.time()
            total_latency = end_time - timing_info["start_time"]
            print(
                f"📊 {timing_info['original_model']} ({timing_info['model']}): {total_latency:.1f}s"
            )

    print(f"\n✨ Dynamic batch evaluation complete!")
    print(
        f"🎯 Processed {len(datasets)} samples across {len(batch_jobs)} models using native batch APIs"
    )
    print(f"💰 50% cost savings achieved across all providers")
    print(f"📁 All results saved as JSONL files in batch_results/ directory")


def main():
    """Main CLI entry point - like proactive_prep.sh but for batch APIs."""
    parser = argparse.ArgumentParser(
        description="Run native batch evaluation across multiple LLM providers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Run like your proactive_prep.sh but with batch APIs
            python run_native_batch_evaluation.py \\
                --models MODELS.OPENAI.GPT_4_1 MODELS.ANTHROPIC.CLAUDE_4_SONNET \\
                --data_dir data/sept22_inputs_hrd/
            
            # Limit to 20 files
            python run_native_batch_evaluation.py \\
                --models MODELS.TOGETHER_AI.DEEPSEEK_R1 MODELS.GOOGLE.GEMINI_25_FLASH \\
                --data_dir data/sept22_inputs_hrd/ \\
                --max_files 20

            # Process all files in directory  
            python run_native_batch_evaluation.py \\
                --models gpt-4o claude-3-5-sonnet-20241022 \\
                --data_dir data/my_test_data/
        """,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Models to test (e.g., MODELS.OPENAI.GPT_4_1 MODELS.ANTHROPIC.CLAUDE_4_SONNET)",
    )

    parser.add_argument(
        "--data_dir", required=True, help="Directory containing input JSON files"
    )

    parser.add_argument(
        "--max_files",
        type=int,
        help="Maximum number of files to process (default: all files)",
    )

    args = parser.parse_args()

    print("Native Batch API Evaluation (Dynamic)")
    print("====================================")
    print(f"Models: {', '.join(args.models)}")
    print(f"Data directory: {args.data_dir}")
    if args.max_files:
        print(f"Max files: {args.max_files}")
    else:
        print("Max files: all files in directory")
    print(f"Mode: ONE batch job per model containing ALL requests")
    print()

    try:
        asyncio.run(
            run_dynamic_batch_evaluation(args.models, args.data_dir, args.max_files)
        )
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()


async def run_default_batch_evaluation(data_dir: str, max_files: int = None):
    """Simple wrapper for backwards compatibility."""
    await run_native_batch_evaluation(data_dir, max_files)


if __name__ == "__main__":
    main()
