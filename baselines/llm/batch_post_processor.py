#!/usr/bin/env python3
"""
Simple batch post-processor for GPT-4.1-mini structured outputs.
Takes batch results, processes with GPT-4.1-mini, outputs structured JSON.
"""

import json
import os
import random
import time
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any
from openai import OpenAI, RateLimitError
from tqdm import tqdm

logging.basicConfig(level=logging.WARNING)  # Reduce logging verbosity
logger = logging.getLogger(__name__)


def parse_batch_response(line: str) -> Dict[str, Any]:
    """Parse a single line from batch results file."""
    data = json.loads(line.strip())
    custom_id = data.get("custom_id", "unknown")

    # Extract content based on provider format
    content = ""
    error = data.get("error")

    if error:
        return {"custom_id": custom_id, "content": "", "error": str(error)}

    response = data.get("response", {})

    # Anthropic format
    if "content" in response and isinstance(response["content"], list):
        if response["content"]:
            content = response["content"][0].get("text", "")

    # Together.ai format
    elif "body" in response:
        choices = response["body"].get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "") or message.get("reasoning", "")

    # OpenAI format
    elif "choices" in response:
        choices = response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content", "")

    return {"custom_id": custom_id, "content": content, "error": None}


def process_with_gpt41mini(
    content: str, custom_id: str, client: OpenAI
) -> Dict[str, Any]:
    """Process content with GPT-4.1-mini to get structured output."""
    if not content.strip():
        return {
            "retrieved_documents": [],
            "bottleneck": "",
            "action": {"function_name": "", "parameters": {}},
        }

    prompt = f"""Extract structured information from this AI response and return ONLY valid JSON:
    Retrieved content is a list of document **IDs** that were relevant to identifying the bottleneck.

    {{
    "retrieved_documents": ["doc_id1", "doc_id2"],
    "bottleneck": "description of the main problem identified", 
    "action": {{
        "function_name": "action_name",
        "parameters": {{"key": "value"}}
    }}
    }}

    Extract all relevant information from the input into this format.
    If the response is incomplete/failed, return empty values. Extract exact document IDs mentioned.

    AI Response:
    {content}

    JSON:"""

    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You extract structured data and return only valid JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )

            result = json.loads(response.choices[0].message.content)

            # Ensure required fields exist
            if "retrieved_documents" not in result:
                result["retrieved_documents"] = []
            if "bottleneck" not in result:
                result["bottleneck"] = ""
            if "action" not in result:
                result["action"] = {"function_name": "", "parameters": {}}
            elif "function_name" not in result["action"]:
                result["action"]["function_name"] = ""
            elif "parameters" not in result["action"]:
                result["action"]["parameters"] = {}

            return result

        except RateLimitError:
            if attempt == 4:
                raise
            wait_time = (2**attempt) + random.uniform(0, 1)
            time.sleep(wait_time)

        except Exception:
            return {
                "retrieved_documents": [],
                "bottleneck": "",
                "action": {"function_name": "", "parameters": {}},
            }

    return {
        "retrieved_documents": [],
        "bottleneck": "",
        "action": {"function_name": "", "parameters": {}},
    }


def process_batch_file(batch_file: Path, output_dir: Path = None) -> list:
    """Process batch file with GPT-4.1-mini."""
    if output_dir is None:
        # Save structured results in the same directory as the batch file
        output_dir = batch_file.parent

    print(f"Processing {batch_file.name}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse all responses
    responses = []
    with open(batch_file, "r") as f:
        for line in f:
            if line.strip():
                responses.append(parse_batch_response(line))

    # Filter valid responses
    valid_responses = [resp for resp in responses if not resp["error"]]

    # Process with threading and progress bar
    saved_files = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_id = {
            executor.submit(
                process_with_gpt41mini, resp["content"], resp["custom_id"], client
            ): resp["custom_id"]
            for resp in valid_responses
        }

        with tqdm(total=len(future_to_id), desc="Processing", unit="files") as pbar:
            for future in as_completed(future_to_id):
                custom_id = future_to_id[future]
                try:
                    result = future.result()
                    result["custom_id"] = custom_id

                    # Write output immediately
                    if custom_id.endswith(".json"):
                        individual_file = output_dir / custom_id
                    else:
                        individual_file = output_dir / f"{custom_id}.json"

                    with open(individual_file, "w") as f:
                        json.dump(result, f, indent=2)
                    saved_files.append(individual_file)

                except Exception:
                    # Write error result immediately
                    error_result = {
                        "custom_id": custom_id,
                        "retrieved_documents": [],
                        "bottleneck": "",
                        "action": {"function_name": "", "parameters": {}},
                    }

                    if custom_id.endswith(".json"):
                        individual_file = output_dir / custom_id
                    else:
                        individual_file = output_dir / f"{custom_id}.json"

                    with open(individual_file, "w") as f:
                        json.dump(error_result, f, indent=2)
                    saved_files.append(individual_file)

                pbar.update(1)

    # Completed - result count shown by tqdm
    return saved_files


def process_all_batches(input_dir: str = "batch_results"):
    """Process all batch files in specified directory structure."""
    results_dir = Path(input_dir)

    # Find all batch JSONL files in the results directory structure
    batch_files = []
    if results_dir.exists():
        for batch_file in results_dir.rglob("*.jsonl"):
            if "batch_" in batch_file.parent.name:
                batch_files.append(batch_file)

    if not batch_files:
        print(f"No batch files found in {input_dir}/")
        return

    successful = 0
    for batch_file in batch_files:
        try:
            output_files = process_batch_file(batch_file, output_dir=None)
            if output_files:
                successful += len(output_files)
        except Exception as e:
            print(f"❌ {batch_file.name}: {e}")

    print(f"✅ {successful} files processed")


def main():
    """Process all batch files with GPT-4.1-mini."""
    parser = argparse.ArgumentParser(
        description="Process batch results with GPT-4.1-mini"
    )
    parser.add_argument(
        "--input-dir",
        default="batch_results",
        help="Directory containing batch result files (default: batch_results)",
    )
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable required")
        return 1

    try:
        process_all_batches(args.input_dir)
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
