#!/usr/bin/env python3
"""
LLM JSON Post-Processor for PROBE Evaluation

This module provides robust JSON post-processing for LLM responses in evaluation.
It handles common LLM JSON formatting issues and uses LLM self-correction when needed.
"""

import json
import re
import logging
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class JSONParseResult:
    """Result of JSON parsing attempt."""

    success: bool
    data: Optional[Dict[str, Any]]
    error: Optional[str]
    method_used: str
    original_response: str
    fixed_response: Optional[str] = None


class LLMJSONPostProcessor:
    """
    Robust JSON post-processor for LLM evaluation responses.

    Tries multiple strategies to extract valid JSON from LLM responses:
    1. Direct parsing
    2. Common fixes (quotes, braces, etc.)
    3. Regex extraction
    4. LLM self-correction
    5. Fallback parsing
    """

    def __init__(self, llm_generate_func: Optional[Callable] = None):
        """
        Initialize the post-processor.

        Args:
            llm_generate_func: Function to call LLM for self-correction
        """
        self.llm_generate_func = llm_generate_func

    def parse_json_response(
        self, response: str, expected_fields: Optional[list] = None
    ) -> JSONParseResult:
        """
        Parse JSON from LLM response using LLM-only approach for consistent evaluation.

        Args:
            response: Raw LLM response text
            expected_fields: List of expected JSON fields (e.g., ["judgment", "reasoning"])

        Returns:
            JSONParseResult with parsed data or error details
        """
        original_response = response.strip()

        # Strategy 1: Direct parsing (always try first)
        result = self._try_direct_parse(original_response)
        if result.success:
            return result

        # Strategy 2: LLM self-correction (primary and only fix method)
        if self.llm_generate_func:
            result = self._try_llm_correction(original_response, expected_fields)
            if result.success:
                return result
        else:
            # No LLM available - return structured error
            return JSONParseResult(
                success=False,
                data=None,
                error="LLM post-processor not available for JSON correction",
                method_used="llm_unavailable",
                original_response=original_response,
            )

        # Strategy 3: Intelligent fallback (last resort with consistent interpretation)
        result = self._try_fallback_parsing(original_response, expected_fields)
        if result.success:
            return result

        # All strategies failed
        return JSONParseResult(
            success=False,
            data=None,
            error="LLM JSON correction failed",
            method_used="llm_correction_failed",
            original_response=original_response,
        )

    def _try_direct_parse(self, response: str) -> JSONParseResult:
        """Try direct JSON parsing."""
        try:
            data = json.loads(response)
            return JSONParseResult(
                success=True,
                data=data,
                error=None,
                method_used="direct_parse",
                original_response=response,
            )
        except json.JSONDecodeError as e:
            return JSONParseResult(
                success=False,
                data=None,
                error=str(e),
                method_used="direct_parse",
                original_response=response,
            )

    def _try_common_fixes(self, response: str) -> JSONParseResult:
        """Apply common JSON fixes and try parsing."""
        fixed = response
        fixes_applied = []

        # Fix 1: Extract JSON from markdown code blocks
        if "```json" in fixed or "```" in fixed:
            # Extract content between ```json and ```
            json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", fixed, re.DOTALL)
            if json_match:
                fixed = json_match.group(1)
                fixes_applied.append("extracted_from_markdown")

        # Fix 2: Remove any text before first { or after last }
        first_brace = fixed.find("{")
        last_brace = fixed.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            fixed = fixed[first_brace : last_brace + 1]
            fixes_applied.append("trimmed_to_braces")

        # Fix 3: Remove trailing commas
        if re.search(r",(\s*[}\]])", fixed):
            fixed = re.sub(r",(\s*[}\]])", r"\1", fixed)
            fixes_applied.append("removed_trailing_commas")

        # Fix 4: Fix unescaped quotes in strings
        # Look for "text with "quotes" inside" patterns
        def fix_inner_quotes(match):
            content = match.group(1)
            # Escape internal quotes
            content = content.replace('"', '\\"')
            return f'"{content}"'

        if '".*".*".*"' in fixed:
            try:
                # Be conservative - only fix obvious cases
                fixed = re.sub(r'"([^"]*"[^"]*)"(?=\s*[,}])', fix_inner_quotes, fixed)
                fixes_applied.append("escaped_inner_quotes")
            except Exception:
                pass  # Skip if regex fails

        # Fix 5: Complete missing braces
        open_braces = fixed.count("{")
        close_braces = fixed.count("}")
        if open_braces > close_braces:
            fixed += "}" * (open_braces - close_braces)
            fixes_applied.append("added_closing_braces")

        # Fix 6: Handle truncated strings
        if fixed.endswith('"') and fixed.count('"') % 2 == 1:
            # Find the last complete key-value pair
            last_complete = fixed.rfind('","')
            if last_complete != -1:
                fixed = fixed[: last_complete + 1] + "}"
                fixes_applied.append("fixed_truncated_string")

        # Try parsing the fixed version
        try:
            data = json.loads(fixed)
            return JSONParseResult(
                success=True,
                data=data,
                error=None,
                method_used=f"common_fixes: {', '.join(fixes_applied)}",
                original_response=response,
                fixed_response=fixed,
            )
        except json.JSONDecodeError as e:
            return JSONParseResult(
                success=False,
                data=None,
                error=f"Fixed version still invalid: {e}",
                method_used="common_fixes",
                original_response=response,
                fixed_response=fixed,
            )

    def _try_regex_extraction(
        self, response: str, expected_fields: Optional[list]
    ) -> JSONParseResult:
        """Extract key-value pairs using regex when JSON is malformed."""
        if not expected_fields:
            expected_fields = ["judgment", "reasoning"]  # Default for evaluation

        extracted = {}

        for field in expected_fields:
            # Try multiple patterns for each field
            patterns = [
                rf'"{field}":\s*"([^"]*)"',  # "field": "value"
                rf'"{field}":\s*([^,\}}]+)',  # "field": value (no quotes)
                rf'{field}:\s*"([^"]*)"',  # field: "value" (no quotes on key)
                rf"{field}:\s*([^,\n\}}]+)",  # field: value (no quotes at all)
            ]

            for pattern in patterns:
                match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
                if match:
                    value = match.group(1).strip()
                    # Clean up the value
                    if value.endswith(","):
                        value = value[:-1]
                    extracted[field] = value
                    break

        if extracted:
            return JSONParseResult(
                success=True,
                data=extracted,
                error=None,
                method_used="regex_extraction",
                original_response=response,
            )
        else:
            return JSONParseResult(
                success=False,
                data=None,
                error="No fields extracted via regex",
                method_used="regex_extraction",
                original_response=response,
            )

    def _try_llm_correction(
        self, response: str, expected_fields: Optional[list]
    ) -> JSONParseResult:
        """Use LLM to self-correct the JSON format."""
        if not self.llm_generate_func:
            return JSONParseResult(
                success=False,
                data=None,
                error="LLM function not available",
                method_used="llm_correction",
                original_response=response,
            )

        expected_fields_str = (
            ", ".join(expected_fields) if expected_fields else "judgment, reasoning"
        )

        correction_prompt = f"""
You are a JSON formatting specialist. The following text should be valid JSON but has formatting issues.

TASK: Fix the JSON formatting and return ONLY the corrected JSON object.

REQUIRED FIELDS: {expected_fields_str}
REQUIRED FORMAT: {{"judgment": "CORRECT|PARTIALLY_CORRECT|INCORRECT", "reasoning": "explanation here"}}

ORIGINAL TEXT TO FIX:
{response}

INSTRUCTIONS:
1. Extract the judgment and reasoning from the original text
2. Return ONLY valid JSON in the exact format shown above
3. Do not include any markdown, explanations, or additional text
4. If the original judgment is unclear, use "INCORRECT" as default

CORRECTED JSON:"""

        try:
            corrected_response = self.llm_generate_func(correction_prompt)

            # Try parsing the corrected response
            try:
                data = json.loads(corrected_response.strip())
                return JSONParseResult(
                    success=True,
                    data=data,
                    error=None,
                    method_used="llm_correction",
                    original_response=response,
                    fixed_response=corrected_response,
                )
            except json.JSONDecodeError as e:
                return JSONParseResult(
                    success=False,
                    data=None,
                    error=f"LLM correction still invalid: {e}",
                    method_used="llm_correction",
                    original_response=response,
                    fixed_response=corrected_response,
                )

        except Exception as e:
            return JSONParseResult(
                success=False,
                data=None,
                error=f"LLM correction failed: {e}",
                method_used="llm_correction",
                original_response=response,
            )

    def _try_fallback_parsing(
        self, response: str, expected_fields: Optional[list]
    ) -> JSONParseResult:
        """Last resort fallback parsing with defaults."""
        if not expected_fields:
            expected_fields = ["judgment", "reasoning"]

        # Look for key information in the text
        fallback_data = {}

        # Try to extract judgment
        if "judgment" in expected_fields:
            judgment = "INCORRECT"  # Default
            response_lower = response.lower()
            if "correct" in response_lower and "incorrect" not in response_lower:
                if "partially" in response_lower or "partial" in response_lower:
                    judgment = "PARTIALLY_CORRECT"
                else:
                    judgment = "CORRECT"
            elif "partially" in response_lower or "partial" in response_lower:
                judgment = "PARTIALLY_CORRECT"
            fallback_data["judgment"] = judgment

        # Try to extract reasoning
        if "reasoning" in expected_fields:
            # Use the original response as reasoning, cleaned up
            reasoning = response.strip()
            # Remove JSON artifacts
            reasoning = re.sub(r'[{}"]', "", reasoning)
            reasoning = re.sub(
                r"judgment\s*[:=]\s*\w+", "", reasoning, flags=re.IGNORECASE
            )
            reasoning = re.sub(
                r"reasoning\s*[:=]\s*", "", reasoning, flags=re.IGNORECASE
            )
            reasoning = reasoning.strip()
            if reasoning:
                fallback_data["reasoning"] = reasoning
            else:
                fallback_data["reasoning"] = (
                    "Unable to parse detailed reasoning from response"
                )

        return JSONParseResult(
            success=True,
            data=fallback_data,
            error=None,
            method_used="fallback_parsing",
            original_response=response,
        )


def create_robust_json_parser(
    llm_generate_func: Optional[Callable] = None,
) -> LLMJSONPostProcessor:
    """
    Factory function to create a robust JSON parser.

    Args:
        llm_generate_func: Optional LLM function for self-correction

    Returns:
        Configured LLMJSONPostProcessor
    """
    return LLMJSONPostProcessor(llm_generate_func)


# Example usage
if __name__ == "__main__":
    # Test with malformed JSON
    processor = LLMJSONPostProcessor()

    test_cases = [
        '{"judgment": "CORRECT", "reasoning": "The analysis is accurate"}',  # Valid
        '{"judgment": "CORRECT", "reasoning": "The analysis is accurate",}',  # Trailing comma
        "judgment: CORRECT, reasoning: The analysis is accurate",  # No braces/quotes
        '```json\n{"judgment": "CORRECT"}\n```',  # Markdown format
        '{"judgment": "CORRECT", "reasoning": "Text with "quotes" inside"}',  # Unescaped quotes
    ]

    for i, test in enumerate(test_cases):
        print(f"\nTest {i + 1}: {test[:50]}...")
        result = processor.parse_json_response(test)
        print(f"Success: {result.success}, Method: {result.method_used}")
        if result.success:
            print(f"Data: {result.data}")
        else:
            print(f"Error: {result.error}")
