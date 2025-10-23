"""
Multi-LLM Client for PersonaSim Evaluation.

This module provides a client that distributes LLM calls across multiple providers
(OpenAI GPT, Google Gemini, Anthropic Claude) to improve diversity and reduce
single-model bias in generated content.
"""

import logging
import random
import time
from typing import List, Dict, Optional, Callable
from enum import Enum
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers."""

    OPENAI = "openai"
    OPENAI_MINI = "openai_mini"  # gpt-4o-mini as fallback
    CLAUDE = "claude"


@dataclass
class LLMConfig:
    """Configuration for a specific LLM provider."""

    provider: LLMProvider
    model_name: str
    api_key_env_var: str
    max_retries: int = 5  # Maximum number of retries per call
    timeout: float = 180.0
    base_delay: float = 1.0  # Base delay for exponential backoff in seconds
    max_delay: float = 60.0  # Maximum delay between retries in seconds


class MultiLLMClient:
    """
    Multi-provider LLM client that distributes calls across different models.

    This client:
    1. Selects providers using round-robin rotation
    2. Handles provider-specific API calls
    3. Implements fallback mechanisms
    4. Tracks usage statistics
    5. Includes retry logic with exponential backoff
    """

    def __init__(
        self,
        configs: List[LLMConfig],
        fallback_order: Optional[List[LLMProvider]] = None,
        enable_parallel: bool = False,
    ):
        """
        Initialize the multi-LLM client.

        Args:
            configs: List of LLM provider configurations
            fallback_order: Order to try providers if primary fails
            enable_parallel: Whether to enable parallel generation for comparison
        """
        self.configs = {config.provider: config for config in configs}
        self.fallback_order = fallback_order or [
            LLMProvider.OPENAI,
            LLMProvider.CLAUDE,
            LLMProvider.OPENAI_MINI,
        ]
        self.enable_parallel = enable_parallel

        # Usage tracking
        self.usage_stats = {provider: 0 for provider in LLMProvider}
        self.error_stats = {provider: 0 for provider in LLMProvider}
        self.retry_stats = {provider: 0 for provider in LLMProvider}
        self._stats_lock = threading.Lock()

        # Round-robin selection
        self._round_robin_index = 0
        self._round_robin_lock = threading.Lock()

        # Initialize provider clients
        self._clients = {}
        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize API clients for each configured provider."""
        import os
        from dotenv import load_dotenv

        # Load environment variables from .env file
        load_dotenv()

        for provider, config in self.configs.items():
            api_key = os.getenv(config.api_key_env_var)
            if not api_key:
                logger.warning(
                    f"No API key found for {provider.value} (env var: {config.api_key_env_var})"
                )
                continue

            try:
                if provider == LLMProvider.OPENAI:
                    from openai import OpenAI

                    # Disable OpenAI's built-in retries so our exponential backoff can work
                    self._clients[provider] = OpenAI(api_key=api_key, max_retries=0)

                elif provider == LLMProvider.OPENAI_MINI:
                    from openai import OpenAI

                    # Use same OpenAI API key for the mini model
                    self._clients[provider] = OpenAI(api_key=api_key, max_retries=0)

                elif provider == LLMProvider.CLAUDE:
                    import anthropic

                    self._clients[provider] = anthropic.Anthropic(api_key=api_key)

                logger.info(
                    f"Initialized {provider.value} client with model {config.model_name}"
                )

            except ImportError as e:
                logger.warning(f"Failed to import {provider.value} client: {e}")
            except Exception as e:
                logger.error(f"Failed to initialize {provider.value} client: {e}")

    def _select_provider(
        self, exclude: Optional[List[LLMProvider]] = None
    ) -> LLMProvider:
        """
        Select a provider using round-robin selection.

        Args:
            exclude: Providers to exclude from selection

        Returns:
            Selected provider
        """
        exclude = exclude or []
        available_providers = [
            provider
            for provider in self.configs.keys()
            if provider not in exclude and provider in self._clients
        ]

        if not available_providers:
            raise RuntimeError("No available LLM providers")

        # Round-robin selection
        with self._round_robin_lock:
            selected_provider = available_providers[
                self._round_robin_index % len(available_providers)
            ]
            self._round_robin_index += 1

        return selected_provider

    def _calculate_delay(
        self, attempt: int, base_delay: float, max_delay: float
    ) -> float:
        """
        Calculate exponential backoff delay.

        Args:
            attempt: Current attempt number (0-indexed)
            base_delay: Base delay in seconds
            max_delay: Maximum delay in seconds

        Returns:
            Delay in seconds
        """
        # Exponential backoff: base_delay * (2 ^ attempt) + jitter
        delay = base_delay * (2**attempt)

        # Add jitter (±25% of the calculated delay)
        jitter = delay * 0.25 * (2 * random.random() - 1)
        delay = delay + jitter

        # Cap at max_delay
        return min(delay, max_delay)

    def _call_openai(self, prompt: str, config: LLMConfig) -> str:
        """Call OpenAI API."""
        # Use the appropriate client based on provider
        if config.provider == LLMProvider.OPENAI_MINI:
            client = self._clients[LLMProvider.OPENAI_MINI]
        else:
            client = self._clients[LLMProvider.OPENAI]

        # Some models (o1, gpt-5-mini) don't support custom temperature
        call_params = {
            "model": config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "timeout": config.timeout,
        }

        # Only add temperature if the model supports it
        if not any(x in config.model_name.lower() for x in ["o1", "gpt-5"]):
            call_params["temperature"] = 0.7

        response = client.chat.completions.create(**call_params)

        return response.choices[0].message.content

    def _call_claude(self, prompt: str, config: LLMConfig) -> str:
        """Call Anthropic Claude API."""
        # Use the provider from config for consistency
        client = self._clients[config.provider]

        response = client.messages.create(
            model=config.model_name,
            max_tokens=9000,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _call_openai_json(self, prompt: str, config: LLMConfig) -> str:
        """Call OpenAI API with JSON formatting."""
        # Use the appropriate client based on provider
        if config.provider == LLMProvider.OPENAI_MINI:
            client = self._clients[LLMProvider.OPENAI_MINI]
        else:
            client = self._clients[LLMProvider.OPENAI]

        # Some models (o1, gpt-5-mini) don't support custom temperature
        call_params = {
            "model": config.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert at generating realistic professional contexts. Always respond with valid, complete JSON. Every object in arrays must include ALL required fields specified in the schema. Do not return partial or incomplete objects.",
                },
                {"role": "user", "content": prompt},
            ],
            "timeout": config.timeout,
            "response_format": {"type": "json_object"},
        }

        # Only add temperature if the model supports it
        if not any(x in config.model_name.lower() for x in ["o1", "gpt-5"]):
            call_params["temperature"] = 0.7

        response = client.chat.completions.create(**call_params)

        return response.choices[0].message.content

    def _call_claude_json(self, prompt: str, config: LLMConfig) -> str:
        """Call Anthropic Claude API with JSON formatting."""
        client = self._clients[LLMProvider.CLAUDE]

        response = client.messages.create(
            model=config.model_name,
            max_tokens=9000,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are an expert at generating realistic professional contexts.\n\n"
                        "CRITICAL REQUIREMENTS:\n"
                        "1. Your response MUST be ONLY valid, complete JSON\n"
                        "2. Every object in arrays MUST include ALL required fields as specified in the prompt\n"
                        "3. Do NOT include partial objects, missing fields, or incomplete data\n"
                        "4. Do NOT include explanatory text, markdown, or code blocks\n"
                        "5. Start your response with { and end with }\n\n"
                        f"{prompt}"
                    ),
                },
            ],
        )

        # Clean up Claude's response - remove common wrapper text
        content = response.content[0].text.strip()

        # Remove markdown code blocks if present
        if content.startswith("```json"):
            content = content.replace("```json", "", 1)  # Remove ```json
        if content.startswith("```"):
            content = content.replace("```", "", 1)  # Remove ```
        if content.endswith("```"):
            content = content.replace("```", "", 1)  # Remove trailing ```

        return content.strip()

    def _call_provider_once(self, provider: LLMProvider, prompt: str) -> str:
        """
        Make a single API call to a specific provider.

        Args:
            provider: LLM provider to use
            prompt: Prompt to send

        Returns:
            Generated response

        Raises:
            Exception: If the API call fails
        """
        config = self.configs[provider]

        if provider == LLMProvider.OPENAI:
            return self._call_openai(prompt, config)
        elif provider == LLMProvider.OPENAI_MINI:
            return self._call_openai(prompt, config)  # Use same OpenAI call method
        elif provider == LLMProvider.CLAUDE:
            return self._call_claude(prompt, config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def _call_provider_once_json(self, provider: LLMProvider, prompt: str) -> str:
        """
        Make a single API call to a specific provider for JSON generation.

        Args:
            provider: LLM provider to use
            prompt: Prompt to send

        Returns:
            Generated JSON response

        Raises:
            Exception: If the API call fails
        """
        config = self.configs[provider]

        if provider == LLMProvider.OPENAI:
            return self._call_openai_json(prompt, config)
        elif provider == LLMProvider.OPENAI_MINI:
            return self._call_openai_json(
                prompt, config
            )  # Use same OpenAI JSON call method
        elif provider == LLMProvider.CLAUDE:
            return self._call_claude_json(prompt, config)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def generate(
        self, prompt: str, preferred_provider: Optional[LLMProvider] = None
    ) -> str:
        """
        Generate content using the multi-LLM system with cross-provider retry strategy.

        On failures, immediately switches to the next available provider instead of
        retrying the same provider multiple times.

        Args:
            prompt: Input prompt
            preferred_provider: Optional preferred provider

        Returns:
            Generated content
        """
        return self._generate_with_cross_provider_retry(
            prompt, preferred_provider, use_json=False
        )

    def generate_json(
        self, prompt: str, preferred_provider: Optional[LLMProvider] = None
    ) -> str:
        """
        Generate JSON content using the multi-LLM system with cross-provider retry strategy.

        On failures, immediately switches to the next available provider instead of
        retrying the same provider multiple times.

        Args:
            prompt: Input prompt
            preferred_provider: Optional preferred provider

        Returns:
            Generated JSON content
        """
        return self._generate_with_cross_provider_retry(
            prompt, preferred_provider, use_json=True
        )

    def _generate_with_cross_provider_retry(
        self,
        prompt: str,
        preferred_provider: Optional[LLMProvider] = None,
        use_json: bool = False,
    ) -> str:
        """
        Generate content with active cross-provider rotation and retry strategy.

        Actively rotates between providers on every call for load distribution,
        and switches to the next provider on failures with exponential backoff.

        Args:
            prompt: Input prompt
            preferred_provider: Optional preferred provider to start with
            use_json: Whether to use JSON-specific API calls

        Returns:
            Generated content

        Raises:
            RuntimeError: If all providers fail after cycling through them
        """
        # Get available providers in order
        available_providers = [
            provider for provider in self.fallback_order if provider in self._clients
        ]

        if not available_providers:
            raise RuntimeError("No available LLM providers")

        # Select starting provider using round-robin (preferred provider overrides this)
        if preferred_provider and preferred_provider in available_providers:
            starting_provider = preferred_provider
        else:
            starting_provider = self._select_provider()

        # Find the index of the starting provider in available_providers
        try:
            start_index = available_providers.index(starting_provider)
        except ValueError:
            # If starting provider not in available_providers, start from 0
            start_index = 0
            starting_provider = available_providers[0]

        # Calculate total attempts across all providers
        max_total_attempts = sum(
            self.configs[provider].max_retries for provider in available_providers
        )

        # Start from the selected provider index
        provider_index = start_index
        total_attempts = 0
        provider_attempt_counts = {provider: 0 for provider in available_providers}

        logger.debug(
            f"Starting provider rotation with {starting_provider.value} "
            f"(index {start_index} of {len(available_providers)} providers)"
        )

        while total_attempts < max_total_attempts:
            current_provider = available_providers[provider_index]
            config = self.configs[current_provider]

            # Check if this provider has exhausted its retry limit
            if provider_attempt_counts[current_provider] >= config.max_retries:
                # Move to next provider
                provider_index = (provider_index + 1) % len(available_providers)

                # If we've cycled through all providers and all are exhausted, break
                if all(
                    provider_attempt_counts[p] >= self.configs[p].max_retries
                    for p in available_providers
                ):
                    break
                continue

            try:
                start_time = time.time()

                # Make the API call (JSON or regular)
                if use_json:
                    response = self._call_provider_once_json(current_provider, prompt)
                else:
                    response = self._call_provider_once(current_provider, prompt)

                elapsed_time = time.time() - start_time

                # Success! Update stats and return
                with self._stats_lock:
                    self.usage_stats[current_provider] += 1
                    # Track retry stats (total attempts - 1 for this provider)
                    if provider_attempt_counts[current_provider] > 0:
                        self.retry_stats[current_provider] += provider_attempt_counts[
                            current_provider
                        ]

                logger.debug(
                    f"Successfully generated {'JSON ' if use_json else ''}content using "
                    f"{current_provider.value} after {total_attempts + 1} total attempts "
                    f"({provider_attempt_counts[current_provider] + 1} for this provider) in {elapsed_time:.2f}s"
                )

                return response

            except Exception as e:
                provider_attempt_counts[current_provider] += 1
                total_attempts += 1

                logger.warning(
                    f"{current_provider.value} {'JSON ' if use_json else ''}API call failed "
                    f"(attempt {provider_attempt_counts[current_provider]}/{config.max_retries} "
                    f"for this provider, {total_attempts}/{max_total_attempts} total): {str(e)}"
                )

                # Apply exponential backoff before switching providers
                if total_attempts < max_total_attempts:
                    delay = self._calculate_delay(
                        provider_attempt_counts[current_provider] - 1,
                        config.base_delay,
                        config.max_delay,
                    )

                    # Switch to next provider for the next attempt
                    next_provider_index = (provider_index + 1) % len(
                        available_providers
                    )
                    next_provider = available_providers[next_provider_index]

                    logger.debug(
                        f"Switching from {current_provider.value} to {next_provider.value} "
                        f"after {delay:.2f}s delay (attempt {total_attempts + 1}/{max_total_attempts})"
                    )

                    time.sleep(delay)
                    provider_index = next_provider_index

        # All providers exhausted
        with self._stats_lock:
            for provider in available_providers:
                self.error_stats[provider] += 1

        # Create detailed error message
        provider_attempts = [
            f"{provider.value}: {provider_attempt_counts[provider]}/{self.configs[provider].max_retries}"
            for provider in available_providers
        ]

        raise RuntimeError(
            f"All LLM providers failed after {total_attempts} total attempts. "
            f"Provider attempts: {', '.join(provider_attempts)}"
        )

    def generate_parallel(self, prompt: str, num_variants: int = 2) -> List[str]:
        """
        Generate multiple variants in parallel using different providers.

        Args:
            prompt: Input prompt
            num_variants: Number of variants to generate

        Returns:
            List of generated variants
        """
        if not self.enable_parallel:
            return [self.generate(prompt)]

        available_providers = list(self._clients.keys())
        if len(available_providers) < num_variants:
            logger.warning(
                f"Only {len(available_providers)} providers available for {num_variants} variants"
            )
            num_variants = len(available_providers)

        selected_providers = random.sample(available_providers, num_variants)

        results = []
        with ThreadPoolExecutor(max_workers=num_variants) as executor:
            # Submit all generation tasks using single attempts (no cross-provider retry in parallel mode)
            future_to_provider = {
                executor.submit(self._call_provider_once, provider, prompt): provider
                for provider in selected_providers
            }

            # Collect results as they complete
            for future in as_completed(future_to_provider):
                provider = future_to_provider[future]
                try:
                    result = future.result()
                    results.append(result)

                    # Update usage stats
                    with self._stats_lock:
                        self.usage_stats[provider] += 1

                except Exception as e:
                    logger.error(
                        f"Parallel generation failed for {provider.value}: {e}"
                    )
                    with self._stats_lock:
                        self.error_stats[provider] += 1

        if not results:
            raise RuntimeError("All parallel generation attempts failed")

        return results

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage, error, and retry statistics."""
        with self._stats_lock:
            return {
                "usage": {
                    provider.value: count
                    for provider, count in self.usage_stats.items()
                },
                "errors": {
                    provider.value: count
                    for provider, count in self.error_stats.items()
                },
                "retries": {
                    provider.value: count
                    for provider, count in self.retry_stats.items()
                },
            }

    def create_llm_function(self) -> Callable[[str], str]:
        """
        Create a function compatible with existing LLM function interfaces.

        Returns:
            Function that takes a prompt and returns generated content
        """
        return lambda prompt: self.generate(prompt)

    def create_json_llm_function(self) -> Callable[[str], str]:
        """
        Create a function compatible with existing LLM function interfaces for JSON generation.

        Returns:
            Function that takes a prompt and returns generated JSON content
        """
        return lambda prompt: self.generate_json(prompt)


def create_default_multi_llm_client() -> MultiLLMClient:
    """
    Create a default multi-LLM client with standard configurations.

    Matches Research_PROBE configuration:
    - Primary: gpt-4.1
    - Secondary: claude-sonnet-4-20250514
    - Fallback: gpt-4.1-mini (via OPENAI_MINI provider)

    Returns:
        Configured MultiLLMClient instance
    """
    configs = [
        LLMConfig(
            provider=LLMProvider.OPENAI,
            model_name="gpt-4.1",
            api_key_env_var="OPENAI_API_KEY",
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
        ),
        LLMConfig(
            provider=LLMProvider.OPENAI_MINI,
            model_name="gpt-4.1-mini",
            api_key_env_var="OPENAI_API_KEY",  # Same API key as main OpenAI
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
        ),
        LLMConfig(
            provider=LLMProvider.CLAUDE,
            model_name="claude-sonnet-4-20250514",
            api_key_env_var="ANTHROPIC_API_KEY",
            max_retries=5,
            base_delay=1.0,
            max_delay=60.0,
        ),
    ]

    return MultiLLMClient(
        configs=configs,
        fallback_order=[
            LLMProvider.OPENAI,
            LLMProvider.CLAUDE,
            LLMProvider.OPENAI_MINI,
        ],
        enable_parallel=True,  # Enable parallel processing for diverse generation
    )
