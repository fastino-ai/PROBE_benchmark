"""LiteLLM model wrapper for multi-provider LLM support."""

import os
from typing import Any, Dict, List, Optional, Union
import litellm
from litellm import completion


class LitellmModel:
    """Wrapper class for LiteLLM models that provides OpenAI-compatible interface."""

    def __init__(self, model: str, api_key: Optional[str] = None, **kwargs):
        """
        Initialize LiteLLM model.

        Args:
            model: Model identifier (e.g., "anthropic/claude-3-opus-20240229")
            api_key: Optional API key (if not set in environment)
            **kwargs: Additional model configuration
        """
        self.model = model
        self.api_key = api_key
        self.kwargs = kwargs

        # Set API key if provided
        if api_key:
            provider = model.split("/")[0] if "/" in model else None
            if provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            elif provider == "google" or provider == "gemini":
                os.environ["GOOGLE_API_KEY"] = api_key
            elif provider == "together_ai":
                os.environ["TOGETHERAI_API_KEY"] = api_key
            # Add more providers as needed

    def create_chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        stop: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ) -> Any:
        """
        Create a chat completion using LiteLLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            LiteLLM response object (OpenAI-compatible format)
        """
        # Merge instance kwargs with call kwargs
        call_kwargs = {**self.kwargs, **kwargs}

        # Add optional parameters if provided
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens
        if top_p is not None:
            call_kwargs["top_p"] = top_p
        if frequency_penalty is not None:
            call_kwargs["frequency_penalty"] = frequency_penalty
        if presence_penalty is not None:
            call_kwargs["presence_penalty"] = presence_penalty
        if stop is not None:
            call_kwargs["stop"] = stop

        # Use litellm completion
        response = completion(model=self.model, messages=messages, **call_kwargs)

        return response

    def __str__(self):
        """String representation of the model."""
        return f"LitellmModel({self.model})"

    def __repr__(self):
        """Detailed representation of the model."""
        return f"LitellmModel(model='{self.model}', kwargs={self.kwargs})"
