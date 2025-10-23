"""Base agent class with LiteLLM support."""

import os
from typing import Any, Dict, List, Optional, Union
from openai import OpenAI

# Explicitly import our local LitellmModel
from baselines.agentic.litellm_model import LitellmModel


class BaseAgent:
    """Base agent class that supports both OpenAI and LiteLLM models."""

    def __init__(self, model: Union[str, LitellmModel] = "gpt-4o", **kwargs):
        """
        Initialize base agent with model.

        Args:
            model: Either a string (for OpenAI models) or a LitellmModel instance
            **kwargs: Additional configuration for the agent
        """
        self.model = model
        self.kwargs = kwargs

        # Initialize client based on model type
        if isinstance(model, str):
            # Standard OpenAI model
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.is_litellm = False
        elif hasattr(model, "create_chat_completion"):
            # LiteLLM model or any model with create_chat_completion method
            self.client = model
            self.is_litellm = True
        else:
            raise ValueError(
                f"Model must be either a string or an object with create_chat_completion method, got {type(model)}"
            )

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
        Create a chat completion using either OpenAI or LiteLLM.

        Args:
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            frequency_penalty: Frequency penalty
            presence_penalty: Presence penalty
            stop: Stop sequences
            **kwargs: Additional parameters

        Returns:
            Response object in OpenAI format
        """
        if self.is_litellm:
            # Use LiteLLM
            return self.client.create_chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )
        else:
            # Use OpenAI
            call_kwargs = kwargs.copy()

            # Add optional parameters if provided
            if temperature is not None:
                call_kwargs["temperature"] = temperature
            if max_tokens is not None:
                # Use max_completion_tokens for newer OpenAI models
                if (
                    hasattr(self, "model")
                    and isinstance(self.model, str)
                    and ("gpt-5" in self.model.lower() or "o1" in self.model.lower())
                ):
                    call_kwargs["max_completion_tokens"] = max_tokens
                else:
                    call_kwargs["max_tokens"] = max_tokens
            if top_p is not None:
                call_kwargs["top_p"] = top_p
            if frequency_penalty is not None:
                call_kwargs["frequency_penalty"] = frequency_penalty
            if presence_penalty is not None:
                call_kwargs["presence_penalty"] = presence_penalty
            if stop is not None:
                call_kwargs["stop"] = stop

            return self.client.chat.completions.create(
                model=self.model, messages=messages, **call_kwargs
            )

    def get_model_string(self) -> str:
        """Get the model identifier as a string."""
        if isinstance(self.model, str):
            return self.model
        elif isinstance(self.model, LitellmModel):
            return self.model.model
        else:
            return str(self.model)

    def supports_temperature(self) -> bool:
        """Check if the model supports custom temperature values."""
        model_name = self.get_model_string().lower()
        # GPT-5 and O1 models don't support custom temperature
        return not ("gpt-5" in model_name or "o1" in model_name)
