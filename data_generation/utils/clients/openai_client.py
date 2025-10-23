"""
OpenAI Client

Standalone OpenAI client utility for data_generation package.
This replaces the dependency on brain.utils.clients.openai_client.
"""

import os
from typing import Optional
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global client cache
_client: Optional[OpenAI] = None


def get_openai_client() -> OpenAI:
    """
    Get OpenAI client instance.

    Creates a cached OpenAI client instance using the OPENAI_API_KEY
    environment variable.

    Returns:
        OpenAI client instance

    Raises:
        ValueError: If OPENAI_API_KEY is not set
    """
    global _client

    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set. "
                "Please set it with: export OPENAI_API_KEY='your-key-here' "
                "or add it to a .env file"
            )

        _client = OpenAI(api_key=api_key)

    return _client


def reset_client() -> None:
    """
    Reset the cached client instance.

    This is useful for testing or when you need to refresh the client
    with new configuration.
    """
    global _client
    _client = None
