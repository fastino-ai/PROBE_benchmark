"""
Utilities for data generation package.

This module provides standalone utilities that were previously
dependencies on the brain module.
"""

from .clients.openai_client import get_openai_client

__all__ = [
    "get_openai_client",
]
