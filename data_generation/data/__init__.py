"""
Data module for PersonaSim agents.

This module provides integration with LinkedIn persona data.
"""

from .linkedin_profile import (
    LinkedInPersona,
    load_linkedin_personas,
    save_personas_to_csv,
    load_personas_from_csv,
    get_persona_by_id,
    download_and_save_dataset,
)

__all__ = [
    "LinkedInPersona",
    "load_linkedin_personas",
    "save_personas_to_csv",
    "load_personas_from_csv",
    "get_persona_by_id",
    "download_and_save_dataset",
]
