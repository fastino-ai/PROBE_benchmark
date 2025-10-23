"""
LinkedIn Persona Integration (PR8).

This module provides integration with the pre-existing LinkedIn dataset
from Hugging Face: fastino/linkedin_profiles

Dataset record examples:
600,"{'Full Name': 'Adriano Lujan, MPA', 'Workplace': 'Marketing and Growth Manager for Optum', 'Location': 'Albuquerque, New Mexico, United States', 'Connections': '233', 'Photo': 'No', 'Followers': '233'}","Adriano Lujan, MPA",Marketing and Growth Manager for Optum,"Experienced Community Coordinator and marketing professional with a demonstrated history of working in the health care industry. Skilled in Health Insurance, Program Evaluation, Volunteer Coordination, Public Speaking, and Health Policy. B.A. in Political Science, and a Master of Public Administration (MPA) in Health Policy and Administration from The University of New Mexico. Licensed Insurance Agent in Health & Accident"

Desired fields:
name: str
occupation: str
location: str
about: str

implementation of file: load data from huggingface using hugginface datasets library, save to csv (method)
populate the class LinkedInPersona with the fields name, occupation, location, and about
"""

import ast
import csv
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any

from datasets import load_dataset, Dataset
from pydantic import BaseModel, Field, field_validator

DATASET_NAME = "fastino/linkedin_profiles"


logger = logging.getLogger(__name__)


# Mock LinkedIn persona for fallback when Hugging Face is not accessible
MOCK_LINKEDIN_PERSONAS = [
    {
        "name": "Sarah Chen, PhD",
        "occupation": "Senior Data Scientist at Google",
        "location": "San Francisco, California, United States",
        "about": "Experienced data scientist with 8+ years in machine learning and AI. PhD in Computer Science from Stanford. Specialized in deep learning, natural language processing, and computer vision. Led multiple cross-functional teams to deliver ML solutions at scale.",
    }
]


class LinkedInPersona(BaseModel):
    """LinkedIn persona model with essential fields.

    This model represents a simplified view of LinkedIn profiles, focusing on
    the core attributes needed for persona-based evaluation: name, occupation,
    location, and about section.
    """

    name: str = Field(..., description="Full name of the person")
    occupation: str = Field(..., description="Current job title or role")
    location: str = Field(..., description="Geographic location")
    about: str = Field(..., description="Professional summary or about section")

    @field_validator("name", "occupation", "location", "about")
    @classmethod
    def _strip_and_require_nonempty(cls, value: str) -> str:
        """Ensure fields are stripped and non-empty."""
        if value is None:
            raise ValueError("must not be None")
        elif isinstance(value, int):
            value = str(value)

        stripped_value = value.strip()
        if not stripped_value:
            raise ValueError("must not be empty")

        return stripped_value


def load_mock_linkedin_personas(limit: Optional[int] = None) -> List[LinkedInPersona]:
    """Load mock LinkedIn personas as fallback when Hugging Face is not accessible.

    Args:
        limit: Maximum number of personas to load (None for all)

    Returns:
        List of LinkedInPersona objects created from mock data
    """
    logger.info("Loading mock LinkedIn personas (Hugging Face not accessible)")

    mock_data = MOCK_LINKEDIN_PERSONAS
    if limit:
        mock_data = mock_data[:limit]

    personas = []
    for data in mock_data:
        try:
            persona = LinkedInPersona(**data)
            personas.append(persona)
        except Exception as e:
            logger.warning(f"Failed to create mock persona: {e}")

    logger.info(f"Successfully loaded {len(personas)} mock personas")
    return personas


def load_linkedin_dataset(dataset_name: str = DATASET_NAME) -> Dataset:
    """Load LinkedIn profiles dataset from Hugging Face.

    Args:
        dataset_name: Name of the Hugging Face dataset to load

    Returns:
        Dataset object containing LinkedIn profiles

    Raises:
        Exception: If dataset cannot be loaded
    """
    try:
        logger.info(f"Loading LinkedIn dataset from Hugging Face: {dataset_name}")
        dataset = load_dataset(dataset_name, split="train")
        logger.info(f"Successfully loaded {len(dataset)} profiles")
        return dataset
    except Exception as e:
        logger.error(f"Failed to load dataset {dataset_name}: {e}")
        raise


def extract_persona_from_record(record: Dict[str, Any]) -> Optional[LinkedInPersona]:
    """Extract LinkedInPersona from a dataset record.

    The dataset has columns: 'Unnamed: 0', 'Intro', 'Full Name', 'Workplace', 'About'
    The 'Intro' column contains a string representation of a dictionary with location info.

    Args:
        record: A single record from the dataset

    Returns:
        LinkedInPersona object if extraction successful, None otherwise
    """
    try:
        # Direct fields from the dataset
        name = record.get("Full Name", "").strip()
        workplace = record.get("Workplace", "").strip()
        about = record.get("About")

        if about is None:
            about = "No about provided"
        else:
            about = about.strip()

        # Extract location from the Intro field (dictionary string)
        intro_str = record.get("Intro", "")
        location = ""

        if intro_str:
            try:
                # Parse the string representation of dictionary
                intro_data = ast.literal_eval(intro_str)
                location = intro_data.get("Location", "").strip()
            except (ValueError, SyntaxError) as e:
                logger.debug(f"Failed to parse intro data: {e}")
                # If parsing fails, location remains empty

        if not all([name, workplace, location, about]):
            logger.warning(
                f"Missing required fields in profile: name={bool(name)}, "
                f"workplace={bool(workplace)}, location={bool(location)}, about={bool(about)}"
            )
            return None

        return LinkedInPersona(
            name=name, occupation=workplace, location=location, about=about
        )

    except Exception as e:
        logger.error(f"Unexpected error extracting persona: {e}")
        return None


def load_linkedin_personas(
    dataset_name: str = DATASET_NAME, limit: Optional[int] = None
) -> List[LinkedInPersona]:
    """Load LinkedIn personas from Hugging Face dataset with mock fallback.

    Args:
        dataset_name: Name of the Hugging Face dataset
        limit: Maximum number of personas to load (None for all)

    Returns:
        List of LinkedInPersona objects
    """
    try:
        # Try to load from Hugging Face first
        dataset = load_linkedin_dataset(dataset_name)

        personas = []
        for i, record in enumerate(dataset):
            if limit and i >= limit:
                break

            persona = extract_persona_from_record(record)
            if persona:
                personas.append(persona)

        logger.info(
            f"Successfully extracted {len(personas)} personas from Hugging Face"
        )
        return personas

    except Exception as e:
        logger.warning(
            f"Failed to load from Hugging Face ({e}), falling back to mock data"
        )
        return load_mock_linkedin_personas(limit=limit)


def save_personas_to_csv(
    personas: List[LinkedInPersona], output_path: Path, encoding: str = "utf-8"
) -> None:
    """Save LinkedInPersona objects to CSV file.

    Args:
        personas: List of LinkedInPersona objects to save
        output_path: Path to output CSV file
        encoding: File encoding to use
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding=encoding) as csvfile:
        fieldnames = ["name", "occupation", "location", "about"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for persona in personas:
            writer.writerow(persona.model_dump())

    logger.info(f"Saved {len(personas)} personas to {output_path}")


def load_personas_from_csv(
    csv_path: Path, encoding: str = "utf-8"
) -> List[LinkedInPersona]:
    """Load LinkedInPersona objects from CSV file.

    Args:
        csv_path: Path to CSV file containing personas
        encoding: File encoding to use

    Returns:
        List of LinkedInPersona objects
    """
    csv_path = Path(csv_path)
    personas = []

    with open(csv_path, "r", encoding=encoding) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            try:
                persona = LinkedInPersona(**row)
                personas.append(persona)
            except Exception as e:
                logger.warning(f"Failed to load persona from row: {e}")

    logger.info(f"Loaded {len(personas)} personas from {csv_path}")
    return personas


def get_persona_by_id(
    personas: List[LinkedInPersona], persona_id: int
) -> Optional[LinkedInPersona]:
    """Get a specific persona by index.

    Args:
        personas: List of LinkedInPersona objects
        persona_id: Index of the persona to retrieve

    Returns:
        LinkedInPersona object if found, None otherwise
    """
    if 0 <= persona_id < len(personas):
        return personas[persona_id]
    return None


# Example usage functions
def download_and_save_dataset(
    output_csv: Path = Path("linkedin_personas.csv"), limit: Optional[int] = None
) -> None:
    """Download LinkedIn dataset and save to CSV.

    Args:
        output_csv: Path to save the CSV file
        limit: Maximum number of records to process
    """
    logger.info("Starting LinkedIn dataset download and conversion")
    personas = load_linkedin_personas(limit=limit)
    save_personas_to_csv(personas, output_csv)
    logger.info(f"Process complete. Saved to {output_csv}")


if __name__ == "__main__":
    # Example: Download first 100 profiles and save to CSV
    download_and_save_dataset(
        output_csv=Path("data_generation/data/linkedin_personas_sample.csv"),
        limit=100,
    )
