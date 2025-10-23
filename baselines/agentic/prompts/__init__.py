"""
Prompt template utilities for agentic baselines.

This module provides utilities to load and render Jinja2 prompt templates.
"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from typing import Dict, Any

# Get the prompts directory
PROMPTS_DIR = Path(__file__).parent

# Create Jinja2 environment
_env = Environment(
    loader=FileSystemLoader(PROMPTS_DIR),
    autoescape=select_autoescape(),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_template(template_name: str, **context: Any) -> str:
    """
    Render a Jinja2 template with the given context.

    Args:
        template_name: Name of the template file (e.g., "action_prompt.j2")
        **context: Template variables to render

    Returns:
        Rendered template as string
    """
    template = _env.get_template(template_name)
    return template.render(**context)


def generate_action_prompt(
    persona: str,
    world_model: str,
    data_sources: str,
    available_actions: str,
    tools: str = None,
) -> str:
    """Generate action prompt using template."""
    return render_template(
        "action_prompt.j2",
        persona=persona,
        world_model=world_model,
        data_sources=data_sources,
        available_actions=available_actions,
        tools=tools or "",
    )


def generate_llm_prompt(
    persona: str,
    world_model: str,
    data_sources: str,
    available_actions: str,
) -> str:
    """Generate LLM prompt using template."""
    return render_template(
        "llm_prompt.j2",
        persona=persona,
        world_model=world_model,
        data_sources=data_sources,
        available_actions=available_actions,
    )
