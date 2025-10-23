"""
Base generator class for standardized interface across all data generators.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

logger = logging.getLogger(__name__)


class BaseGenerator(ABC):
    """Base class providing common functionality and standardized interface."""

    def __init__(
        self,
        template_dir: Optional[Path] = None,
        template_subdir: str = "",
        llm_generate_func: Optional[Callable[[str], str]] = None,
        debug_dir: Optional[Path] = None,
        max_workers: int = 8,
    ):
        """
        Initialize base generator with common functionality.

        Args:
            template_dir: Directory containing Jinja2 templates
            template_subdir: Subdirectory within prompts/ for this generator
            llm_generate_func: Function that takes a prompt and returns LLM response (from MultiLLMClient)
            debug_dir: Optional directory for debug output
            max_workers: Maximum number of parallel workers for batch processing
        """
        # Setup template directory
        if template_dir is None and template_subdir:
            template_dir = Path(__file__).parent.parent / "prompts" / template_subdir
        self.template_dir = template_dir

        # Setup Jinja2 environment
        if template_dir:
            self.jinja_env = Environment(
                loader=FileSystemLoader(str(template_dir)),
                autoescape=select_autoescape(["html", "xml"]),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        else:
            self.jinja_env = None

        # LLM functionality - MultiLLMClient handles retries and failover
        self.llm_generate_func = llm_generate_func or self._default_llm_func

        # Debug functionality
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self._debug_seq = 0

        # Parallelism configuration
        self.max_workers = max_workers

        logger.info(
            f"Initialized {self.__class__.__name__} with max_workers={max_workers}"
        )

    @abstractmethod
    def generate(self, **kwargs) -> Any:
        """
        Standard generation method that all generators must implement.

        Args:
            **kwargs: Generator-specific arguments

        Returns:
            Generated output (type varies by generator)
        """
        pass

    def _default_llm_func(self, prompt: str) -> str:
        """Default LLM function that raises an error if no function is provided."""
        raise NotImplementedError(
            "No LLM generation function provided. Pass llm_generate_func from MultiLLMClient to constructor."
        )

    def _next_debug_name(self, name: str, ext: str) -> str:
        """Return a monotonically increasing filename for debug artifacts."""
        if not self.debug_dir:
            raise ValueError("Debug directory not configured")

        self._debug_seq += 1
        return f"{self._debug_seq:03d}_{name}.{ext}"

    def _render_template(self, template_name: str, **context) -> str:
        """
        Render a Jinja2 template with the given context.

        Args:
            template_name: Name of the template file
            **context: Variables to pass to the template

        Returns:
            Rendered template string
        """
        if not self.jinja_env:
            raise ValueError("Jinja2 environment not initialized")

        template = self.jinja_env.get_template(template_name)
        return template.render(**context)

    def _parse_json_response(
        self, response: str, response_type: str = "response"
    ) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks and common formatting issues.

        Args:
            response: Raw LLM response string
            response_type: Type of response for error messages

        Returns:
            Parsed JSON as dictionary

        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            # First try direct parsing
            content = response.strip()
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            json_pattern = r"```(?:json)?\s*\n?([\s\S]*?)\n?```"
            json_matches = re.findall(json_pattern, content)

            if json_matches:
                for json_content in json_matches:
                    try:
                        return json.loads(json_content)
                    except json.JSONDecodeError:
                        continue

            # If no code blocks, try to find JSON-like content
            json_start = content.find("{")
            json_end = content.rfind("}")

            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    json_content = content[json_start : json_end + 1]
                    return json.loads(json_content)
                except json.JSONDecodeError:
                    pass

            # If all else fails, raise error with helpful message
            logger.error(
                f"Failed to parse {response_type} JSON from response: {response[:200]}..."
            )
            raise ValueError(f"Invalid JSON in {response_type} response")
