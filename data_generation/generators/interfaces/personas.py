from typing import Dict, Any, List, Callable
from pydantic import BaseModel, Field, ConfigDict, field_validator


class Persona(BaseModel):
    """Simplified persona model for checklist-based evaluation.

    This Pydantic model is intentionally minimal to make it easy for an agent
    to instantiate from a natural language description. Use ``other_info`` for
    additional context needed by scenarios/evaluations (e.g., timezone,
    work_style, communication_preferences, constraints). The ``other_info``
    dictionary is intentionally flexible: both keys and values may be
    natural-language phrases. This allows agents to capture ad-hoc attributes
    that help generate varied personas (e.g., "prefers async updates",
    "frustrations": "frequent context switching").
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Full name of the persona, e.g., 'Alex Chen'")
    occupation: str = Field(
        ...,
        description="Primary role and organization/industry, e.g., 'Product Manager at TechCorp'",
    )
    goals: List[str] = Field(
        default_factory=list,
        description="Top goals/priorities as concise, actionable phrases",
    )
    other_info: Dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional context such as timezone, work_style, communication preferences,"
            " role_seniority, constraints, or domain-specific details. Keys and values"
            " may be natural-language phrases; both structured and free-form attributes"
            " are allowed so agents can express varied persona traits."
        ),
    )

    @field_validator("name", "occupation")
    @classmethod
    def _strip_and_require_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty")
        return value

    @field_validator("goals")
    @classmethod
    def _normalize_goals(cls, value: List[str]) -> List[str]:
        normalized: List[str] = [
            g.strip() for g in value if isinstance(g, str) and g.strip()
        ]
        if not normalized:
            raise ValueError("goals must include at least one non-empty item")
        return normalized

    def to_prompt_context(self) -> str:
        """Render the persona as a compact prompt context string."""
        lines: List[str] = [
            f"Name: {self.name}",
            f"Occupation: {self.occupation}",
            "Goals:",
        ]
        lines.extend([f"- {g}" for g in self.goals])
        if self.other_info:
            lines.append("Other Info:")
            for key, val in self.other_info.items():
                lines.append(f"- {key}: {val}")
        return "\n".join(lines)

    @classmethod
    def build_extraction_prompt(cls, natural_description: str) -> str:
        """Create a schema-guided prompt for extracting a Persona from text.

        Builds field guidance dynamically from the current Pydantic JSON schema so
        prompts stay aligned with the latest class definition.
        """
        schema = cls.model_json_schema()
        props: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        schema_lines: List[str] = []
        for field_name, meta in props.items():
            field_type = meta.get("type", "object")
            desc = meta.get("description", "")
            req_tag = " (required)" if field_name in required else " (optional)"
            detail = f" – {desc}" if desc else ""
            schema_lines.append(f"- {field_name}: {field_type}{req_tag}{detail}")

        schema_block = "\n".join(schema_lines)

        example_json = (
            "{\n"
            '  "name": "Alex Chen",\n'
            '  "occupation": "Product Manager at TechCorp",\n'
            '  "goals": ["Launch Q3 product", "Hire senior engineer"],\n'
            '  "other_info": {\n'
            '    "timezone": "America/Los_Angeles",\n'
            '    "work_style": "proactive",\n'
            '    "frustrations": "frequent context switching"\n'
            "  }\n"
            "}"
        )

        instructions = (
            "Extract a Persona from the following description.\n"
            "Return ONLY a valid JSON object that matches this schema (use exact keys):\n"
            f"{schema_block}\n"
            "Rules:\n"
            "- Do not invent facts absent from the description.\n"
            "- If strongly implied, include sensible details in other_info.\n"
            "- Keep goals as concise, actionable phrases.\n"
            "- Include timezone/work_style if available, and communication preferences if mentioned.\n"
            "- other_info keys and values may be natural-language phrases; prefer short, descriptive keys.\n"
            "Description:\n"
            f"{natural_description}\n"
            "Output JSON (no extra text):\n"
            f"{example_json}"
        )
        return instructions

    @classmethod
    def from_natural_language(
        cls,
        description: str,
        generate_json: Callable[[str], str],
    ) -> "Persona":
        """Create a Persona by calling a text-to-JSON function with an extraction prompt.

        Args:
            description: Natural language description of the persona.
            generate_json: Callable that accepts a prompt and returns JSON string.

        Returns:
            Persona instance parsed from the returned JSON.
        """
        prompt = cls.build_extraction_prompt(description)
        raw = generate_json(prompt)
        import json as _json

        data = _json.loads(raw)
        return cls(**data)

    @classmethod
    def build_extraction_messages(
        cls,
        natural_description: str,
        include_examples: bool = True,
    ) -> List[Dict[str, str]]:
        """Build chat messages for few-shot extraction of a Persona.

        Returns a list of messages suitable for chat.completions.create.
        Includes a system message with schema and rules, optional few-shot
        examples, and a final user message with the provided description.
        """
        schema = cls.model_json_schema()
        props: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        schema_lines: List[str] = []
        for field_name, meta in props.items():
            field_type = meta.get("type", "object")
            desc = meta.get("description", "")
            req_tag = " (required)" if field_name in required else " (optional)"
            detail = f" – {desc}" if desc else ""
            schema_lines.append(f"- {field_name}: {field_type}{req_tag}{detail}")

        schema_block = "\n".join(schema_lines)

        system_msg = (
            "You are a structured extraction assistant. Return ONLY a valid JSON object. "
            "No code fences, no extra text. Follow this schema (use exact keys):\n"
            f"{schema_block}\n"
            "Rules:\n"
            "- Do not invent facts absent from the description.\n"
            "- If strongly implied, include sensible details in other_info.\n"
            "- Keep goals as concise, actionable phrases.\n"
            "- other_info keys and values may be natural-language phrases; prefer short, descriptive keys.\n"
        )

        messages: List[Dict[str, str]] = [{"role": "system", "content": system_msg}]

        if include_examples:
            import json as _json

            for desc, ex in _few_shot_examples():
                messages.append({"role": "user", "content": f"Description:\n{desc}"})
                messages.append(
                    {"role": "assistant", "content": _json.dumps(ex.model_dump())}
                )

        messages.append(
            {"role": "user", "content": f"Description:\n{natural_description}"}
        )
        return messages


"""Example personas using the simplified Persona model."""

# Example 1: Executive persona (minimal but useful for generation)
example_ceo_sarah = Persona(
    name="Sarah Chen",
    occupation="Chief Executive Officer at TechCorp",
    goals=[
        "Oversee Q3 product launch",
        "Maintain board readiness",
    ],
    other_info={
        "timezone": "America/Los_Angeles",
        "work_style": "decisive, prefers executive summaries",
        "communication_preferences": "brief emails, fast decisions",
    },
)

# Example 2: Product manager persona (concise, varied other_info)
example_pm_alex = Persona(
    name="Alex Rodriguez",
    occupation="Senior Product Manager at TechCorp",
    goals=[
        "Ship Q3 features on time",
        "Hire a staff engineer",
    ],
    other_info={
        "timezone": "America/Los_Angeles",
        "work_style": "collaborative, proactive",
        "communication_preferences": "prefers async updates, detailed requirements",
        "frustrations": "frequent context switching",
    },
)


def _few_shot_examples() -> List:
    """Return few-shot pairs of (natural_description, Persona instance)."""
    return [
        (
            "CEO at TechCorp in PST; oversees Q3 product launch and board readiness; "
            "prefers brief emails and executive summaries; decisive.",
            example_ceo_sarah,
        ),
        (
            "Senior PM at TechCorp; ships Q3 features; hiring a staff engineer; "
            "collaborative and proactive; prefers async updates and detailed requirements; PST.",
            example_pm_alex,
        ),
    ]
