import json
from typing import Dict, List, Any, Union
from baselines.agentic.base_agent import BaseAgent
from baselines.agentic.litellm_model import LitellmModel
from baselines.agentic.action_prompt_template import generate_action_prompt


class Agent(BaseAgent):
    """LLM-based baseline agent that uses action prompt template for single-step inference."""

    def __init__(self, model: Union[str, LitellmModel] = "gpt-4o"):
        """Initialize agent with specified LLM model.

        Args:
            model: Either a string (for OpenAI models) or a LitellmModel instance
        """
        super().__init__(model=model)

    def run(
        self,
        memory: List[Dict[str, Any]],
        world_model: Dict[str, Any],
        persona: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run inference on a test sample using single comprehensive prompt.

        Args:
            memory: Combined list of true_positives and distractors documents
            world_model: World model containing available_actions and context
            persona: User persona information from test sample

        Returns:
            Dict containing:
            - retrieved_documents: List of document IDs from memory
            - bottleneck: Identified bottleneck string
            - action: Dict with function_name and parameters
        """

        # Generate comprehensive prompt using action prompt template
        prompt = generate_action_prompt(
            persona=json.dumps(persona, indent=2),
            world_model=json.dumps(world_model, indent=2),
            data_sources=json.dumps(memory, indent=2),
            available_actions=json.dumps(
                world_model.get("available_actions", []), indent=2
            ),
        )

        # Single LLM call with comprehensive prompt
        response = self.create_chat_completion(
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON response, stripping code block markers if present
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            print(f"Failed to parse response: {content}")
            return {
                "retrieved_documents": [],
                "bottleneck": f"Failed to parse JSON response: {str(e)}",
                "action": {},
            }

        return {
            "retrieved_documents": result.get("retrieved_documents", []),
            "bottleneck": result.get("bottleneck", ""),
            "action": result.get("action", {}),
        }
