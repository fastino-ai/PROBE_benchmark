import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from baselines.agentic.base_agent import BaseAgent
from baselines.agentic.litellm_model import LitellmModel
from baselines.agentic.utils import create_datastore
from baselines.agentic.action_prompt_template import generate_action_prompt

# Configure logging with more detail
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Agent(BaseAgent):
    """Simple ReAct-style agent driven by the action prompt template."""

    def __init__(
        self,
        model: Union[str, LitellmModel] = "gpt-4.1",
        max_turns: int = 20,
        context_token_threshold: int = 100_000,
    ):
        super().__init__(model=model)
        self.max_turns = max_turns
        self.context_token_threshold = context_token_threshold
        self.observation_max_chars = 5000
        self.selected_action = None  # Track the final action selected
        self.retrieved_documents = set()  # Track documents retrieved
        self.identified_bottleneck = None  # Track the bottleneck description

    def run(
        self,
        memory: List[Dict[str, Any]],
        world_model: Dict[str, Any],
        persona: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        logger.info("=" * 60)
        logger.info("STARTING REACT AGENT RUN")
        logger.info("=" * 60)
        logger.info(f"Memory documents: {len(memory)}")
        logger.info(f"Persona: {persona.get('name', 'Unknown') if persona else 'None'}")

        # Reset tracking variables for new run
        self.selected_action = None
        self.retrieved_documents = set()
        self.identified_bottleneck = None

        datastore = create_datastore(memory, use_embeddings=False, mock_mode=True)
        logger.debug(f"Created datastore with {len(datastore.documents)} documents")

        persona_blob = self._format_persona(persona)
        world_model_blob = json.dumps(world_model, indent=2)
        actions_blob = json.dumps(world_model.get("available_actions", []), indent=2)
        data_sources_blob = self._format_data_sources(memory)

        # Build tool descriptions including available actions as tools
        tool_descriptions = self._build_tool_descriptions(
            world_model.get("available_actions", [])
        )
        system_prompt = generate_action_prompt(
            persona=persona_blob,
            tools=tool_descriptions,
            world_model=world_model_blob,
            data_sources=data_sources_blob,
            available_actions=actions_blob,
        )
        system_prompt += (
            "\n\nYou must reason and act using the ReAct pattern. On each turn output:\n"
            "Thought: <your reasoning>\n"
            "Action: <tool_name>\n"
            "Action Input: <tool arguments>\n"
            "\n"
            "First use sql_reader or semantic_search to find documents and identify the bottleneck.\n"
            "Once you've identified the bottleneck, select the appropriate action tool (e.g., check_in_raj, email_michael, etc.)\n"
            "and provide the bottleneck description and any required parameters as JSON input.\n"
            "\n"
            "Example final action:\n"
            "Thought: The bottleneck is missing client satisfaction reports from Raj Patel. I should check in with him.\n"
            "Action: check_in_raj\n"
            'Action Input: {"bottleneck": "Client satisfaction reports have not been received from Raj Patel since last quarter", "parameters": {"recipient_slack_id": "raj_patel", "message_body": "Hi Raj, checking on the client satisfaction reports...", "follow_up_time": "2 days"}}'
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "Begin the analysis."},
        ]
        logger.debug("System prompt prepared, starting ReAct loop")

        for turn in range(self.max_turns):
            logger.info(f"\n{'='*40}")
            logger.info(f"TURN {turn+1}/{self.max_turns}")
            logger.info(f"{'='*40}")

            response = self.create_chat_completion(messages=messages)
            content = response.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": content})

            # Log the thought/action/input
            logger.debug(f"Assistant response (first 500 chars):\n{content[:500]}...")

            action, action_input, final_payload = self._parse_react_message(content)

            logger.info(
                f"Parsed - Action: {action}, Has Final Answer: {final_payload is not None}"
            )
            if action_input:
                logger.debug(
                    f"Action Input (first 200 chars): {str(action_input)[:200]}"
                )

            if final_payload is not None:
                logger.info("Final answer provided, ending ReAct loop")
                logger.debug(
                    f"Final payload: {final_payload[:200] if isinstance(final_payload, str) else str(final_payload)[:200]}"
                )
                return self._prepare_final_output(final_payload)

            if action is None:
                logger.warning("No action detected at turn %s", turn)
                break

            logger.info(f"Executing tool: {action}")
            observation = self._execute_tool(
                action, action_input, datastore, world_model
            )
            logger.debug(f"Observation (first 300 chars): {observation[:300]}...")
            messages.append({"role": "user", "content": f"Observation: {observation}"})

            # Check if we have selected a final action
            if self.selected_action:
                logger.info("Final action has been selected, returning result")
                return {
                    "retrieved_documents": list(self.retrieved_documents),
                    "bottleneck": self.identified_bottleneck
                    or "Unable to provide detailed bottleneck description",
                    "action": self.selected_action,
                }

            if self._estimate_tokens(messages) > self.context_token_threshold:
                messages = self._summarize_progress(messages)

        logger.warning(
            "Max turns reached without final answer. Returning fallback output."
        )
        fallback = self._fallback_result(datastore, world_model)
        return fallback

    def _build_tool_descriptions(
        self, available_actions: List[Dict[str, Any]] = None
    ) -> str:
        tools = [
            (
                "sql_reader",
                "Execute read-only SQL (SELECT) queries against the in-memory "
                "documents table. Schema: id (TEXT PRIMARY KEY), type (TEXT), "
                "payload (JSON object). Accepts either a raw SQL string or JSON with `query` and "
                "optional `offset` for paginating long responses. Returns rows with payload "
                "automatically parsed from JSON. If the observation is truncated, rerun "
                "with the provided `next_offset` to continue.",
            ),
            (
                "semantic_search",
                "Perform semantic or keyword search over documents. Input must be a JSON "
                "object with `query` (string), optional `top_k` (int, default 5), and optional "
                "`offset` for paginating long responses. Returns matching document ids with "
                "brief content excerpts and indicates when truncation occurs.",
            ),
        ]

        # Add available actions as tools
        if available_actions:
            for action in available_actions:
                action_id = action.get("id", "")
                action_desc = action.get("description", "")
                # Create tool description for the action
                tools.append(
                    (
                        action_id,
                        f"{action_desc} When you select this action, provide the bottleneck description and required parameters as JSON.",
                    )
                )

        return "\n".join(f"- {name}: {desc}" for name, desc in tools)

    def _format_persona(self, persona: Optional[Dict[str, Any]]) -> str:
        if not persona:
            return "No additional persona details provided."
        return json.dumps(persona, indent=2)

    def _format_data_sources(self, documents: List[Dict[str, Any]]) -> str:
        lines = []
        for doc in documents:
            doc_id = doc.get("id", "unknown_document")
            doc_type = doc.get("type", "unknown_type")
            metadata = doc.get("metadata", {}) or {}
            doc_type_meta = metadata.get("doc_type")
            lines.append(f"- {doc_id} ({doc_type_meta or doc_type})")
        return "\n".join(lines) if lines else "No documents available."

    def _parse_react_message(
        self, content: str
    ) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        # Check if this message contains a Final Answer
        if "Final Answer:" in content:
            final_section = content.split("Final Answer:", maxsplit=1)[1]
            return None, None, final_section.strip()

        # Check for action
        action_match = re.search(r"Action:\s*(\w+)", content)
        if not action_match:
            return None, None, None
        action = action_match.group(1).strip()

        # Special case: if action is "finish", look for Final Answer instead of Action Input
        if action.lower() == "finish":
            if "Final Answer:" in content:
                final_section = content.split("Final Answer:", maxsplit=1)[1]
                return None, None, final_section.strip()
            else:
                # Finish without Final Answer - agent needs to provide it
                return action, "", None

        # For other actions, look for Action Input
        input_match = re.search(r"Action Input:\s*(.*)", content, re.DOTALL)
        if not input_match:
            return action, "", None

        remainder = input_match.group(1)
        input_lines = []
        for line in remainder.splitlines():
            if re.match(r"^(Thought|Observation|Final Answer):", line.strip()):
                break
            input_lines.append(line)
        action_input = "\n".join(input_lines).strip()
        return action, action_input, None

    def _execute_tool(
        self, action: str, action_input: str, datastore, world_model: Dict[str, Any]
    ) -> str:
        if action == "sql_reader":
            query, offset = self._parse_sql_input(action_input)
            logger.info(f"SQL Query: {query}")
            if not query:
                return "sql_reader requires a SELECT query."
            if not query.lower().startswith("select"):
                return "sql_reader only supports SELECT statements."
            rows = datastore.sql_query(query)
            logger.info(f"SQL Results: {len(rows)} rows returned")
            # Track retrieved documents
            for row in rows:
                if "id" in row:
                    self.retrieved_documents.add(row["id"])
            serialized = json.dumps(rows, indent=2)
            return self._paginate_observation(serialized, offset)

        if action == "semantic_search":
            try:
                payload = json.loads(action_input) if action_input else {}
            except json.JSONDecodeError:
                payload = {"query": action_input}
            query = payload.get("query", "")
            top_k = int(payload.get("top_k", 5)) if payload.get("top_k") else 5
            offset = (
                int(payload.get("offset", 0))
                if payload.get("offset") is not None
                else 0
            )
            logger.info(f"Semantic Search Query: '{query}', top_k={top_k}")
            doc_ids = datastore.semantic_search(query, top_k=top_k)
            logger.info(f"Semantic Search Results: {len(doc_ids)} documents found")
            logger.debug(f"Document IDs: {doc_ids}")
            # Track retrieved documents
            for doc_id in doc_ids:
                self.retrieved_documents.add(doc_id)
            results = []
            for doc_id in doc_ids:
                content = datastore.get_document_content(doc_id)
                results.append({"id": doc_id, "excerpt": content[:300]})
            serialized = json.dumps(results, indent=2)
            return self._paginate_observation(serialized, offset)

        if action == "finish":
            return (
                "You've selected Action: finish. Now provide your Final Answer with the complete JSON "
                "containing retrieved_documents, bottleneck description, and your selected action from "
                "the Available Actions list. Format:\n"
                'Final Answer: {"retrieved_documents": [...], "bottleneck": "...", "action": {...}}'
            )

        # Check if this is one of the available actions
        available_actions = world_model.get("available_actions", [])
        action_ids = [a.get("id") for a in available_actions]

        if action in action_ids:
            # This is a final action selection!
            logger.info(f"Final action selected: {action}")

            # Parse the input to get bottleneck description and parameters
            try:
                if action_input:
                    input_data = (
                        json.loads(action_input)
                        if action_input.startswith("{")
                        else {"description": action_input}
                    )
                else:
                    input_data = {}
            except json.JSONDecodeError:
                input_data = {"description": action_input}

            # Store the selected action
            self.selected_action = {
                "function_name": action,
                "parameters": input_data.get("parameters", {}),
            }

            # Store the bottleneck description if provided
            if "bottleneck" in input_data:
                self.identified_bottleneck = input_data["bottleneck"]
            elif "description" in input_data:
                self.identified_bottleneck = input_data["description"]

            # Build the final result
            result = {
                "retrieved_documents": list(self.retrieved_documents),
                "bottleneck": self.identified_bottleneck or f"Issue requiring {action}",
                "action": self.selected_action,
            }

            # Return a success message and trigger the end of the agent loop
            return f"✅ Bottleneck identified and action '{action}' selected. Final result: {json.dumps(result, indent=2)}"

        return f"Unknown action '{action}'."

    def _parse_sql_input(self, action_input: str) -> Tuple[str, int]:
        action_input = action_input.strip()
        if not action_input:
            return "", 0

        if action_input.startswith("{"):
            try:
                payload = json.loads(action_input)
                query = payload.get("query", "")
                offset = (
                    int(payload.get("offset", 0))
                    if payload.get("offset") is not None
                    else 0
                )
                return query, max(0, offset)
            except json.JSONDecodeError:
                pass

        # Remove surrounding quotes if present (agent sometimes wraps SQL in quotes)
        if action_input.startswith('"') and action_input.endswith('"'):
            action_input = action_input[1:-1]
            # Unescape any escaped quotes inside
            action_input = action_input.replace('\\"', '"').replace("\\'", "'")

        return action_input, 0

    def _paginate_observation(self, serialized: str, offset: int) -> str:
        if offset < 0:
            offset = 0

        total_length = len(serialized)
        if offset >= total_length:
            payload = {
                "offset": offset,
                "chunk": "",
                "truncated": False,
                "note": "Offset beyond result length; nothing more to show.",
            }
            return json.dumps(payload, indent=2)

        end = min(total_length, offset + self.observation_max_chars)
        chunk = serialized[offset:end]
        truncated = end < total_length
        payload = {"offset": offset, "chunk": chunk, "truncated": truncated}
        if truncated:
            payload["next_offset"] = end
            payload["note"] = (
                "Result truncated. Re-run the tool with offset="
                f"{end} to continue from the next character."
            )
        return json.dumps(payload, indent=2)

    def _prepare_final_output(self, payload: str) -> Dict[str, Any]:
        logger.info("Preparing final output from agent's answer")
        payload = payload.strip().strip("`")
        if payload.startswith("json"):
            payload = payload[4:].strip()
        logger.debug(f"Cleaned payload (first 300 chars): {payload[:300]}...")
        try:
            result = json.loads(payload)
            logger.info(f"Successfully parsed JSON result")
            if "retrieved_documents" in result:
                logger.info(f"Retrieved documents: {result['retrieved_documents']}")
            if "bottleneck" in result:
                logger.info(
                    f"Identified bottleneck: {result['bottleneck'][:100] if result['bottleneck'] else 'None'}..."
                )
            if "action" in result:
                logger.info(
                    f"Selected action: {result['action'].get('function_name', 'unknown')}"
                )
            return result
        except json.JSONDecodeError:
            logger.error("Failed to parse final payload: %s", payload[:500])
            return {
                "retrieved_documents": [],
                "bottleneck": payload,
                "action": {"function_name": "", "parameters": {}},
            }

    def _fallback_result(
        self, datastore, world_model: Dict[str, Any]
    ) -> Dict[str, Any]:
        logger.warning(
            "FALLBACK: Agent failed to find bottleneck, using fallback result"
        )

        # Use tracked documents if we have them, otherwise do a semantic search
        if self.retrieved_documents:
            docs = list(self.retrieved_documents)
            logger.info(f"Using retrieved documents: {docs}")
        else:
            docs = datastore.semantic_search("deadline feedback", top_k=2)
            logger.info(
                f"Fallback semantic search for 'deadline feedback' returned: {docs}"
            )

        # Use selected action if we have one, otherwise use first available action
        if self.selected_action:
            action = self.selected_action
            logger.info(f"Using previously selected action: {action['function_name']}")
        else:
            available_actions = world_model.get("available_actions", [])
            if available_actions and len(available_actions) > 0:
                fallback_action = available_actions[0].get("id", "")
            else:
                fallback_action = ""
                logger.warning("No available actions found in world model")
            action = {"function_name": fallback_action, "parameters": {}}
            logger.info(f"Fallback action selected: {fallback_action}")

        # Use identified bottleneck if we have one
        bottleneck = (
            self.identified_bottleneck
            or "Unable to derive bottleneck within allotted turns."
        )

        return {"retrieved_documents": docs, "bottleneck": bottleneck, "action": action}

    def _estimate_tokens(self, messages: List[Dict[str, str]]) -> int:
        total_chars = sum(len(m.get("content", "")) for m in messages)
        return total_chars // 4

    def _summarize_progress(
        self, messages: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        history_text = self._messages_to_text(messages)
        summary_prompt = (
            "Create a concise summary of the Thoughts, Actions, and Observations so far so the agent "
            "can continue working without important omissions."
        )
        summary_response = self.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You summarize agent trajectories succinctly.",
                },
                {"role": "user", "content": f"{summary_prompt}\n\n{history_text}"},
            ]
        )
        summary = summary_response.choices[0].message.content or ""
        new_messages = [
            messages[0],
            {"role": "system", "content": f"Summary so far:\n{summary}"},
        ]
        if len(messages) >= 2:
            new_messages.extend(messages[-2:])
        return new_messages

    def _messages_to_text(self, messages: List[Dict[str, str]]) -> str:
        rows = []
        for msg in messages[1:]:
            rows.append(f"{msg['role'].upper()}: {msg.get('content', '')}")
        return "\n\n".join(rows)

    def run_from_file(self, sample_path: str) -> Dict[str, Any]:
        with open(sample_path, "r", encoding="utf-8") as handle:
            sample = json.load(handle)
        persona = sample.get("persona")
        world_model = sample.get("world_model", {})
        memory = sample.get("true_positives", []) + sample.get("distractors", [])
        return self.run(memory=memory, world_model=world_model, persona=persona)
