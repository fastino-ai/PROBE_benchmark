import re
import json
import tiktoken
from typing import Dict, List, Any, Union
from baselines.agentic.base_agent import BaseAgent
from baselines.agentic.litellm_model import LitellmModel
from baselines.agentic.utils import create_datastore
from baselines.agentic.prompts import render_template


class Agent(BaseAgent):
    """Reflexion agent implementing verbal reinforcement learning for productivity tasks."""

    def __init__(
        self,
        model: Union[str, LitellmModel] = "gpt-4o",
        max_trials: int = 2,
        max_steps: int = 4,
    ):
        """Initialize reflexion agent.

        Args:
            model: Either a string (for OpenAI models) or a LitellmModel instance
            max_trials: Maximum number of trials (original: 10)
            max_steps: Maximum steps per trial (original: 6)
        """
        super().__init__(model=model)
        self.max_trials = max_trials
        self.max_steps = max_steps
        self.reflections: List[str] = []
        self.reflections_str = ""
        self.trial_count = 0

        # Initialize tokenizer for context window monitoring
        try:
            self.enc = tiktoken.encoding_for_model("gpt-4")
        except KeyError:
            # Fallback to cl100k_base encoding if model not found
            self.enc = tiktoken.get_encoding("cl100k_base")

        # Workflow analysis state
        self.scratchpad: str = ""
        self.step_n: int = 0
        self.finished = False
        self.retrieved_documents: List[str] = []
        self.bottleneck: str = ""
        self.selected_action: Dict[str, Any] = {}
        self.datastore = None
        self.world_model = {}

        # Workflow state tracking (ensure sequential execution)
        self.has_retrieved = False
        self.has_analyzed = False
        self.last_analyzed_docs: List[str] = []

        # Early stopping for repeated errors
        self.consecutive_invalid_actions = 0
        self.max_consecutive_invalid = 2

        # Store full analysis data separately from scratchpad
        self._full_analysis_data = []

    def run(
        self,
        memory: List[Dict[str, Any]],
        world_model: Dict[str, Any],
        persona: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Run reflexion loop until success or max trials reached."""
        self.datastore = create_datastore(memory)
        self.world_model = world_model

        for trial in range(self.max_trials):
            self.trial_count = trial

            # Run workflow analysis with reflection if not first trial
            if trial > 0 and not self._is_correct():
                self._reflect()

            # Reset and run analysis
            self._reset()
            self._run_workflow_analysis()

            # Evaluate result
            result = self._get_current_result()
            reward = self._evaluator_step(result, world_model)

            # If successful, return result
            if reward >= 0.8:  # Success threshold
                return result

            # Store reflection for next trial if not successful
            if trial < self.max_trials - 1:  # Don't reflect on last trial
                reflection = self._reflection_step(result, reward, world_model)
                self.reflections.append(reflection)

        # Return best attempt if max trials reached
        final_result = self._get_current_result()

        # Try to ensure we have an action - use fallback if needed
        if not final_result.get("action") or not final_result["action"].get(
            "function_name"
        ):
            print(
                f"[WARNING] No action after {self.max_trials} trials. State: finished={self.finished}, "
                f"has_retrieved={self.has_retrieved}, has_analyzed={self.has_analyzed}"
            )
            self._select_fallback_action()
            final_result = self._get_current_result()

        # If still no action, create a minimal valid one
        if not final_result.get("action") or not final_result["action"].get(
            "function_name"
        ):
            print("[WARNING] Creating minimal fallback action")
            available_actions = self.world_model.get("available_actions", [])
            if available_actions:
                final_result["action"] = {
                    "function_name": available_actions[0].get("id", ""),
                    "parameters": {},
                }
            else:
                final_result["action"] = {"function_name": "", "parameters": {}}

        return final_result

    def _run_workflow_analysis(self):
        """Run the workflow analysis following Reflexion's step-by-step pattern."""
        while not self._is_finished() and not self._is_halted():
            # Check step limit before executing step to ensure we don't exceed max_steps
            if self.step_n >= self.max_steps:
                print(
                    f"[DEBUG] Reached max steps limit: {self.step_n}/{self.max_steps}"
                )
                break
            self._step()

        # Handle incomplete workflow gracefully instead of crashing
        if not self._is_finished():
            halt_reason = self._get_halt_reason()
            print(
                f"[WARNING] Workflow halted without completion. Reason: {halt_reason}"
            )
            print(
                f"[WARNING] State: finished={self.finished}, has_retrieved={self.has_retrieved}, "
                f"has_analyzed={self.has_analyzed}, selected_action={bool(self.selected_action)}"
            )
            print(
                f"[WARNING] Steps: {self.step_n}/{self.max_steps}, consecutive_errors={self.consecutive_invalid_actions}"
            )

            # Try to complete with fallback action if we have partial progress
            if self.has_retrieved and self.has_analyzed and not self.selected_action:
                print("[WARNING] Attempting to select fallback action...")
                self._select_fallback_action()
            elif not self.has_analyzed and self.has_retrieved:
                print("[WARNING] Attempting to analyze retrieved documents...")
                # Try to analyze the retrieved documents
                if self.retrieved_documents:
                    try:
                        doc_ids_str = ",".join(self.retrieved_documents[:3])
                        observation = self._execute_analyze(doc_ids_str)
                        self.scratchpad += f"\n> Analyze[{doc_ids_str}]\n{observation}"
                    except Exception as e:
                        print(f"[WARNING] Failed to analyze documents: {e}")
                # Then try to select fallback action
                self._select_fallback_action()
            elif not self.has_retrieved:
                print("[WARNING] No documents retrieved, using minimal fallback...")
                # Do a quick retrieval
                try:
                    observation = self._execute_retrieve("bottleneck issue deadline")
                    self.scratchpad += (
                        f"\n> Retrieve[bottleneck issue deadline]\n{observation}"
                    )
                    self.has_retrieved = True
                except Exception as e:
                    print(f"[WARNING] Failed to retrieve documents: {e}")
                # Select fallback action
                self._select_fallback_action()

    def _step(self):
        """Execute one step of workflow analysis following canonical reflexion pattern."""
        print(
            f"[DEBUG] Step {self.step_n}: has_retrieved={self.has_retrieved}, has_analyzed={self.has_analyzed}, selected_action={bool(self.selected_action)}"
        )

        # Normal operation - let LLM follow the workflow naturally
        print(f"[DEBUG] Normal LLM operation (step {self.step_n})")
        response = self._prompt_agent().strip()
        action = self._extract_first_action(response)
        action_type, argument = self._parse_action(action)

        # Add the action to scratchpad
        self.scratchpad += f"\n> {action}"
        print(f"> {action}")

        # Execute action and get observation (like canonical)
        if action_type == "Retrieve":
            observation = self._execute_retrieve(argument)
            self.consecutive_invalid_actions = 0  # Reset on valid action
        elif action_type == "Analyze":
            observation = self._execute_analyze(argument)
            self.consecutive_invalid_actions = 0  # Reset on valid action
        elif action_type == "SelectAction":
            observation = self._execute_select_action(argument)
            self.consecutive_invalid_actions = 0  # Reset on valid action
            if "successfully" in observation.lower():
                self.finished = True
        else:
            observation = "Invalid action type. Valid actions are: Retrieve[query], Analyze[document_ids], SelectAction[action_details]"
            self.consecutive_invalid_actions += 1

        # Add observation to scratchpad
        self.scratchpad += f"\n{observation}"
        print(observation)

        # Truncate scratchpad if it gets too long (sliding window like original Reflexion)
        self._truncate_scratchpad_if_needed()

        self.step_n += 1

    def _prompt_agent(self) -> str:
        """Generate agent response using current scratchpad (canonical pattern)."""
        persona = self.world_model.get("persona", {})

        # Simple formatting like baseline agent
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )
        world_model_str = json.dumps(self.world_model, indent=2)

        # Get step-specific instructions and examples
        step_instructions = self._get_step_instructions()
        step_example = self._get_step_example()

        prompt = render_template(
            "workflow_analysis.j2",
            reflections=self.reflections_str,
            persona=persona_str,
            world_model=world_model_str,
            step_instructions=step_instructions,
            step_example=step_example,
            scratchpad=self.scratchpad,
        )

        # Check if model supports stop parameter (GPT-5 models don't)
        model_name = getattr(self, "model", "")
        supports_stop = not (
            isinstance(model_name, str)
            and ("gpt-5" in model_name.lower() or "o1" in model_name.lower())
        )

        if self.supports_temperature():
            if supports_stop:
                response = self.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    stop=["\n"],
                )
            else:
                response = self.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}], temperature=0.1
                )
        else:
            if supports_stop:
                response = self.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}], stop=["\n"]
                )
            else:
                response = self.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}]
                )

        return self._format_step(response.choices[0].message.content)

    def _extract_first_action(self, response: str) -> str:
        """Extract the first action from a potentially multi-action response."""
        import re

        # Special handling for SelectAction since it contains JSON that can span multiple lines
        if "SelectAction[" in response:
            # Extract SelectAction with its complete JSON content
            start_idx = response.find("SelectAction[")
            if start_idx != -1:
                json_start = start_idx + len("SelectAction[")
                bracket_count = 0
                in_json = False

                for i, char in enumerate(response[json_start:]):
                    if char == "{":
                        bracket_count += 1
                        in_json = True
                    elif char == "}":
                        bracket_count -= 1
                        if in_json and bracket_count == 0:
                            # Found the end of JSON, now look for the closing ]
                            remaining = response[json_start + i + 1 :]
                            closing_bracket = remaining.find("]")
                            if closing_bracket != -1:
                                end_idx = json_start + i + 1 + closing_bracket + 1
                                return response[start_idx:end_idx]

                # If we couldn't find proper closing, try to complete the JSON
                incomplete_action = response[start_idx:]
                if incomplete_action.count("{") > incomplete_action.count("}"):
                    return incomplete_action.rstrip() + "}]}"
                return incomplete_action

        # For other actions, use simpler pattern matching
        action_pattern = r"((?:Retrieve|Analyze)\[[^\]]+\])"
        actions = re.findall(action_pattern, response)

        if actions:
            return actions[0]

        # Fallback: extract first part before any observation
        parts = re.split(r"(?:Found documents:|Document|Action selected)", response, 1)
        first_part = parts[0].strip()
        first_part = re.sub(r"^>+\s*", "", first_part).strip()

        return first_part if first_part else response

    def _get_step_instructions(self) -> str:
        """Get instructions for current step only."""
        if not self.has_retrieved:
            return "STEP 1: Retrieve[query] - Retrieve relevant documents using semantic search"
        elif not self.has_analyzed:
            return "STEP 1: Retrieve[query] - Retrieve relevant documents using semantic search\nSTEP 2: Analyze[document_ids] - Analyze retrieved documents to identify a bottleneck"
        else:
            return "STEP 1: Retrieve[query] - Retrieve relevant documents using semantic search\nSTEP 2: Analyze[document_ids] - Analyze retrieved documents to identify a bottleneck\nSTEP 3: SelectAction[action_details] - Select and configure an action from the available_actions to address the bottleneck"

    def _get_step_example(self) -> str:
        """Get example for current step only."""
        if not self.has_retrieved:
            return "> Retrieve[urgent meetings conflicts deadlines]"
        elif not self.has_analyzed:
            return f"> Retrieve[urgent meetings conflicts deadlines]\n> Analyze[{','.join(self.retrieved_documents)}]"
        else:
            return '> Retrieve[urgent meetings conflicts deadlines]\n> Analyze[{\',\'.join(self.retrieved_documents)}]\n> SelectAction[{"function_name": "action_id", "parameters": {...}}]'

    def _execute_retrieve(self, query: str) -> str:
        """Execute document retrieval action."""
        retrieved_docs = self.datastore.semantic_search(query, 3)
        # Only add new documents to avoid duplicates
        for doc in retrieved_docs:
            if doc not in self.retrieved_documents:
                self.retrieved_documents.append(doc)
        self.has_retrieved = True
        return f"Found documents: {retrieved_docs}"

    def _execute_analyze(self, document_ids: str) -> str:
        """Execute document analysis action."""
        doc_ids = [doc.strip() for doc in document_ids.split(",")]

        # Get documents and analyze them for bottlenecks
        documents_text = []
        for doc_id in doc_ids:
            doc = self.datastore.get_document(doc_id)
            if doc:
                doc_type = doc.get("type", "document")
                payload = doc.get("payload", {})
                content = payload.get("content", "") or payload.get("body", "")
                title = (
                    payload.get("title", "") or payload.get("subject", "") or "Untitled"
                )
                documents_text.append(
                    f"Document {doc_id} ({doc_type}): {title}\n{content}"
                )

        # Use LLM to analyze documents and identify bottleneck
        analysis_prompt = f"""Analyze these documents to identify the main bottleneck or issue requiring intervention:

{chr(10).join(documents_text)}

What specific bottleneck do you identify? Be concise."""

        if self.supports_temperature():
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}], temperature=0.1
            )
        else:
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": analysis_prompt}]
            )

        bottleneck_analysis = response.choices[0].message.content.strip()

        # Track state
        self.last_analyzed_docs = doc_ids.copy()
        self.has_analyzed = True
        self.bottleneck = bottleneck_analysis

        return bottleneck_analysis

        # Extract bottleneck from analysis - always update if we have analysis
        if analysis:
            # Try to extract a more meaningful bottleneck description
            if "bottleneck" in analysis.lower():
                # Look for explicit bottleneck mentions
                bottleneck_text = analysis
            else:
                # Create bottleneck description from analysis
                bottleneck_text = f"Analysis reveals: {analysis}..."
            # Always update bottleneck with latest analysis
            self.bottleneck = bottleneck_text.strip()
            print(f"[DEBUG] Set bottleneck to: {self.bottleneck}")
        else:
            print(f"[DEBUG] No analysis content to extract bottleneck from")

        return analysis

    def _select_fallback_action(self):
        """Select a fallback action when workflow doesn't complete normally."""
        available_actions = self.world_model.get("available_actions", [])

        if not available_actions:
            print("[WARNING] No available actions to select from")
            return

        # Use first available action as fallback
        fallback_action_id = available_actions[0].get("id", "")

        print(f"[WARNING] Selecting fallback action: {fallback_action_id}")

        self.selected_action = {"function_name": fallback_action_id, "parameters": {}}

        # Set a generic bottleneck if we don't have one
        if not self.bottleneck:
            self.bottleneck = "Unable to complete full analysis within step limit. Selecting default action based on available information."

    def _execute_select_action(self, action_details: str) -> str:
        """Execute action selection."""
        # Parse JSON response, stripping code block markers if present
        content = action_details.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        print(f"[DEBUG] Parsing action content: {content}")

        # Additional JSON cleaning for common LLM formatting issues
        if content.startswith("{  {"):
            # Remove extra opening brace: "{  {" -> "{"
            content = content[3:]
        elif content.startswith("{ {"):
            # Remove extra opening brace: "{ {" -> "{"
            content = content[2:]

        try:
            action_data = json.loads(content)
        except json.JSONDecodeError as e:
            # Try to fix common JSON issues
            try:
                # Try removing trailing extra braces
                if content.endswith("}}"):
                    content = content[:-1]
                    action_data = json.loads(content)
                else:
                    raise e
            except json.JSONDecodeError as e2:
                error_msg = f"SELECTACTION JSON PARSING FAILED: {e2}\n"
                error_msg += f"Raw action_details: {action_details}\n"
                error_msg += f"Cleaned content: {content}"
                raise RuntimeError(error_msg)

        # Validate action exists in available actions
        available_actions = self.world_model.get("available_actions", [])
        action_ids = [action["id"] for action in available_actions]

        # Check if action has function_name that matches an available action ID
        function_name = action_data.get("function_name")

        # Also handle backward compatibility if someone uses 'id' field
        if not function_name:
            function_name = action_data.get("id")

        print(
            f"[DEBUG] Looking for function_name: '{function_name}' in available actions: {action_ids}"
        )

        if function_name and function_name in action_ids:
            # Store action in the expected format
            self.selected_action = {
                "function_name": function_name,
                "parameters": action_data.get("parameters", {}),
            }
            print(f"[DEBUG] Successfully selected action: {self.selected_action}")
            return "Action selected successfully. This should help resolve the identified bottleneck."
        else:
            print(
                f"[DEBUG] Action not found: '{function_name}', available: {action_ids}"
            )
            # Try to find a similar action as fallback
            similar_action = None
            if function_name:
                for action_id in action_ids:
                    if (
                        function_name.lower() in action_id.lower()
                        or action_id.lower() in function_name.lower()
                    ):
                        similar_action = action_id
                        break

            if similar_action:
                print(
                    f"[DEBUG] Found similar action: {similar_action}, using as fallback"
                )
                self.selected_action = {
                    "function_name": similar_action,
                    "parameters": action_data.get("parameters", {}),
                }
                return f"Action selected (corrected from '{function_name}' to '{similar_action}'). This should help resolve the identified bottleneck."
            else:
                return f"Invalid action '{function_name}'. Available actions are: {', '.join(action_ids[:5])}{'...' if len(action_ids) > 5 else ''}. Please select from the available actions."

    def _parse_thought_action(self, response: str) -> tuple:
        """Parse LLM response to extract Thought and Action."""
        import re

        # Look for pattern: "Thought: ... Action: ..."
        thought_pattern = r"Thought:\s*(.+?)(?=Action:|$)"
        action_pattern = r"Action:\s*(.+?)(?=Observation:|$|Thought:)"

        thought_match = re.search(thought_pattern, response, re.DOTALL | re.IGNORECASE)
        action_match = re.search(action_pattern, response, re.DOTALL | re.IGNORECASE)

        if thought_match and action_match:
            thought = thought_match.group(1).strip()
            action = action_match.group(1).strip()
            return (thought, action)

        return None

    def _parse_action(self, action_str: str) -> tuple:
        """Parse action string to extract type and argument."""
        import re

        pattern = r"^(\w+)\[(.+)\]$"
        match = re.match(pattern, action_str.strip())

        if match:
            action_type = match.group(1)
            argument = match.group(2)
            return action_type, argument

        return None, None

    def _extract_fallback_info(self, action_text: str):
        """Extract information from malformed responses as fallback."""

        # Try to find JSON in the action text
        json_pattern = r"\{[^{}]*\}"
        matches = re.findall(json_pattern, action_text)

        for match in matches:
            try:
                data = json.loads(match)
                if "function_name" in data:
                    # This looks like an action
                    available_actions = self.world_model.get("available_actions", [])
                    action_ids = [action["id"] for action in available_actions]
                    if data["function_name"] in action_ids:
                        self.selected_action = {
                            "function_name": data["function_name"],
                            "parameters": data.get("parameters", {}),
                        }
                        return

            except json.JSONDecodeError:
                continue

        # Try to find document IDs
        doc_matches = re.findall(r"([a-zA-Z0-9_]+(?:_[a-fA-F0-9]{8,}))", action_text)
        if doc_matches:
            for doc_id in doc_matches:
                if doc_id not in self.retrieved_documents:
                    self.retrieved_documents.append(doc_id)

    def _reflect(self):
        """Generate reflection based on previous failed attempt."""
        print("Running Reflexion strategy...")
        reflection = self._prompt_reflection()
        self.reflections.append(reflection)
        self.reflections_str = self._format_reflections()
        print(self.reflections_str)

    def _prompt_reflection(self) -> str:
        """Generate reflection prompt."""
        persona = self.world_model.get("persona", {})

        # Simple formatting like baseline agent
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )

        # Create world_model without available_actions
        world_model_clean = {
            k: v for k, v in self.world_model.items() if k != "available_actions"
        }
        world_model_str = json.dumps(world_model_clean, indent=2)

        prompt = render_template(
            "reflection.j2",
            persona=persona_str,
            world_model=world_model_str,
            scratchpad=self.scratchpad,
        )

        if self.supports_temperature():
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
        else:
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

        return self._format_step(response.choices[0].message.content)

    def _evaluator_step(
        self, result: Dict[str, Any], world_model: Dict[str, Any]
    ) -> float:
        """Evaluator assesses action quality via LLM reward scoring."""
        persona = world_model.get("persona", {})
        personal_context = world_model.get("personal_context", {})

        # Format persona as JSON string for consistency
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )

        # Format world model for evaluation
        world_model_text = []
        if personal_context.get("current_priorities"):
            world_model_text.append(
                f"Current Priorities: {personal_context.get('current_priorities', [])}"
            )
        if personal_context.get("pain_points"):
            world_model_text.append(
                f"Pain Points: {personal_context.get('pain_points', [])}"
            )

        evaluation_prompt = render_template(
            "evaluation.j2",
            persona=persona_str,
            world_model=(
                "\n".join(world_model_text)
                if world_model_text
                else "No additional context"
            ),
            retrieved_documents=result["retrieved_documents"],
            bottleneck=result["bottleneck"],
            action=result["action"],
        )

        if self.supports_temperature():
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": evaluation_prompt}],
                temperature=0.1,
            )
        else:
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": evaluation_prompt}]
            )

        score_text = response.choices[0].message.content.strip()
        print(f"[DEBUG] Evaluator response: {score_text}")
        score = float(score_text)
        return max(0.0, min(1.0, score))  # Clamp to [0,1]

    def _reflection_step(
        self, result: Dict[str, Any], reward: float, world_model: Dict[str, Any]
    ) -> str:
        """Generate reflection for future trials."""
        persona = world_model.get("persona", {})
        personal_context = world_model.get("personal_context", {})

        # Format persona as JSON string for consistency
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )

        # Format world model for reflection
        world_model_text = []
        if personal_context.get("current_priorities"):
            world_model_text.append(
                f"Current Priorities: {personal_context.get('current_priorities', [])}"
            )
        if personal_context.get("pain_points"):
            world_model_text.append(
                f"Pain Points: {personal_context.get('pain_points', [])}"
            )

        prompt = render_template(
            "reflection_step.j2",
            reward=f"{reward:.2f}",
            retrieved_documents=result["retrieved_documents"],
            bottleneck=result["bottleneck"],
            action=result["action"],
            persona=persona_str,
            world_model=(
                "\n".join(world_model_text)
                if world_model_text
                else "No additional context"
            ),
            scratchpad=self.scratchpad,
        )

        if self.supports_temperature():
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], temperature=0.2
            )
        else:
            response = self.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )

        return response.choices[0].message.content.strip()

    def _reset(self):
        """Reset agent state for new trial."""
        self.scratchpad = ""
        self.step_n = 0
        self.finished = False
        self.retrieved_documents = []
        self.bottleneck = ""
        self.selected_action = {}
        # Reset workflow state tracking
        self.has_retrieved = False
        self.has_analyzed = False
        self.last_analyzed_docs = []
        # Reset error tracking
        self.consecutive_invalid_actions = 0
        # Reset analysis data
        self._full_analysis_data = []

    def _is_finished(self) -> bool:
        """Check if workflow analysis is complete."""
        return (
            self.finished
            and self.has_retrieved
            and self.has_analyzed
            and bool(self.selected_action)
        )

    def _is_halted(self) -> bool:
        """Check if agent should halt due to context window limits or repeated errors."""
        # Check for repeated invalid actions (early stopping)
        if self.consecutive_invalid_actions >= self.max_consecutive_invalid:
            print(
                f"[DEBUG] Halting due to repeated invalid actions: {self.consecutive_invalid_actions}"
            )
            return True

        return False

    def _get_halt_reason(self) -> str:
        """Get detailed reason why workflow halted."""
        # Check step limit (though this is now checked earlier in the workflow)
        if self.step_n >= self.max_steps and not self.finished:
            return f"MAX_STEPS_REACHED ({self.step_n}/{self.max_steps})"

        # Check for repeated invalid actions
        if self.consecutive_invalid_actions >= self.max_consecutive_invalid:
            return f"REPEATED_INVALID_ACTIONS ({self.consecutive_invalid_actions}/{self.max_consecutive_invalid})"

        return "UNKNOWN_HALT_REASON"

    def _is_correct(self) -> bool:
        """Check if previous attempt was correct (placeholder)."""
        return self.finished and bool(self.selected_action)

    def _get_current_result(self) -> Dict[str, Any]:
        """Get current analysis result."""
        # Use last analyzed documents as retrieved_documents (as requested)
        result_docs = (
            self.last_analyzed_docs
            if self.last_analyzed_docs
            else self.retrieved_documents or []
        )
        result_bottleneck = self.bottleneck or ""
        result_action = self.selected_action or {}

        # If we have no retrieved documents, try to extract them from various sources
        if not result_docs:
            # Try to extract from bottleneck text
            if result_bottleneck:
                doc_matches = re.findall(
                    r"Document ([a-zA-Z0-9_]+) shows", result_bottleneck
                )
                if doc_matches:
                    result_docs = list(set(doc_matches))  # Remove duplicates

            # Try to extract from scratchpad (look for Analyze[doc_id] calls)
            if not result_docs and self.scratchpad:
                analyze_matches = re.findall(
                    r"Action:\s*Analyze\[([^\]]+)\]", self.scratchpad
                )
                if analyze_matches:
                    # Split comma-separated document IDs and clean them
                    for match in analyze_matches:
                        doc_ids = [doc.strip() for doc in match.split(",")]
                        result_docs.extend(doc_ids)
                    result_docs = list(set(result_docs))  # Remove duplicates

        # If bottleneck is still empty, try to extract from scratchpad
        if not result_bottleneck and self.scratchpad:
            # Look for analysis content in scratchpad
            analysis_pattern = r"The document.*?reveals.*?(?=Thought:|Action:|$)"
            matches = re.findall(
                analysis_pattern, self.scratchpad, re.DOTALL | re.IGNORECASE
            )
            if matches:
                # Take the first substantial analysis
                analysis = matches[0].strip()
                if len(analysis) > 50:  # Make sure it's substantial
                    result_bottleneck = analysis

            # Also look for explicit bottleneck mentions
            if not result_bottleneck:
                bottleneck_pattern = (
                    r"(?:bottleneck|issue|problem).*?(?=\.|Thought:|Action:|$)"
                )
                matches = re.findall(
                    bottleneck_pattern, self.scratchpad, re.DOTALL | re.IGNORECASE
                )
                if matches:
                    result_bottleneck = matches[0].strip()

        # If we have a very long bottleneck, try to extract a cleaner summary
        if result_bottleneck and len(result_bottleneck) > 500:
            # Try to find the actual bottleneck description
            lines = result_bottleneck.split("\n")
            summary_lines = []
            for line in lines:
                if (
                    "bottleneck" in line.lower()
                    or "issue" in line.lower()
                    or "problem" in line.lower()
                    or "delay" in line.lower()
                ):
                    summary_lines.append(line.strip())

            if summary_lines:
                result_bottleneck = ". ".join(summary_lines)  # First 3 relevant lines
            else:
                result_bottleneck = result_bottleneck

        return {
            "retrieved_documents": result_docs,
            "bottleneck": result_bottleneck,
            "action": result_action,
            "trial": self.trial_count,
            "reflections_used": len(self.reflections),
        }

    def _format_reflections(self) -> str:
        """Format reflections using sliding window like original Reflexion."""
        if not self.reflections:
            return ""
        else:
            # Sliding window: keep only the most recent 2 reflections to manage context
            recent_reflections = self.reflections[-2:]
            reflections_text = "\n- ".join([r.strip() for r in recent_reflections])
            return "Previous learnings:\n- " + reflections_text

    def _truncate_scratchpad_if_needed(self) -> None:
        """Truncate scratchpad using sliding window approach like original Reflexion."""
        # Count tokens in scratchpad
        try:
            scratchpad_tokens = len(self.enc.encode(self.scratchpad))
            # If scratchpad alone is > 3000 tokens, truncate it
            if scratchpad_tokens > 3000:
                # Keep only the most recent interactions (sliding window)
                lines = self.scratchpad.split("\n")
                # Keep last 20 lines (roughly last 2-3 interactions)
                truncated_lines = lines[-20:]
                self.scratchpad = "\n".join(truncated_lines)
                print(
                    f"[DEBUG] Truncated scratchpad from {scratchpad_tokens} tokens to ~{len(self.enc.encode(self.scratchpad))} tokens"
                )
        except Exception as e:
            print(f"[DEBUG] Error truncating scratchpad: {e}")

    def _format_step(self, step: str) -> str:
        """Format step output."""
        return step.strip("\n").strip().replace("\n", "")
