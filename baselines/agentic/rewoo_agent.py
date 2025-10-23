import re
import json
from typing import Dict, List, Any, Union
from baselines.agentic.base_agent import BaseAgent
from baselines.agentic.litellm_model import LitellmModel
from baselines.agentic.utils import create_datastore
from baselines.agentic.prompts import render_template


class Node:
    """Basic Node to be inherited from (copied from original nodes/Node.py)."""

    def __init__(self, name):
        self.name = name

    def run(self, input, log=False):
        raise NotImplementedError


class LLMNode(Node):
    """Basic LLM node that calls for a Large Language Model (copied from original nodes/LLMNode.py)."""

    def __init__(self, name="BaseLLMNode", model_name="text-davinci-003"):
        super().__init__(name)
        self.model_name = model_name

    def run(self, input, log=False):
        response = self.call_llm(input)
        completion = response["output"]
        if log:
            return response
        return completion

    def call_llm(self, prompt):
        """Call LLM with proper API handling following BaseAgent pattern."""
        messages = [{"role": "user", "content": prompt}]

        # Check if model supports temperature parameter
        model_name = getattr(self.parent_agent, "model", "")
        if isinstance(model_name, str) and (
            "gpt-5" in model_name.lower() or "o1" in model_name.lower()
        ):
            # GPT-5 and O1 models don't support custom temperature
            response = self.parent_agent.create_chat_completion(messages=messages)
        else:
            # Other models support temperature=0.5 (canonical ReWOO setting)
            response = self.parent_agent.create_chat_completion(
                messages=messages, temperature=0.5
            )

        return {"input": prompt, "output": response.choices[0].message.content}


class SemanticDatastoreWorker(Node):
    """Worker that searches documents by semantic similarity."""

    def __init__(self, name="semantic_search"):
        super().__init__(name)
        self.description = "Worker that searches documents by semantic similarity. Useful for exploring the datastore to find relevant documents about workflow bottlenecks, client communications, meetings, or development tasks. Input should be a search query, optionally with k=N to specify number of results."
        self._datastore = None

    def set_datastore(self, datastore):
        """Set the datastore instance to use for queries."""
        self._datastore = datastore

    def run(self, input, log=False):
        if not self._datastore:
            evidence = "No datastore available"
        else:
            # Parse k parameter from query if specified
            k = self._extract_k_from_query(input)
            clean_query = self._clean_query(input)

            # Semantic search with dynamic k
            results = self._datastore.semantic_search(clean_query, top_k=k)

            # Document retrieval and formatting
            results_list = []
            for doc_id in results:
                doc = self._datastore.get_document(doc_id)
                if doc:
                    results_list.append(
                        {
                            "id": doc_id,
                            "type": doc.get("type", "document"),
                            "payload": doc.get("payload", {}),
                        }
                    )

            evidence = (
                json.dumps(results_list, indent=2)
                if results_list
                else "No relevant documents found"
            )

        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence

    def _extract_k_from_query(self, query):
        """Extract k parameter from query string, default to 3."""
        match = re.search(r"k=(\d+)", query)
        return int(match.group(1)) if match else 3

    def _clean_query(self, query):
        """Remove k parameter from query string."""
        return re.sub(r",?\s*k=\d+", "", query).strip()


class SQLDatastoreWorker(Node):
    """Worker that executes SQL queries on the document datastore."""

    def __init__(self, name="sql_reader"):
        super().__init__(name)
        self.description = "Worker that executes SQL queries on the document database. Database schema: documents(id, type, payload). The payload column contains JSON data. Use 'payload LIKE' for text search within JSON content. Example: SELECT * FROM documents WHERE type='email' AND payload LIKE '%keyword%'. Input should be a valid SQL query."
        self._datastore = None

    def set_datastore(self, datastore):
        """Set the datastore instance to use for queries."""
        self._datastore = datastore

    def run(self, input, log=False):
        if not self._datastore:
            evidence = "No datastore available"
        else:
            try:
                # Execute SQL query
                results = self._datastore.sql_query(input)
                evidence = (
                    json.dumps(results, indent=2) if results else "No results found"
                )
            except Exception as e:
                # The datastore already provides helpful error messages
                evidence = f"SQL query error: {str(e)}"

        if log:
            print(f"Running {self.name} with input {input}\nOutput: {evidence}\n")
        return evidence


class LLMWorker(Node):
    """Worker that uses LLM for general reasoning (copied from original LLMWorker)."""

    def __init__(self, name="LLM", model_name="text-davinci-003", parent_agent=None):
        super().__init__(name)
        self.model_name = model_name
        self.parent_agent = parent_agent
        self.description = "Worker that uses LLM for general reasoning. Useful for analyzing retrieved documents to identify bottleneck patterns, determining intervention needs, or reasoning about workflow issues. Input can be any instructions or questions."

    def run(self, input, log=False):
        prompt = f"""Instructions: Respond in short directly with no extra words.

Query: {input}"""

        # Check if model supports temperature parameter
        model_name = (
            getattr(self.parent_agent, "model", "")
            if hasattr(self, "parent_agent") and self.parent_agent
            else ""
        )
        if isinstance(model_name, str) and (
            "gpt-5" in model_name.lower() or "o1" in model_name.lower()
        ):
            # GPT-5 and O1 models don't support custom temperature
            response = self.parent_agent.create_chat_completion(
                messages=[{"role": "user", "content": prompt}]
            )
        else:
            # Other models support temperature=0.5 (canonical ReWOO setting)
            response = self.parent_agent.create_chat_completion(
                messages=[{"role": "user", "content": prompt}], temperature=0.5
            )

        evidence = response.choices[0].message.content.strip()
        if log:
            return {"input": input, "output": evidence}
        return evidence


def get_worker_registry(model_name="text-davinci-003", parent_agent=None):
    """Create worker registry with specified model and parent agent."""
    return {
        "semantic_search": SemanticDatastoreWorker(),
        "sql_reader": SQLDatastoreWorker(),
        "LLM": LLMWorker(model_name=model_name, parent_agent=parent_agent),
    }


class Planner(LLMNode):
    """Planner component (copied from original nodes/Planner.py)."""

    def __init__(
        self, workers, worker_registry, model_name="text-davinci-003", parent_agent=None
    ):
        super().__init__("Planner", model_name)
        self.parent_agent = parent_agent
        self.workers = workers
        self.worker_registry = worker_registry

    def run(self, persona=None, world_model=None, log=False):
        tools_prompt = self._generate_tools_prompt()

        # Format persona and world_model as JSON strings like baseline agent
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )
        world_model_str = json.dumps(world_model, indent=2) if world_model else "{}"

        prompt = render_template(
            "rewoo_planner.j2",
            tools=tools_prompt,
            persona=persona_str,
            world_model=world_model_str,
        )
        response = self.call_llm(prompt)
        completion = response["output"]
        if log:
            return response
        return completion

    def _get_worker(self, name):
        if name in self.worker_registry:
            return self.worker_registry[name]
        else:
            raise ValueError("Worker not found")

    def _generate_tools_prompt(self):
        prompt = "Available tools:\n"
        for name in self.workers:
            worker = self._get_worker(name)
            prompt += f"{worker.name}[input]: {worker.description}\n"
        return prompt


class Solver(LLMNode):
    """Solver component - updated to generate structured JSON output."""

    def __init__(self, model_name="text-davinci-003", parent_agent=None):
        super().__init__("Solver", model_name)
        self.parent_agent = parent_agent

    def run(self, worker_log, world_model=None, persona=None, log=False):
        world_model_str = json.dumps(world_model, indent=2) if world_model else "{}"
        persona_str = (
            json.dumps(persona, indent=2)
            if persona
            else "No persona information provided"
        )

        prompt = render_template(
            "rewoo_solver.j2",
            worker_log=worker_log,
            world_model=world_model_str,
            persona=persona_str,
        )

        response = self.call_llm(prompt)
        completion = response["output"]
        if log:
            return response
        return completion


class PWS:
    """Main class chaining Planner, Worker and Solver (copied from original algos/PWS.py)."""

    def __init__(
        self,
        available_tools=["semantic_search", "sql_reader", "LLM"],
        planner_model="text-davinci-003",
        solver_model="text-davinci-003",
        parent_agent=None,
    ):
        self.workers = available_tools
        self.worker_registry = get_worker_registry(
            planner_model, parent_agent=parent_agent
        )
        self.planner = Planner(
            workers=self.workers,
            worker_registry=self.worker_registry,
            model_name=planner_model,
            parent_agent=parent_agent,
        )
        self.solver = Solver(model_name=solver_model, parent_agent=parent_agent)
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}
        self._current_world_model = {}
        self._current_persona = {}

    def run(self, world_model=None, persona=None):
        """Main run method - updated to pass context to Solver."""
        # run is stateless, so we need to reset the evidences
        self._reinitialize()

        # Store context for Solver
        self._current_world_model = world_model or {}
        self._current_persona = persona or {}

        result = {}

        # Plan - pass persona and world_model directly like baseline agent
        planner_response = self.planner.run(
            persona=persona, world_model=world_model, log=True
        )
        plan = planner_response["output"]
        self.plans = self._parse_plans(plan)
        self.planner_evidences = self._parse_planner_evidences(plan)

        # Work
        self._get_worker_evidences()
        worker_log = ""
        for i in range(len(self.plans)):
            e = f"#E{i + 1}"
            worker_log += f"{self.plans[i]}\nEvidence:\n{self.worker_evidences[e]}\n\n"

        # Solve
        solver_response = self.solver.run(
            worker_log,
            world_model=getattr(self, "_current_world_model", {}),
            persona=getattr(self, "_current_persona", {}),
            log=True,
        )
        output = solver_response["output"]

        result["output"] = output
        result["worker_log"] = worker_log

        return result

    def _parse_plans(self, response):
        """Parse plan steps from planner response (copied from original)."""
        plans = []
        for line in response.splitlines():
            if line.startswith("Plan:") or line.startswith("### Plan:"):
                plans.append(line)
        return plans

    def _parse_planner_evidences(self, response):
        """Parse evidence requirements from planner response (copied from original)."""
        evidences = {}
        for line in response.splitlines():
            # Use regex for more robust parsing
            match = re.match(r"^#E(\d+)\s*=\s*(.+)$", line.strip())
            if match:
                evidence_num = match.group(1)
                tool_call = match.group(2).strip()
                evidence_key = f"#E{evidence_num}"
                evidences[evidence_key] = tool_call
        return evidences

    def _get_worker_evidences(self):
        """Use planner evidences to assign tasks to respective workers (copied from original)."""
        for e, tool_call in self.planner_evidences.items():
            if "[" not in tool_call:
                self.worker_evidences[e] = tool_call
                continue
            tool, tool_input = tool_call.split("[", 1)
            tool_input = tool_input[:-1]
            # find variables in input and replace with previous evidences
            for var in re.findall(r"#E\d+", tool_input):
                if var in self.worker_evidences:
                    tool_input = tool_input.replace(
                        var, "[" + self.worker_evidences[var] + "]"
                    )
            if tool in self.workers:
                self.worker_evidences[e] = self.worker_registry[tool].run(tool_input)
            else:
                self.worker_evidences[e] = "No evidence found"

    def _reinitialize(self):
        """Reinitialize state for new run (copied from original)."""
        self.plans = []
        self.planner_evidences = {}
        self.worker_evidences = {}


class Agent(BaseAgent):
    """ReWOO-based baseline agent following original PWS architecture."""

    def __init__(self, model: Union[str, LitellmModel] = "text-davinci-003"):
        """Initialize ReWOO agent with specified model.

        Args:
            model: Either a string (for OpenAI models) or a LitellmModel instance
        """
        super().__init__(model=model)
        model_str = self.get_model_string()
        self.pws = PWS(
            available_tools=["semantic_search", "sql_reader", "LLM"],
            planner_model=model_str,
            solver_model=model_str,
            parent_agent=self,
        )

    def run(
        self,
        memory: List[Dict[str, Any]],
        world_model: Dict[str, Any],
        persona: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """
        Run ReWOO inference on a test sample.

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
        datastore = create_datastore(memory)
        self.pws.worker_registry["semantic_search"].set_datastore(datastore)
        self.pws.worker_registry["sql_reader"].set_datastore(datastore)

        # Run ReWOO inference with context for Solver
        rewoo_result = self.pws.run(world_model=world_model, persona=persona)

        # Parse JSON response, stripping code block markers if present
        content = rewoo_result.get("output", "").strip()
        if content.startswith("```"):
            lines = content.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].strip() == "```":
                lines = lines[:-1]
            content = "\n".join(lines)

        try:
            result = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Response content: {content}")
            # Check if response seems truncated
            if not content.strip().endswith("}"):
                print(
                    "Warning: JSON response appears to be truncated (doesn't end with '}')"
                )
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
