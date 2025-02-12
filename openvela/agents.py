import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Required, TypedDict

from sqlalchemy import create_engine, text

from openvela.llms import GroqModel, Model, OllamaModel
from openvela.memory import JsonShortTermMemory, WorkflowMemory
from openvela.tools import AIFunctionTool


@dataclass
class Agent:
    """
    Represents an individual agent within the OpenVela framework.
    Each agent is configured with specific settings, utilizes a language model, and interacts with tools and memory.
    """

    settings: dict
    model: Model = field(default_factory=GroqModel)
    tools: Optional[List[AIFunctionTool]] = None
    tools_choice: Optional[str] = None
    memory_id: Optional[str] = None
    options: dict = field(default_factory=dict)
    name: str = field(init=False)
    prompt: str = field(init=False)
    memory: WorkflowMemory = field(init=False)

    def __post_init__(self):
        """
        Post-initialization processing to set agent's name and prompt from settings.
        Initializes additional attributes like fluid_input and input.
        """
        self.name = self.settings.get("name", "Agent")
        self.prompt = self.settings.get("prompt", "")
        self.description = self.settings.get("description", "")
        self.memory = WorkflowMemory(memory_id=self.memory_id)  # Set the memory prompt
        self.extra_info = self.settings.get("extra_info", "")
        self.input = self.settings.get("input", "")
        self.options = self.settings.get("options", {})

        # logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    def set_memory(self, memory: WorkflowMemory):
        """
        Sets the memory ID for the agent's memory.

        Args:
            memory_id (str): The ID to set for the memory.
        """

        self.memory = memory
        self.memory_id = memory.memory_id

    def generate(self, **kwargs) -> str:
        """
        Processes a single thought using the language model.

        Args:
            **kwargs: Additional keyword arguments to pass to the model.

        Returns:
            str: The model's response to the thought.
        """
        single_thought_messages = []
        messages = self.memory.load()
        extra_args = {**kwargs, **self.options}
        single_thought_messages.extend(messages)
        if kwargs.get("max_previous_messages"):
            single_thought_messages = single_thought_messages[
                -int(kwargs.get("max_previous_messages")) :
            ]
        single_thought_messages.insert(0, {"role": "system", "content": self.prompt})

        if self.extra_info:
            self.input += f"\n\n{self.extra_info}"
        single_thought_messages.append({"role": "user", "content": self.input})
        self.memory.add_message("", "user", self.input)

        print(single_thought_messages)
        response = self.model.generate_response(single_thought_messages, **extra_args)

        self.memory.add_message(self.name, "assistant", response)
        return response

    def __eq__(self, other):
        """
        Checks equality based on the agent's name.

        Args:
            other: Another agent instance to compare with.

        Returns:
            bool: True if names are equal, False otherwise.
        """
        if isinstance(other, Agent):
            return self.name == other.name
        return False

    def __hash__(self):
        """
        Returns the hash based on the agent's name.

        Returns:
            int: The hash of the agent's name.
        """
        return hash(self.name)


@dataclass
class SupervisorAgent(Agent):
    """
    Specialized agent that supervises the workflow, managing the sequence of
    agents and evaluating their outputs.
    """

    task: str = ""
    agent_type: Literal["selector", "simple"] = "simple"
    end_agent: Optional[Agent] = None
    start_agent: Optional[Agent] = None
    agents: List[Agent] = field(default_factory=list)

    def generate(self, **kwargs):
        return super().generate(**kwargs)

    def choose_next_agent(self, current_agent: Agent, latest_output: str) -> dict:
        """
        Decide which agent should act next or whether the workflow should FINISH.

        Returns a dict of the form:
            {
                "next_agent": "<AgentName or FINISH>",
                "next_input": "<content for the next agent or final output>"
            }


        The workflow can interpret "FINISH" in "next_agent" to conclude the process.
        """
        # -----------------------------
        # SIMPLE MODE: Move in order
        # -----------------------------
        if self.agent_type == "simple":
            try:
                # If we're at the start, return the first actual agent
                if current_agent == self.start_agent:
                    next_agent = self.agents[0]
                else:
                    index = self.agents.index(current_agent)
                    next_agent = self.agents[index + 1]
                return {
                    "next_agent": next_agent.name,
                    "next_input": latest_output,  # pass along the output as the next input
                }
            except (ValueError, IndexError):
                # Reached the end or invalid index - return FINISH
                return {"next_agent": "FINISH", "next_input": latest_output}

        # --------------------------------
        # SELECTOR MODE: Use the LLM to decide
        # --------------------------------
        elif self.agent_type == "selector":
            try:
                # 1) Prepare agent list
                agent_list = "\n".join(
                    [
                        f"Agent Name: {agent.name}\nAgent Description: {agent.description}"
                        for agent in self.agents
                    ]
                )

                # 2) Load conversation from memory

                messages = self.memory.load_messages_with_agent_names()

                # Convert messages so that any assistant role is replaced with the agent_name (if available)

                # 3) Build a conversation string or keep it in a list as needed
                conversation = "\n".join(
                    [f"{msg['agent_name']}: {msg['content']}" for msg in messages]
                )

                # 4) System prompt for the LLM
                system_instructions = f"""
You are the Supervisor Agent. Your role is to determine the next action for fulfilling the original TASK. 
Consider the following:

- AGENT LIST:
{agent_list}

- LATEST AGENT OUTPUT:
{latest_output}

- TASK:
{self.task}

If the LATEST AGENT OUTPUT has completely satisfied the TASK requirements, respond in JSON with:
{{
  "next_agent": "FINISH" String Type only,
  "next_input": "<create the most detailed answer for the TASK to serve as final output based on the conversation>"
}}
Replace the next_input with the best possible answer based on the conversation, you need to pass key informations on the input.
the value of the key "next_input" should be the final output of the conversation and should be the most detailed answer for the TASK.
the value of the key "next_input" should be a String Type unless the user ask for a diferent format.


Otherwise, select the best-suited agent from the AGENT LIST and specify the new input or instructions for that agent.
Then respond in JSON with:
{{
  "next_agent": "<AgentName>" String Type only,
  "thinking": "<reasoning or explanation for the choice>" String Type Only,
  "next_input": "<Next input or instructions for the chosen agent>" String Type Only
}}

Replace <AgentName> with the actual name of the chosen agent
Replace <Next input or instructions fort the chosen agent> with:
- The next question or directive to guide the agent
- Any additional information or context required for the agent to proceed
- All the information that the agent needs to process the information correctly
- If the agent needs to provide a specific type of response, specify it clearly
- If the agent needs to follow a specific format or structure, provide detailed instructions
- If the agent needs to consider certain aspects or criteria, outline them clearly
- If the agent needs to focus on particular elements or details, highlight them explicitly
- If the agent needs to address specific challenges or issues, describe them in detail
- If the agent needs to explore alternative solutions or approaches, suggest them explicitly
- If the agent needs to make a research or analysis, provide clear guidelines and information




Remember to guide the conversation towards the completion of the TASK by selecting the most appropriate agent and providing relevant instructions.

Additional Instructions:

{self.prompt}
                """

                # 5) Build messages for the LLM
                llm_messages = [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": "CONVERSATION:\n" + conversation},
                ]

                # 6) Generate a response in JSON
                response = self.model.generate_response(llm_messages, format="json")

                # 7) Parse JSON
                parsed_response = json.loads(response)

                # 8) Return the structure that the workflow expects
                self.memory.add_message(
                    "SupervisorAgentAsUser", "user", parsed_response.get("next_input")
                )
                return {
                    "next_agent": parsed_response.get("next_agent", "FINISH"),
                    "next_input": parsed_response.get("next_input", latest_output),
                }

            except Exception as e:
                logging.error(f"Error in supervisor agent (selector mode): {e}")
                # If there's any error, gracefully end
                return {"next_agent": "FINISH", "next_input": latest_output}

        else:
            logging.warning(f"Unknown supervisor mode: {self.agent_type}. Finishing.")
            return {"next_agent": "FINISH", "next_input": latest_output}


@dataclass
class StartAgent(Agent):
    """
    Represents the starting agent in a workflow.
    Inherits all functionalities from the base Agent class.
    """

    pass


@dataclass
class EndAgent(Agent):
    """
    Represents the ending agent in a workflow.
    Inherits all functionalities from the base Agent class.
    """

    pass


@dataclass
class FluidValidator(Agent):
    """
    Specialized agent that validates the task description before generating agents.
    """

    def generate(self, **kwargs):
        return super().generate(**kwargs)

    def validate_output(self, task_description: str, answer: str) -> dict:
        class ValidatorOutput(TypedDict):
            valid: Required[bool]
            feedback: Required[str]

        """
        Validates the task description to ensure it meets the requirements for agent generation.

        Args:
            task_description (str): The task description to validate.

        Returns:
            bool: True if the task description is valid, False otherwise.
        """
        prompt = """
        You are the FluidValidator. Your role is to validate the ANSWER provided based on the TASK.
        You will assess the response to ensure it aligns with the requirements and expectations of the task.
        Analyze the response and determine if it accurately addresses the main objectives and provides a comprehensive solution.
        Evaluate the response based on the context of the task and the information provided in the ANSWER.
        Provide feedback on the accuracy and relevance of the response, highlighting any areas that require improvement.
        You will receive the TASK description and the ANSWER to evaluate in the following format:
        "
        TASK: [Task description]
        ANSWER: [Response to evaluate]
        "
        Return the response in the following JSON format:
            
            {
            "valid": boolean,
            "feedback": str [Containing feedback on the response]
            }
        
        """

        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"TASK: {task_description}\nANSWER: {answer}"},
        ]
        response = self.model.generate_response(messages, format="json")
        return ValidatorOutput(**json.loads(response))


@dataclass
class FluidAgent(Agent):
    """
    Specialized agent that dynamically generates and manages a set of agents based on a task description.
    Facilitates the creation of complex workflows by defining a sequence of agents.
    """

    def generate(self, **kwargs):
        return super().generate(**kwargs)

    def generate_agents_from_task(self, task_description: str, **kwargs) -> List[Dict]:
        """
        Generates agent definitions based on the provided task description.

        Utilizes the language model to create a JSON structure defining a comprehensive workflow of agents.

        Args:
            task_description (str): The description of the task to generate agents for.

        Returns:
            List[Dict]: A list of agent definitions as dictionaries.

        Logs an error if JSON decoding fails and returns an empty list.
        """
        prompt = """
         Based on the following task description, generate a JSON object defining a comprehensive workflow of agents to accomplish the main task. The workflow should initiate with a `StartAgent`, proceed through multiple intermediary agents in a cascading sequence, and conclude with an `EndAgent`. Each agent should add relevant insights or process data based on its unique function, enriching the conversation and enhancing the final output.

**JSON Structure:**

{
  "agents": [
    {
      "name": "StartAgent",
      "prompt": "System prompt of the StartAgent. This agent initiates the workflow based on the task description or user input. Provide a detailed context (minimum 50 words) explaining the StartAgent's role in setting up the workflow.",
      "input": "Initial task description or user input to begin the workflow."
    },
    {
      "name": "Agent1",
      "prompt": "System prompt of Agent1. Define this agent's specific function within the workflow, detailing how it processes or adds insights to the information received from the StartAgent.",
      "input": "Simulated question or directive that Agent1 uses to enhance the workflow."
    },
    {
      "name": "Agent2",
      "prompt": "System prompt of Agent2. Describe this agent's role in further processing the data or insights provided by Agent1, contributing to the enrichment of the conversation.",
      "input": "Simulated question or directive that Agent2 uses to build upon Agent1's contributions."
    },
    ...
    {
      "name": "EndAgent",
      "prompt": "System prompt of the EndAgent. This agent synthesizes all the insights and data gathered from the preceding agents to provide a comprehensive and cohesive final response. Explain the EndAgent's role in integrating the workflow's collective efforts.",
      "input": "Final directive to synthesize and deliver the completed task outcome."
    }
  ]
}

**Instructions:**

1. **Workflow Initiation:**
   - **StartAgent:**
     - **Name:** Assign as "StartAgent".
     - **Prompt:** Craft a detailed system prompt (minimum 50 words) that defines the StartAgent's role in initiating the workflow based on the task description or user input.
     - **Input:** Create an initial question or directive that the StartAgent uses to begin the workflow.

2. **Intermediate Agents:**
   - **Sequence:** Add multiple agents (Agent1, Agent2, etc.) that follow in a logical sequence.
   - **Name:** Assign meaningful names to each intermediary agent (e.g., "ResearchAgent", "AnalysisAgent").
   - **Prompt:** For each agent, write a detailed system prompt that defines its unique function within the workflow, explaining how it processes data or adds insights based on inputs from the preceding agent.
   - **Input:** Develop simulated questions or directives that each agent would use to further the workflow and enhance the final answer.

3. **Workflow Progression:**
   - Ensure each agent builds upon the information and insights provided by the previous agents.
   - Maintain a logical and cohesive flow of information from one agent to the next.

4. **EndAgent:**
   - **Name:** Assign as "EndAgent".
   - **Prompt:** Develop a comprehensive system prompt (minimum 50 words) that defines the EndAgent's role in synthesizing all gathered insights and data to deliver the final, cohesive response.
   - **Input:** Create a final directive that instructs the EndAgent to integrate the workflow's collective efforts and complete the main task.

5. **Structure and Formatting:**
   - Ensure all agents are separated by a comma.
   - Adhere strictly to the provided JSON structure.
   - All keys (`name`, `prompt`, `input`) must be present and contain non-empty string values.
   - Be detailed in both prompts and inputs to facilitate a comprehensive final output.

6. **Learning and Enhancement:**
   - Design each agent to learn from the previous messages, enabling progressively improved responses throughout the workflow.

7. **Output Only:**
   - Return only the JSON object containing the agents.
   - Do not include any additional text or explanations.
   
8. **Limit of agents:**
    - Define the dificuty of the task in hard, medium or easy:
        -Hard examples:
            - "Can you explain the concept of quantum entanglement and its implications in modern physics?"
            - "Summarize the key findings of the latest research on renewable energy technologies."
            - "Can you help me outline a novel set in a dystopian future where technology controls society?"
        -Medium examples:
            - "Can you help me draft a professional email to request a meeting?"
            - "How can I create a pivot table in Excel to analyze my sales data?"
            - "I'm getting an error in my Python code. Can you help me debug it?"
        -Easy examples:
            - "What is the capital of Japan?"
            - "Can you correct this sentence: 'She dont like apples.'?"
    - Just create the number of agents according to the difficulty:
        - Hard: 5 agents
        - Medium: 4 agents
        - Easy: 3 agents
        
9. **Identify the language:**
    - Identify the language of the task description and set the language in the agents workflow.
    - Do not translate the json structure keys, just the values.
    
10. **Do not modify the structure:**
    - Do not modify the structure of the json, just the values.
    - the json needs to follow the structure:
        {
            "agents": [
                {
                    "name": "name of the agent",
                    "prompt": "System prompt of the agent. Describe the agent's role in the workflow and its function.",
                    "input": "Simulated user input. Provide a question or directive for the agent to process or respond to. It has to be related to the previous agent."
                }
                ...
            ]
        }



**Example Structure:**

```json
{
  "agents": [
    {
      "name": "StartAgent",
      "prompt": "You are the StartAgent. Your role is to initiate the workflow by analyzing the task description or user input. You will set the foundation for subsequent agents by outlining the primary objectives and necessary information required to address the task effectively.",
      "input": "Please provide a detailed overview of the main objectives and key information needed to accomplish the task."
    },
    {
      "name": "ResearchAgent",
      "prompt": "You are the ResearchAgent. Your function is to gather relevant data and information based on the overview provided by the StartAgent. You will compile comprehensive research that will serve as the foundation for further analysis.",
      "input": "What are the latest findings and relevant information pertaining to the main objectives outlined by the StartAgent?"
    },
    {
      "name": "AnalysisAgent",
      "prompt": "You are the AnalysisAgent. Your role is to evaluate the data collected by the ResearchAgent, identifying patterns, trends, and key insights that will enhance the understanding of the task.",
      "input": "Based on the research data, what are the significant trends and insights that emerge?"
    },
    {
      "name": "StrategyAgent",
      "prompt": "You are the StrategyAgent. Your task is to develop strategies and recommendations utilizing the insights provided by the AnalysisAgent to address the main objectives effectively.",
      "input": "What strategies and recommendations can be formulated from the identified insights to achieve the task's objectives?"
    },
    {
      "name": "EndAgent",
      "prompt": "You are the EndAgent. Your responsibility is to synthesize all the information, insights, and strategies developed by the previous agents to deliver a comprehensive and cohesive final response that thoroughly addresses the main task.",
      "input": "Please integrate all gathered data, insights, and strategies to provide a detailed and cohesive final answer to the main task."
    }
  ]
}
        """

        messages = self.memory.load()
        messages.insert(0, {"role": "system", "content": prompt})
        messages.append({"role": "user", "content": task_description})
        response = self.model.generate_response(messages, format="json", **kwargs)
        print(response)

        try:
            agents_json = json.loads(response)
            return agents_json.get("agents", [])
        except json.JSONDecodeError:
            logging.error("Failed to decode agents JSON.")
            return []

    def create_agents(
        self, agents_definitions: List[Dict], memory_id: str
    ) -> List[Agent]:
        """
        Creates Agent instances from their definitions.

        Args:
            agents_definitions (List[Dict]): A list of agent definitions.
            memory (WorkflowMemory): The workflow memory to associate with agents.

        Returns:
            List[Agent]: A list of instantiated Agent objects.
        """
        logging.debug(f"{self.name} is creating agents from definitions.")
        agents = []
        for agent_def in agents_definitions:
            agent = Agent(settings=agent_def, memory_id=memory_id, model=self.model)
            agents.append(agent)
        return agents


@dataclass
class SQLAgent(Agent):
    """
    Specialized Agent for generating read-only SQL queries.
    It uses the language model to generate or refine queries based on the user's request,
    then executes them via SQLAlchemy and returns the results.
    """

    # Example of user questions and reference queries that can be provided
    example_queries: List[Dict[str, str]] = field(default_factory=list)
    # The SQL dialect (e.g., "postgresql", "mysql", "sqlite", etc.)
    sql_dialect: str = ""
    # The SQLAlchemy database URL (e.g., "postgresql://user:password@host/dbname")
    sqlalchemy_engine_url: str = ""
    # A textual description of the database structure/schema for context
    database_structure: str = ""
    # Additional instructions for the language model (if any)

    # Instructions for formatting the final response
    formatter_prompt: str = ""

    def generate(self, **kwargs) -> str:
        """
        Uses the language model to generate or refine a SQL query from the user's request,
        then executes the query in a read-only manner and returns the results.
        """
        max_attempts = 3
        attempt = 0
        last_error = None  # Stores the error from the SQL execution if it occurs

        while attempt < max_attempts:
            attempt += 1

            # 1. Build a string of example queries (if provided)
            query_examples = ""
            for ex in self.example_queries:
                query_examples += (
                    f"- Question: {ex.get('question')}\n"
                    f"  Query: {ex.get('sql_query')}\n"
                )

            # 2. If there's an error from the previous attempt, get a friendly explanation
            error_info = ""
            if last_error:
                error_explanation_prompt = (
                    "You are an assistant tasked with explaining SQL error messages in a concise and user-friendly manner. "
                    "Given the error detail below, please return a JSON object in the following format:\n"
                    '{"error": "Friendly explanation of what might have gone wrong with the SQL query and how to fix it."}\n'
                    f"Error detail: {last_error}"
                    f"SQL Query: {sql_query}"
                    f"User input: {self.input}"
                )
                error_response = self.model.generate_response(
                    [{"role": "system", "content": error_explanation_prompt}],
                    format="json",
                )
                try:
                    error_json = json.loads(error_response)
                    explained_error = error_json.get("error", last_error)
                    error_info = f"\nPrevious attempt error: {explained_error}\n"
                except json.JSONDecodeError:
                    # Fallback to the raw error if JSON parsing fails
                    error_info = f"\nPrevious attempt error: {last_error}\n"

            # 3. Create the system prompt with all the context and instructions
            system_prompt = (
                f"You are a read-only SQL Agent using the '{self.sql_dialect}' dialect. "
                "Your role is to generate simple, clear SELECT queries or other non-mutating SQL statements based solely on the user's request. "
                "You have access to the following database structure:\n"
                f"{self.database_structure}\n\n"
                "IMPORTANT: You MUST ONLY generate read-only queries (e.g. SELECT, SHOW, etc.). "
                "Do NOT use JOIN, and avoid creating overly complex queries. "
                "Return the SQL query enclosed in a valid JSON object with the key 'sql_query'. "
                "For example:\n\n"
                '{"sql_query": "SELECT column FROM table_name WHERE condition;"}\n\n'
                "Here are some example queries to guide you:\n\n"
                f"{query_examples}\n\n"
                f"{error_info}"
                "Additional instructions:\n"
                f"{self.prompt}\n\n"
                "Please generate a valid, simple SELECT query in the specified SQL dialect."
            )

            # 4. Build the user message with the current input
            user_content = f"\n{self.input}\n\n"

            # 5. Prepare the messages for the LLM
            query_messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            # 6. Generate the LLM response (expected to be a JSON with the key "sql_query")
            llm_response = self.model.generate_response(query_messages, format="json")

            # 7. Format the response to ensure it's valid JSON with the SQL query
            sql_formatter_prompt = (
                "You are responsible for ensuring the SQL query is correctly formatted in JSON. "
                "The content should be a JSON object with a single key 'sql_query'. "
                "Do not modify the SQL query; just ensure the format is valid JSON. "
                "If the provided content is not valid JSON, return a corrected JSON object with the key 'sql_query'."
            )
            sql_formatter_messages = [
                {"role": "system", "content": sql_formatter_prompt},
                {"role": "user", "content": llm_response},
            ]
            sql_query_formatted = self.model.generate_response(
                sql_formatter_messages, format="json"
            )

            # 8. Parse the JSON to extract the SQL query
            try:
                response_json = json.loads(sql_query_formatted)
                sql_query = response_json.get("sql_query", "").strip()

            except json.JSONDecodeError:
                logging.error("Failed to parse LLM response as valid JSON.")
                last_error = "LLM returned invalid JSON."
                if attempt >= max_attempts:
                    return "Error: The agent could not produce valid JSON for the SQL query after multiple attempts."
                continue

            # 9. Ensure the generated query is read-only (only SELECT or SHOW)
            upper_query = sql_query.upper()
            if not (upper_query.startswith("SELECT") or upper_query.startswith("SHOW")):
                last_error = "Generated query is not read-only."
                if attempt >= max_attempts:
                    return (
                        "Error: The generated query is not read-only. "
                        "SQLAgent is restricted to read-only statements."
                    )
                continue

            # 10. Execute the query using SQLAlchemy in a read-only manner
            try:
                engine = create_engine(self.sqlalchemy_engine_url)
                with engine.connect() as connection:
                    # Optionally enforce read-only at the database level if supported
                    result = connection.execute(text(sql_query))
                    rows = result.fetchall()

                # Prepare a string version of the results
                header = result.keys() if result.returns_rows else []
                rows_str = "\n".join([str(dict(zip(header, row))) for row in rows])

                # 11. Format the results for the user with a clear and friendly output
                formatter_prompt = (
                    "You will receive raw output from a database query. "
                    "Format this output in a clear, user-friendly manner. "
                    "Return the formatted output in JSON format with the key 'formatted_response'. "
                    "For example:\n\n"
                    '{"formatted_response": "Here is the nicely formatted result..."}\n\n'
                    f"Additional instructions: {self.formatter_prompt}\n\n"
                    "User's Input:\n"
                    f"{user_content}\n\n"
                    "Raw Output:\n"
                    f"{rows_str}"
                )
                formatter_messages = [
                    {"role": "system", "content": formatter_prompt},
                    {"role": "user", "content": rows_str},
                ]
                formatter_response = self.model.generate_response(
                    formatter_messages, **kwargs, format="json"
                )
                formatter_response_json = json.loads(formatter_response)
                self.memory.add_message(
                    self.name,
                    "assistant",
                    formatter_response_json.get("formatted_response", rows_str),
                )
                if not rows:
                    return f"No rows returned for query:\n{sql_query}"
                return formatter_response_json.get("formatted_response", rows_str)

            except Exception as e:
                # 12. Capture any execution error, log it, and include it in the next prompt
                logging.error(f"Error executing SQL query: {e}")
                last_error = str(e)
                self.memory.add_message(
                    "ErrorFromAgentRunning", "assistant", last_error
                )
                if attempt >= max_attempts:
                    return (
                        f"Error executing SQL query after multiple attempts:\n{str(e)}"
                    )
                continue
