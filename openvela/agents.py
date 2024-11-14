import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Required, TypedDict

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
    memory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
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
        self.memory = WorkflowMemory(memory_id=self.memory_id)  # Set the memory prompt
        self.input = ""
        self.extra_info = self.settings.get("extra_info", "")
        self.fluid_input = self.settings.get("input", "")
        self.options = self.settings.get("options", {})

        # logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    def process(self, input_data: str) -> str:
        """
        Executes the agent's processing sequence using the Chain of Thought approach.

        Args:
            input_data (str): The input data to process.

        Returns:
            str: The final output after processing.
        """
        observed_data = self.observe(input_data)
        context = self.understand(observed_data)
        goals = self.identify_goals(context)
        info = self.retrieve_information(goals)
        plan = self.plan(info)
        analysis = self.analyze(plan)
        decision = self.decide(analysis)
        action_result = self.execute(decision)
        self.monitor(action_result)
        self.reflect(action_result)
        self.learn(action_result)
        output = self.communicate(action_result)
        return output

    def observe(self, input_data: str) -> str:
        """
        Observes and records the input data.

        Args:
            input_data (str): The data to observe.

        Returns:
            str: The observed data.
        """
        logging.debug(f"{self.name} observes input data.")
        self.input = input_data
        return input_data

    def understand(self, data: str) -> str:
        """
        Processes the observed data to form an understanding.

        Utilizes the language model to generate a response based on current memory.

        Args:
            data (str): The data to understand.

        Returns:
            str: The model's understanding of the data.
        """
        logging.debug(f"{self.name} is understanding the data.")
        messages = self.memory.load()
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def identify_goals(self, context: str) -> List[str]:
        """
        Identifies goals based on the provided context.

        Constructs a prompt incorporating the main task and context, and generates goals using the model.

        Args:
            context (str): The context from which to identify goals.

        Returns:
            List[str]: A list of identified goals.
        """
        logging.debug(f"{self.name} is identifying goals for the main task")
        prompt = f"Based on the following context, identify the objectives for the main task '{self.input}':\n{context}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        goals = response.strip().split(
            "\n"
        )  # Assuming the goals are separated by new lines
        self.memory.add_message("assistant", response)
        return goals

    def retrieve_information(self, goals: List[str]) -> str:
        """
        Retrieves information relevant to the identified goals.

        Args:
            goals (List[str]): The list of goals to retrieve information for.

        Returns:
            str: The retrieved information.
        """
        logging.debug(f"{self.name} is retrieving information.")
        prompt = f"Retrieve information relevant to the following goals for the main task '{self.input}':\n{goals}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def plan(self, info: str) -> str:
        """
        Creates a plan based on the retrieved information.

        Args:
            info (str): The information to base the plan on.

        Returns:
            str: The formulated plan.
        """
        logging.debug(f"{self.name} is planning.")
        prompt = f"Based on the following information, create a plan for the main task '{self.input}':\n{info}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def analyze(self, plan: str) -> str:
        """
        Analyzes the created plan for effectiveness and potential issues.

        Args:
            plan (str): The plan to analyze.

        Returns:
            str: The analysis of the plan.
        """
        logging.debug(f"{self.name} is analyzing the plan.")
        prompt = f"Analyze the following plan for effectiveness and potential issues for the main task '{self.input}':\n{plan}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def decide(self, analysis: str) -> str:
        """
        Makes a decision based on the analysis of the plan.

        Args:
            analysis (str): The analysis to base the decision on.

        Returns:
            str: The decision made.
        """
        logging.debug(f"{self.name} is making a decision.")
        prompt = f"Based on the analysis, decide on the best course of action for the main task '{self.input}':\n{analysis}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def execute(self, decision: str) -> str:
        """
        Executes the decision and provides the result.

        Args:
            decision (str): The decision to execute.

        Returns:
            str: The result of the execution.
        """
        logging.debug(f"{self.name} is executing the decision.")
        prompt = f"Execute the following decision and provide the result for the main task '{self.input}':\n{decision}"
        execute_messages = []
        messages = self.memory.load()
        for message in messages:
            if message["role"] == "assistant":
                execute_messages.append(message)
        execute_messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(execute_messages)
        self.memory.add_message("assistant", response)
        return response

    def monitor(self, result: str):
        """
        Monitors the result of the execution.

        Placeholder for implementing monitoring logic as needed.

        Args:
            result (str): The result to monitor.
        """
        logging.debug(f"{self.name} is monitoring the result.")
        pass

    def reflect(self, result: str):
        """
        Reflects on the result to derive lessons learned.

        Args:
            result (str): The result to reflect upon.
        """
        prompt = f"Reflect on the following result and note any lessons learned for the main task '{self.input}':\n{result}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)

    def learn(self, result: str):
        """
        Placeholder for implementing learning logic based on the result.

        Args:
            result (str): The result to learn from.
        """
        pass

    def communicate(self, result: str) -> str:
        """
        Prepares the final output to be communicated.

        Args:
            result (str): The result to communicate.

        Returns:
            str: The final output.
        """
        return result

    def generate_thoughts(self, input_data: str) -> List[str]:
        """
        Generates multiple thoughts or ideas based on the input data.

        Args:
            input_data (str): The input data to base thoughts on.

        Returns:
            List[str]: A list of generated thoughts.
        """
        prompt = f"{self.prompt}\nGenerate several thoughts or ideas based on the following input for the main task '{self.input}':\n{input_data}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages, format="json")
        thoughts = response.strip().split("\n")
        self.memory.add_message("assistant", response)
        return thoughts

    def single_thought_process(self, **kwargs) -> str:
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
            self.fluid_input += f"\n\n{self.extra_info}"
        single_thought_messages.append({"role": "user", "content": self.fluid_input})
        self.memory.add_message("user", self.fluid_input)
        print(f"\n\nAgent: {self.name}\n")
        print(f"\nUser: {self.fluid_input}\n")

        response = self.model.generate_response(single_thought_messages, **extra_args)
        print(f"Assistant: {response}\n")
        self.memory.add_message("assistant", response)
        return response

    def respond(self, input_data: str) -> str:
        """
        Facilitates the agent's response to input data.

        Args:
            input_data (str): The input data to respond to.

        Returns:
            str: The agent's response.
        """
        return self.process(input_data)

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
    Specialized agent that supervises the workflow, managing the sequence of agents and evaluating their outputs.
    """

    end_agent: Agent = None
    start_agent: Agent = None
    agents: List[Agent] = field(default_factory=list)

    def choose_next_agent(self, current_agent: Agent, output: str) -> Agent:
        """
        Determines the next agent in the workflow based on the current agent's output.

        Args:
            current_agent (Agent): The agent that just completed processing.
            output (str): The output from the current agent.

        Returns:
            Agent: The next agent to process the output. Defaults to end_agent if no further agents.
        """
        try:
            if current_agent == self.start_agent:
                return self.agents[0]
            else:
                index = self.agents.index(current_agent)
                return self.agents[index + 1]
        except (ValueError, IndexError):
            return self.end_agent

    def evaluate_thoughts(self, thoughts: List[str]) -> List[str]:
        """
        Evaluates a list of thoughts and selects the best ones.

        Args:
            thoughts (List[str]): The list of thoughts to evaluate.

        Returns:
            List[str]: The selected best thoughts.
        """
        prompt = f"Evaluate the following thoughts and select the best ones for the main task '{self.input}'and return just then separated by ',':\n{thoughts}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        best_thoughts = response.strip().split(
            ","
        )  # Assuming thoughts are separated by commas
        self.memory.add_message("assistant", response)
        return best_thoughts

    def combine_outputs(self, outputs: List[str]) -> str:
        """
        Combines multiple outputs into a coherent final result.

        Args:
            outputs (List[str]): The list of outputs to combine.

        Returns:
            str: The combined final result.
        """
        logging.debug(f"{self.name} is combining outputs.")
        prompt = f"Combine the following outputs into a coherent result for the main task '{self.input}':\n{outputs}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response


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
