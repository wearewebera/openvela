import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llms import GroqModel, Model, OllamaModel
from memory import JsonShortTermMemory
from tools import AIFunctionTool

from openvela.memory import WorkflowMemory


@dataclass
class Agent:
    settings: dict
    model: Model = field(default_factory=GroqModel)
    tools: Optional[List[AIFunctionTool]] = None
    tools_choice: Optional[str] = None
    memory: WorkflowMemory = field(default_factory=lambda: WorkflowMemory())
    name: str = field(init=False)
    prompt: str = field(init=False)

    def __post_init__(self):
        self.name = self.settings.get("name", "Agent")
        self.prompt = self.settings.get("prompt", "")
        self.fluid_input = self.settings.get("input", "")
        self.memory.prompt = self.prompt  # Set the memory prompt
        self.input = ""
        # logging.info(f"{self.name} initialized with prompt: {self.prompt}")

    # Chain of Thought sequence
    def process(self, input_data: str) -> str:
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
        logging.debug(f"{self.name} observes input data.")
        self.input = input_data
        return input_data

    def understand(self, data: str) -> str:
        logging.debug(f"{self.name} is understanding the data.")
        # Use the model to generate an understanding of the data
        messages = self.memory.load()
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def identify_goals(self, context: str) -> List[str]:
        logging.debug(f"{self.name} is identifying goals for the main task")
        # Use the model to identify goals
        prompt = f"Based on the following context, identify the objectives for the main task '{self.input}' for the main task '{self.input}':\n{context}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        goals = response.strip().split(
            "\n"
        )  # Assuming the goals are separated by new lines
        self.memory.add_message("assistant", response)
        return goals

    def retrieve_information(self, goals: List[str]) -> str:
        logging.debug(f"{self.name} is retrieving information.")
        # Use the model to retrieve information relevant to the goals
        prompt = f"Retrieve information relevant to the following goals for the main task '{self.input}':\n{goals}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def plan(self, info: str) -> str:
        logging.debug(f"{self.name} is planning.")
        # Use the model to create a plan
        prompt = f"Based on the following information, create a plan for the main task '{self.input}':\n{info}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def analyze(self, plan: str) -> str:
        logging.debug(f"{self.name} is analyzing the plan.")
        # Use the model to analyze the plan
        prompt = f"Analyze the following plan for effectiveness and potential issues for the main task '{self.input}':\n{plan}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def decide(self, analysis: str) -> str:
        logging.debug(f"{self.name} is making a decision.")
        # Use the model to make a decision
        prompt = f"Based on the analysis, decide on the best course of action for the main task '{self.input}':\n{analysis}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response

    def execute(self, decision: str) -> str:
        logging.debug(f"{self.name} is executing the decision.")
        # Use the model to execute the decision
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
        logging.debug(f"{self.name} is monitoring the result.")
        # Implement monitoring logic if needed
        pass

    def reflect(self, result: str):

        # Use the model to reflect on the result
        prompt = f"Reflect on the following result and note any lessons learned for the main task '{self.input}':\n{result}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)

    def learn(self, result: str):

        # Implement learning logic if needed
        pass

    def communicate(self, result: str) -> str:

        # Return the final result
        return result

    def generate_thoughts(self, input_data: str) -> List[str]:

        prompt = f"{self.prompt}\nGenerate several thoughts or ideas based on the following input for the main task '{self.input}':\n{input_data}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        # Assume that the response contains multiple thoughts separated by new lines
        thoughts = response.strip().split("\n")
        self.memory.add_message("assistant", response)
        return thoughts

    def single_thought_process(self, **kwargs) -> str:

        # Use the model to process a single thought
        single_thought_messages = []
        messages = self.memory.load()

        single_thought_messages.extend(messages)
        single_thought_messages.insert(0, {"role": "system", "content": self.prompt})
        single_thought_messages.append({"role": "user", "content": self.fluid_input})
        self.memory.add_message("user", self.fluid_input)
        print(f"\n\nAgent: {self.name}\n")
        print(f"\nUser: {self.fluid_input}\n")
        response = self.model.generate_response(single_thought_messages, **kwargs)
        print(f"Assistant: {response}\n")
        self.memory.add_message("assistant", response)
        return response

    def respond(self, input_data: str) -> str:
        return self.process(input_data)

    def __eq__(self, other):
        if isinstance(other, Agent):
            return self.name == other.name
        return False

    def __hash__(self):
        return hash(self.name)


@dataclass
class SupervisorAgent(Agent):
    end_agent: Agent = None
    start_agent: Agent = None
    agents: List[Agent] = field(default_factory=list)

    def choose_next_agent(self, current_agent: Agent, output: str) -> Agent:

        try:
            if current_agent == self.start_agent:
                return self.agents[0]
            else:
                index = self.agents.index(current_agent)
                return self.agents[index + 1]
        except (ValueError, IndexError):
            return self.end_agent

    def evaluate_thoughts(self, thoughts: List[str]) -> List[str]:
        # Use the model to evaluate thoughts
        prompt = f"Evaluate the following thoughts and select the best ones for the main task '{self.input}'and return just then separated by ',':\n{thoughts}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        best_thoughts = response.strip().split(
            ","
        )  # Assuming thoughts are separated by new lines
        self.memory.add_message("assistant", response)
        return best_thoughts

    def combine_outputs(self, outputs: List[str]) -> str:
        logging.debug(f"{self.name} is combining outputs.")
        # Use the model to combine outputs if needed
        prompt = f"Combine the following outputs into a coherent result for the main task '{self.input}':\n{outputs}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)
        return response


@dataclass
class StartAgent(Agent):
    pass


@dataclass
class EndAgent(Agent):
    pass


@dataclass
class FluidAgent(Agent):
    def generate_agents_from_task(self, task_description: str) -> List[Dict]:
        logging.debug(f"{self.name} is generating agents from task.")
        # Use the model to generate agent definitions in JSON
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
        - Hard: 7 agents,
        - Medium: 5 agents
        - Easy: 3 agents
        




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
        response = self.model.generate_response(messages, format="json")

        try:
            agents_json = json.loads(response)

            return agents_json.get("agents", [])
        except json.JSONDecodeError:
            logging.error("Failed to decode agents JSON .")
            return []

    def create_agents(
        self, agents_definitions: List[Dict], memory: WorkflowMemory
    ) -> List[Agent]:
        logging.debug(f"{self.name} is creating agents from definitions.")
        agents = []
        for agent_def in agents_definitions:
            agent = Agent(settings=agent_def, memory=memory)
            agents.append(agent)
        return agents
