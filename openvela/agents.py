import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from llms import Model, OllamaModel
from memory import JsonShortTermMemory
from tools import AIFunctionTool

from openvela.memory import WorkflowMemory


@dataclass
class Agent:
    settings: dict
    model: Model = field(default_factory=OllamaModel)
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
        logging.info(f"{self.name} initialized with prompt: {self.prompt}")

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
        logging.debug(f"{self.name} is reflecting on the result.")
        # Use the model to reflect on the result
        prompt = f"Reflect on the following result and note any lessons learned for the main task '{self.input}':\n{result}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        self.memory.add_message("assistant", response)

    def learn(self, result: str):
        logging.debug(f"{self.name} is learning from the result.")
        # Implement learning logic if needed
        pass

    def communicate(self, result: str) -> str:
        logging.debug(f"{self.name} is communicating the result.")
        # Return the final result
        return result

    def generate_thoughts(self, input_data: str) -> List[str]:
        logging.debug(f"{self.name} is generating thoughts.")
        prompt = f"{self.prompt}\nGenerate several thoughts or ideas based on the following input for the main task '{self.input}':\n{input_data}"

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages)
        # Assume that the response contains multiple thoughts separated by new lines
        thoughts = response.strip().split("\n")
        self.memory.add_message("assistant", response)
        return thoughts

    def single_thought_process(self) -> str:
        logging.debug(f"{self.name} is processing a single thought.")
        # Use the model to process a single thought
        single_thought_messages = []
        messages = self.memory.load()

        single_thought_messages.extend(messages)

        single_thought_messages.append({"role": "system", "content": self.prompt})
        single_thought_messages.append({"role": "user", "content": self.fluid_input})
        self.memory.add_message("user", self.fluid_input)
        response = self.model.generate_response(single_thought_messages)
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
        logging.debug(f"{self.name} is choosing the next agent.")
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
        prompt = f"Based on the following task description, generate agent definitions in JSON format for dict('agents': [dict('name': 'name of the agent', 'prompt': it's the system prompt of the agent --it defines the function of the agent within the logic to enhance the final answer, define a context of what the agent is within the workflow min 50 words--, 'input': question that you create to simulate a question that helps find a better answer)]) for the main task:\n{task_description}\n\nRemeber to separate the agents with a comma ','.\nJust return the agents JSON and follow a logic sequence of actions. \n The prompts and inputs should be related to complete the objective of the task. \n analyze the task and generate agents that can help to complete the task. \n The agents should learn from the previous messages and provide a better answer to the next agent. \n All the keys and values are required and strings can not be empty. \n Be detailed in the prompts and inputs to provide a better final output.\n The last agent should be the end agent.\n The end agent is the agent that will provide the final output with the the experience learned from the conversation and the main task. "

        messages = self.memory.load()
        messages.append({"role": "user", "content": prompt})
        response = self.model.generate_response(messages, format="json")
        print(response)
        try:
            agents_json = json.loads(response)
            print(agents_json)
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
