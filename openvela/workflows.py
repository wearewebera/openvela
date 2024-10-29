
import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
from memory import AgentMemory, JsonMemoryFormat, WorkflowMemory
from tasks import Task


class Workflow(ABC):
    def __init__(
        self,
        task: Task,
        agents: List[Agent],
        supervisor: SupervisorAgent,
        start_agent: StartAgent,
        end_agent: EndAgent,
        subworkflows: Optional[List["Workflow"]] = None,
    ):
        self.task = task
        self.agents = agents
        self.supervisor = supervisor
        self.start_agent = start_agent
        self.end_agent = end_agent
        self.subworkflows = subworkflows if subworkflows else []

        # Initialize WorkflowMemory with JsonMemoryFormat as default
        self.memory = WorkflowMemory(
            file_path=".openvela/workflow_memory.json",
            memory_format=JsonMemoryFormat(),
        )

        # Initialize AgentMemory to store agents' info
        self.agent_memory = AgentMemory(
            file_path=".openvela/agents_info.json",
            memory_format=JsonMemoryFormat(),
        )

        # Store agents' info using AgentMemory
        for agent in self.agents + [self.start_agent, self.end_agent, self.supervisor]:
            self.agent_memory.add_agent_info(agent.name, agent.prompt)


        self.verbose = False  # Add verbose attribute


    @abstractmethod
    def run(self) -> str:
        pass


class ChainOfThoughtWorkflow(Workflow):
    def run(self) -> str:
        logging.info("Starting ChainOfThoughtWorkflow.")
        current_agent = self.start_agent
        current_input = self.task.prompt
        self.memory.add_message("user", current_input)

        while current_agent != self.end_agent:
            logging.debug(f"Current agent: {current_agent.name}")

            if self.verbose:
                logging.info(f"Input to {current_agent.name}: {current_input}")

            # Agent responds using their own process method
            output = current_agent.respond(current_input)

            if self.verbose:
                logging.info(f"Output from {current_agent.name}: {output}")

            self.memory.add_message("assistant", output)

            # The next agent's input is the previous agent's output
            current_input = output
            current_agent = self.supervisor.choose_next_agent(current_agent, output)
            # Add the new user input
            self.memory.add_message("user", current_input)

        # End agent responds using all messages
        logging.debug(f"Current agent: {current_agent.name}")
        final_output = self.end_agent.respond(current_input)
        self.memory.add_message("assistant", final_output)


        if self.verbose:
            logging.info(f"Output from {current_agent.name}: {final_output}")


        return final_output


class TreeOfThoughtWorkflow(Workflow):
    def run(self) -> str:
        logging.info("Starting TreeOfThoughtWorkflow.")
        thoughts = []
        current_agent = self.start_agent
        current_input = self.task.prompt

        # Generate initial thoughts
        outputs = current_agent.generate_thoughts(current_input)
        thoughts.extend(outputs)


        if self.verbose:
            logging.info(f"Initial thoughts: {thoughts}")

        # Supervisor evaluates thoughts
        best_thoughts = self.supervisor.evaluate_thoughts(thoughts)

        if self.verbose:
            logging.info(f"Best thoughts after evaluation: {best_thoughts}")


        final_outputs = []

        for thought in best_thoughts:
            self.memory.add_message("assistant", thought)


            if self.verbose:
                logging.info(f"Processing thought: {thought}")

            # Process thought with subworkflows or end agent
            if self.subworkflows:
                for subworkflow in self.subworkflows:
                    final_outputs.append(subworkflow.run())
            else:
                final_response = self.end_agent.respond(thought)
                self.memory.add_message("assistant", final_response)
                final_outputs.append(final_response)


            if self.verbose:
                logging.info(f"Final response: {final_response}")

        # Combine outputs
        final_output = self.supervisor.combine_outputs(final_outputs)
        if self.verbose:
            logging.info(f"Combined final output: {final_output}")

        return final_output


class FluidChainOfThoughtWorkflow(Workflow):
    def __init__(
        self,
        task: Task,
        fluid_agent: FluidAgent,
        supervisor: SupervisorAgent,
        start_agent: StartAgent,
        end_agent: EndAgent,
        subworkflows: Optional[List["Workflow"]] = None,
    ):
        super().__init__(task, [], supervisor, start_agent, end_agent, subworkflows)
        self.fluid_agent = fluid_agent
        self.agents = []
        self.final_output = ""

    def run(self) -> str:
        logging.info("Starting FluidChainOfThoughtWorkflow.")

        current_agent = self.start_agent
        current_input = self.task.prompt
        self.memory.add_message("user", current_input)

        if self.verbose:
            logging.info(f"Initial task input: {current_input}")

        while True:
            logging.debug(f"Current agent: {current_agent.name}")
            if self.verbose:
                logging.info(f"Input to {current_agent.name}: {current_input}")

            # Agent responds using their own process method
            output = current_agent.respond(current_input)

            if self.verbose:
                logging.info(f"Output from {current_agent.name}: {output}")

            self.memory.add_message("assistant", output)

            # Determine the next agent
            next_agent = self.supervisor.choose_next_agent(current_agent, output)

            if next_agent == self.end_agent:
                break

            current_agent = next_agent
            current_input = output
            self.memory.add_message("user", current_input)

        # End agent responds using all messages
        logging.debug(f"Current agent: {self.end_agent.name}")
        if self.verbose:
            logging.info(f"Input to {self.end_agent.name}: {current_input}")
        final_output = self.end_agent.respond(current_input)
        self.memory.add_message("assistant", final_output)

        if self.verbose:
            logging.info(f"Output from {self.end_agent.name}: {final_output}")

        self.final_output = final_output
        return final_output

