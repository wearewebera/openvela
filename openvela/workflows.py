import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from openvela.agents import (
    Agent,
    EndAgent,
    FluidAgent,
    FluidValidator,
    StartAgent,
    SupervisorAgent,
)
from openvela.memory import AgentMemory, JsonMemoryFormat, WorkflowMemory
from openvela.tasks import Task


class Workflow(ABC):
    """
    Abstract base class representing a generic workflow.
    Defines the structure and essential components required to execute a workflow.
    """

    def __init__(
        self,
        task: Task,
        agents: List[Agent],
        supervisor: SupervisorAgent,
        start_agent: Optional[StartAgent],
        end_agent: Optional[EndAgent],
        subworkflows: Optional[List["Workflow"]] = None,
    ):
        """
        Initializes the Workflow instance.

        Args:
            task (Task): The task to be executed within the workflow.
            agents (List[Agent]): A list of intermediary agents involved in the workflow.
            supervisor (SupervisorAgent): The supervisor agent overseeing the workflow.
            start_agent (Optional[StartAgent]): The agent that initiates the workflow.
            end_agent (Optional[EndAgent]): The agent that concludes the workflow.
            subworkflows (Optional[List["Workflow"]], optional): A list of sub-workflows. Defaults to None.
        """
        self.task = task
        self.agents = agents
        self.supervisor = supervisor
        self.start_agent = start_agent
        self.end_agent = end_agent
        self.subworkflows = subworkflows if subworkflows else []
        self.memory_id = str(uuid.uuid4())
        # Initialize WorkflowMemory with JsonMemoryFormat as default
        self.memory = WorkflowMemory(
            memory_id=self.memory_id,
            memory_format=JsonMemoryFormat(),
        )

        # Initialize AgentMemory to store agents' info
        self.agent_memory = AgentMemory(
            file_path=f".openvela/{self.memory_id}_agents_info.json",
            memory_format=JsonMemoryFormat(),
        )

        # Store agents' info using AgentMemory
        for agent in self.agents + [self.start_agent, self.end_agent, self.supervisor]:
            if agent is not None:
                self.agent_memory.add_agent_info(agent.name, agent.prompt)

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Executes the workflow.

        Must be implemented by subclasses to define specific workflow execution logic.

        Returns:
            str: The final output of the workflow.
        """
        pass


class ChainOfThoughtWorkflow(Workflow):
    """
    Implements a Chain of Thought workflow where agents process data sequentially.
    Each agent processes the output of the previous one, culminating in the end agent.
    """

    def __init__(
        self,
        task: Task,
        agents: List[Agent],
        supervisor: SupervisorAgent,
        start_agent: Optional[StartAgent],
        end_agent: Optional[EndAgent],
        subworkflows: Optional[List["Workflow"]] = None,
        validate_output: bool = False,
        max_attempts: int = 3,
        validator: Optional[FluidValidator] = None,
    ):
        """
        Initializes the ChainOfThoughtWorkflow instance.

        Args:
            task (Task): The task to be executed within the workflow.
            agents (List[Agent]): A list of intermediary agents involved in the workflow.
            supervisor (SupervisorAgent): The supervisor agent overseeing the workflow.
            start_agent (Optional[StartAgent]): The agent that initiates the workflow.
            end_agent (Optional[EndAgent]): The agent that concludes the workflow.
            subworkflows (Optional[List["Workflow"]], optional): A list of sub-workflows. Defaults to None.
            validate_output (bool, optional): Whether to validate the output and remake the loop if invalid. Defaults to False.
            max_attempts (int, optional): Maximum number of attempts to get a valid output. Defaults to 3.
            validator (Optional[FluidValidator], optional): The validator agent to use. Defaults to None.
        """
        super().__init__(task, agents, supervisor, start_agent, end_agent, subworkflows)
        self.validate_output = validate_output
        self.max_attempts = max_attempts
        if validator is None and validate_output:
            self.validator = FluidValidator(model=self.agents[0].model, settings={})
        else:
            self.validator = validator
        self.final_output = ""

    def run(self, **kwargs) -> str:
        """
        Executes the Chain of Thought workflow.

        The workflow starts with the start agent, processes input through intermediary agents,
        and concludes with the end agent. If validation is enabled, it will validate the final output
        and remake the loop if the output is invalid, up to the maximum number of attempts.

        Returns:
            str: The final output after processing through all agents.
        """
        logging.info("Starting ChainOfThoughtWorkflow.")

        if self.validate_output:
            max_iterations = self.max_attempts  # Prevent infinite loops
            iteration = 0
            while iteration < max_iterations:
                iteration += 1
                self.memory.clear_memory()
                current_agent = self.start_agent
                current_input = self.task.prompt

                self.memory.add_message("user", current_input)

                while current_agent != self.end_agent:
                    logging.debug(f"Current agent: {current_agent.name}")
                    current_agent.fluid_input = current_input
                    # Agent responds using their own process method
                    output = current_agent.single_thought_process(**kwargs)
                    self.memory.add_message("assistant", output)

                    # The next agent's input is the previous agent's output
                    current_agent = self.supervisor.choose_next_agent(
                        current_agent, output
                    )
                    current_input = current_agent.fluid_input
                    if current_agent.fluid_input == "":
                        current_input = output
                    # Add the new user input
                    self.memory.add_message("user", current_input)

                # End agent responds using all messages
                logging.debug(f"Current agent: {current_agent.name}")
                self.end_agent.fluid_input = current_input
                self.final_output = self.end_agent.single_thought_process(**kwargs)
                self.memory.add_message("assistant", self.final_output)

                # Validate the output
                validate_output = self.validator.validate_output(
                    task_description=self.task.prompt, answer=self.final_output
                )
                print(validate_output)
                if validate_output.get("valid"):
                    # Output is valid, exit the loop
                    return self.final_output, self.memory_id
                else:
                    # Incorporate feedback into the task prompt

                    logging.info(
                        f"Iteration {iteration}: Feedback added to task prompt."
                    )
            # If maximum iterations are reached without valid output
            logging.warning("Maximum iterations reached without valid output.")
            return self.final_output, self.memory_id
        else:
            # Original behavior without validation
            current_agent = self.start_agent
            current_input = self.task.prompt

            self.memory.add_message("user", current_input)

            while current_agent != self.end_agent:
                logging.debug(f"Current agent: {current_agent.name}")
                current_agent.fluid_input = current_input
                # Agent responds using their own process method
                output = current_agent.single_thought_process(**kwargs)
                self.memory.add_message("assistant", output)

                # The next agent's input is the previous agent's output
                current_agent = self.supervisor.choose_next_agent(current_agent, output)
                current_input = current_agent.fluid_input
                if current_agent.fluid_input == "":
                    current_input = output
                # Add the new user input
                self.memory.add_message("user", current_input)

            # End agent responds using all messages
            logging.debug(f"Current agent: {current_agent.name}")
            self.end_agent.fluid_input = current_input
            final_output = self.end_agent.single_thought_process(**kwargs)
            self.memory.add_message("assistant", final_output)

            return final_output, self.memory_id


class TreeOfThoughtWorkflow(Workflow):
    """
    Implements a Tree of Thought workflow where multiple thoughts are generated and evaluated,
    allowing for parallel processing and selection of the best paths.
    """

    def run(self, **kwargs) -> str:
        """
        Executes the Tree of Thought workflow.

        The workflow generates multiple thoughts from the start agent, evaluates them,
        processes each selected thought, and combines the outputs into a final result.

        Returns:
            str: The final combined output after processing selected thoughts.
        """
        logging.info("Starting TreeOfThoughtWorkflow.")
        thoughts = []
        current_agent = self.start_agent
        current_input = self.task.prompt

        # Generate initial thoughts
        outputs = current_agent.generate_thoughts(current_input)
        thoughts.extend(outputs)

        # Supervisor evaluates thoughts
        best_thoughts = self.supervisor.evaluate_thoughts(thoughts)
        final_outputs = []

        for thought in best_thoughts:
            self.memory.add_message("assistant", thought)

            # Process thought with subworkflows or end agent
            if self.subworkflows:
                for subworkflow in self.subworkflows:
                    final_outputs.append(subworkflow.run())
            else:
                final_response = self.end_agent.respond(thought)
                self.memory.add_message("assistant", final_response)
                final_outputs.append(final_response)

        # Combine outputs
        final_output = self.supervisor.combine_outputs(final_outputs)
        return final_output, self.memory_id


class FluidChainOfThoughtWorkflow(Workflow):
    """
    Implements a Fluid Chain of Thought workflow where agents are dynamically generated based on the task.
    Facilitates flexible and adaptive processing by creating agents on-the-fly.
    """

    def __init__(
        self,
        task: Task,
        fluid_agent: FluidAgent,
        supervisor: SupervisorAgent,
        start_agent: Optional[StartAgent] = None,
        end_agent: Optional[EndAgent] = None,
        subworkflows: Optional[List["Workflow"]] = None,
        max_attempts: int = 3,
        max_previous_messages: int = None,
    ):
        """
        Initializes the FluidChainOfThoughtWorkflow instance.

        Args:
            task (Task): The task to be executed within the workflow.
            fluid_agent (FluidAgent): The agent responsible for generating other agents based on the task.
            supervisor (SupervisorAgent): The supervisor agent overseeing the workflow.
            start_agent (Optional[StartAgent], optional): The agent that initiates the workflow. Defaults to None.
            end_agent (Optional[EndAgent], optional): The agent that concludes the workflow. Defaults to None.
            subworkflows (Optional[List["Workflow"]], optional): A list of sub-workflows. Defaults to None.
        """
        super().__init__(task, [], supervisor, start_agent, end_agent, subworkflows)
        self.fluid_agent = fluid_agent
        self.agents = []
        self.final_output = ""
        self.validator = FluidValidator(model=fluid_agent.model, settings={})
        self.max_attempts = max_attempts
        self.max_previous_messages = max_previous_messages

    def run(self, **kwargs) -> str:
        logging.info("Starting FluidChainOfThoughtWorkflow.")
        max_iterations = self.max_attempts  # Prevent infinite loops
        iteration = 0
        kwargs.update({"max_previous_messages": self.max_previous_messages})
        while iteration < max_iterations:
            iteration += 1
            # Generate agents from the (possibly updated) task prompt
            agents_definitions = self.fluid_agent.generate_agents_from_task(
                self.task.prompt, **kwargs
            )
            self.agents = self.fluid_agent.create_agents(
                agents_definitions, memory_id=self.memory_id
            )
            # Update the supervisor's agents list
            self.supervisor.agents = self.agents
            agents_quantity = len(self.agents)
            current_agent = self.agents[0]
            if self.start_agent:
                agents_quantity += 1
                if not self.start_agent.fluid_input:
                    self.start_agent.fluid_input = self.task.prompt
                current_agent = self.start_agent
            current_input = self.task.prompt
            self.count = 0
            print(self.agents)
            current_agent.memory_id = self.memory_id
            while self.count != agents_quantity:
                if current_agent != self.start_agent:
                    current_agent = self.agents[self.count]
                # Agent responds using their own process method
                output = current_agent.single_thought_process(**kwargs)
                # The next agent's input is the previous agent's output
                self.final_output = output
                current_agent = self.supervisor.choose_next_agent(current_agent, output)
                self.count += 1
            if self.end_agent:
                self.end_agent.memory_id = self.memory_id
                self.final_output = self.end_agent.single_thought_process(**kwargs)
            validate_output = self.validator.validate_output(
                task_description=self.task.prompt, answer=self.final_output
            )
            print(validate_output)
            if validate_output.get("valid"):
                # Output is valid, exit the loop
                return self.final_output, self.memory_id
            else:
                # Incorporate feedback into the task prompt
                feedback = validate_output.get("feedback", "")
                self.task.prompt += f"\nThis is a new run of the task into this workflow, use the feedback provider by the validator agent\nFeedback: {feedback}"
                self.memory.clear_memory()
                logging.info(f"Iteration {iteration}: Feedback added to task prompt.")
        # If maximum iterations are reached without valid output
        logging.warning("Maximum iterations reached without valid output.")
        return self.final_output, self.memory_id
