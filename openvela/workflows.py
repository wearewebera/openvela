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
        start_agent: Optional[StartAgent] = None,
        end_agent: Optional[EndAgent] = None,
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
        self.set_memory_on_agents()
        self.set_supervisor_memory()

    def set_memory_on_agents(self):
        for agent in self.agents:
            agent.set_memory(self.memory)

    def set_supervisor_memory(self):
        self.supervisor.set_memory(self.memory)

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
                    current_agent.input = current_input
                    # Agent responds using their own process method
                    output = current_agent.generate(**kwargs)
                    self.memory.add_message("assistant", output)

                    # The next agent's input is the previous agent's output
                    current_agent = self.supervisor.choose_next_agent(
                        current_agent, output
                    )
                    current_input = current_agent.input
                    if current_agent.input == "":
                        current_input = output
                    # Add the new user input
                    self.memory.add_message("user", current_input)

                # End agent responds using all messages
                logging.debug(f"Current agent: {current_agent.name}")
                self.end_agent.input = current_input
                self.final_output = self.end_agent.generate(**kwargs)
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
                current_agent.input = current_input
                # Agent responds using their own process method
                output = current_agent.generate(**kwargs)
                self.memory.add_message("assistant", output)

                # The next agent's input is the previous agent's output
                current_agent = self.supervisor.choose_next_agent(current_agent, output)
                current_input = current_agent.input
                if current_agent.input == "":
                    current_input = output
                # Add the new user input
                self.memory.add_message("user", current_input)

            # End agent responds using all messages
            logging.debug(f"Current agent: {current_agent.name}")
            self.end_agent.input = current_input
            final_output = self.end_agent.generate(**kwargs)
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


class AutoSelectWorkflow(Workflow):
    def __init__(
        self,
        task: Task,
        agents: List[Agent],
        supervisor: SupervisorAgent,
        subworkflows: Optional[List["Workflow"]] = None,
        validate_output: bool = False,
        max_attempts: int = 3,
        validator: Optional[FluidValidator] = None,
    ):
        """
        Initializes the AutoSelectWorkflow instance.

        Args:
            task (Task): The task to be executed within the workflow.
            agents (List[Agent]): A list of intermediary agents involved in the workflow.
            supervisor (SupervisorAgent): The supervisor agent overseeing the workflow.
            subworkflows (Optional[List["Workflow"]], optional): A list of sub-workflows. Defaults to None.
            validate_output (bool, optional): Whether to validate the output and remake the loop if invalid. Defaults to False.
            max_attempts (int, optional): Maximum number of attempts to get a valid output. Defaults to 3.
            validator (Optional[FluidValidator], optional): The validator agent to use. Defaults to None.
        """
        super().__init__(task, agents, supervisor, subworkflows)
        self.validate_output = validate_output
        self.max_attempts = max_attempts
        if validator is None and validate_output:
            self.validator = FluidValidator(model=self.agents[0].model, settings={})
        else:
            self.validator = validator
        self.final_output = ""

    def run(self, **kwargs) -> str:
        logging.info("Starting AutoSelectWorkflow.")
        self.supervisor.agents = self.agents
        self.supervisor.task = self.task.prompt

        # ---------------
        # Validation loop
        # ---------------
        if self.validate_output:
            logging.info("Output validation is enabled. Entering validation loop.")
            max_iterations = self.max_attempts
            iteration = 0

            while iteration < max_iterations:
                iteration += 1
                logging.info(f"Starting iteration {iteration}/{max_iterations}.")

                # Clear memory at each iteration to start fresh
                logging.debug("Memory cleared for new iteration.")

                # For the very first iteration, let the supervisor pick the first agent
                if iteration == 1:
                    logging.debug("First iteration: letting supervisor pick agent.")
                    decision = self.supervisor.choose_next_agent(
                        current_agent=None, latest_output="Starting new iteration"
                    )
                else:
                    logging.debug("Re-iterating after invalidation.")
                    decision = self.supervisor.choose_next_agent(
                        current_agent=None,
                        latest_output="Re-iterating after invalidation",
                    )
                    logging.info(
                        f"Supervisor chose agent: {decision['next_agent']} with input: {decision['next_input']}"
                    )
                next_agent_name = decision["next_agent"]
                next_input = decision["next_input"]
                logging.info(
                    f"Supervisor chose agent: {next_agent_name} with input: {next_input}"
                )

                # If supervisor says FINISH right away, break
                if next_agent_name == "FINISH":
                    logging.info("Supervisor indicated FINISH immediately.")
                    self.final_output = next_input
                    break

                # Otherwise, run the selected agent chain
                output_dict = self._run_agent_chain(
                    next_agent_name, next_input, **kwargs
                )
                logging.info(f"Agent chain returned: {output_dict}")

                # If the final output for that chain is "FINISH," or the supervisor chooses FINISH
                if output_dict["next_agent"] == "FINISH":
                    self.final_output = output_dict["output"]
                else:
                    self.final_output = output_dict["output"]

                # Validate the output
                logging.debug(f"Validating output: {self.final_output}")
                validation_result = self.validator.validate_output(
                    task_description=self.task.prompt, answer=self.final_output
                )

                if validation_result.get("valid"):
                    logging.info("Output is valid. Exiting validation loop.")
                    return self.final_output
                else:
                    logging.warning(f"Iteration {iteration}: Output invalid. Retrying.")
                    continue

            # If we exit the while due to max iterations
            logging.warning("Maximum iterations reached without valid output.")
            return self.final_output

        # ------------------------------------------------
        # If no validation is requested, do simpler logic
        # ------------------------------------------------
        else:

            logging.info("Validation disabled. Running single agent chain flow.")
            decision = self.supervisor.choose_next_agent(None, "Workflow started.")
            next_agent_name = decision["next_agent"]
            next_input = decision["next_input"]

            print(f"Supervisor chose agent: {next_agent_name} with input: {next_input}")
            # logging.info(
            #     f"Supervisor chose agent: {next_agent_name} with input: {next_input}"
            # )

            # If it says FINISH right away, we are done
            if next_agent_name == "FINISH":
                logging.info(
                    "Supervisor indicated FINISH immediately. Returning final output."
                )
                return next_input

            # Otherwise, proceed with agent chain
            output_dict = self._run_agent_chain(next_agent_name, next_input, **kwargs)

            # logging.info(f"Agent chain returned: {output_dict}")

            if output_dict["next_agent"] == "FINISH":
                return output_dict["output"]
            else:
                return output_dict["output"]

    def _run_agent_chain(
        self, start_agent_name: str, agent_input: str, **kwargs
    ) -> dict:
        """
        Helper method to run a chain of agents under supervisor direction
        until a FINISH or an exception occurs. Returns a dict:
           {
             "next_agent": "FINISH" or <AgentName>,
             "output": <agent's final output>
           }
        """

        current_agent = [a for a in self.agents if a.name == start_agent_name]

        if not current_agent:
            logging.warning(
                f"No agent found named {start_agent_name}. Finishing immediately."
            )
            return {"next_agent": "FINISH", "output": agent_input}
        current_agent = current_agent[0]

        # Clear memory for the chain run

        logging.debug("Memory cleared at the start of agent chain.")

        while True:
            # Let the current agent process

            current_agent.input = agent_input
            output = current_agent.generate(**kwargs)

            # Check if the agent decided to FINISH
            if output == "FINISH":
                logging.info(f"Agent '{current_agent.name}' indicated FINISH.")
                return {"next_agent": "FINISH", "output": agent_input}

            # Otherwise, let the supervisor pick the next agent
            decision = self.supervisor.choose_next_agent(current_agent, output)
            next_agent_name = decision["next_agent"]
            next_input = decision["next_input"]
            logging.info(
                f"Supervisor chose next agent: {next_agent_name} with input: {next_input}"
            )

            if next_agent_name == "FINISH":
                logging.info("Supervisor indicated FINISH.")
                return {"next_agent": "FINISH", "output": next_input}

            # Switch to the chosen agent
            next_agent = [a for a in self.agents if a.name == next_agent_name]
            if not next_agent:
                logging.warning(f"No agent found named {next_agent_name}. Finishing.")
                return {"next_agent": "FINISH", "output": next_input}
            current_agent = next_agent[0]

            # Update agent_input for the next iteration
            agent_input = next_input

            # Update memory


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
                if not self.start_agent.input:
                    self.start_agent.input = self.task.prompt
                current_agent = self.start_agent
            current_input = self.task.prompt
            self.count = 0

            current_agent.memory_id = self.memory_id
            while self.count != agents_quantity:
                if current_agent != self.start_agent:
                    current_agent = self.agents[self.count]
                # Agent responds using their own process method
                output = current_agent.generate(**kwargs)
                # The next agent's input is the previous agent's output
                self.final_output = output
                current_agent = self.supervisor.choose_next_agent(current_agent, output)
                self.count += 1
            if self.end_agent:
                self.end_agent.memory_id = self.memory_id
                self.final_output = self.end_agent.generate(**kwargs)
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
