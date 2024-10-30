import logging

# Import classes from your library
from agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
from logs import configure_logging
from memory import JsonReader
from tasks import Task
from workflows import (
    ChainOfThoughtWorkflow,
    FluidChainOfThoughtWorkflow,
    TreeOfThoughtWorkflow,
)


def test_chain_of_thought_workflow():

    # Configure logging
    configure_logging()

    # Define agents with specific prompts
    start_agent = StartAgent(
        settings={
            "name": "StartAgent",
            "prompt": "You are the StartAgent. Begin the task by introducing the main character.",
        }
    )

    middle_agent = Agent(
        settings={
            "name": "MiddleAgent",
            "prompt": "You are the MiddleAgent. Develop the plot by introducing challenges.",
        }
    )

    end_agent = EndAgent(
        settings={
            "name": "EndAgent",
            "prompt": "You are the EndAgent. Conclude the story by resolving the challenges.",
        }
    )

    supervisor_agent = SupervisorAgent(
        settings={
            "name": "SupervisorAgent",
            "prompt": "Ensure the story flows logically and each agent contributes appropriately.",
        },
        start_agent=start_agent,
        end_agent=end_agent,
        agents=[middle_agent],
    )

    # Define the task
    task = Task(
        agents=[
            "StartAgent",
            "MiddleAgent",
            "EndAgent",
        ],
        prompt="Create a story about a brave knight on a quest to find a mythical treasure.",
    )

    # Create the workflow
    workflow = ChainOfThoughtWorkflow(
        task=task,
        agents=[middle_agent],
        supervisor=supervisor_agent,
        start_agent=start_agent,
        end_agent=end_agent,
    )

    # Run the workflow
    final_output = workflow.run()
    with open("chain_result.txt", "w") as f:
        f.write(final_output)
        f.close()
    # Optionally, print the message history

    # Use JsonReader to load the data
    json_reader = JsonReader(file_path=".openvela/workflow_memory.json")
    data_dict = json_reader.json_to_dict()


def test_tree_of_thought_workflow():

    # Configure logging
    configure_logging()

    # Define agents
    start_agent = StartAgent(
        settings={
            "name": "StartAgent",
            "prompt": "You are the StartAgent. Propose several ideas for a new app.",
        }
    )

    end_agent = EndAgent(
        settings={
            "name": "EndAgent",
            "prompt": "You are the EndAgent. Select the best idea and elaborate on it.",
        }
    )

    supervisor_agent = SupervisorAgent(
        settings={
            "name": "SupervisorAgent",
            "prompt": "Evaluate the proposed ideas and select the most promising ones.",
        },
        start_agent=start_agent,
        end_agent=end_agent,
    )

    # Define the task
    task = Task(
        agents=[
            "StartAgent",
            "EndAgent",
        ],
        prompt="Brainstorm ideas for a mobile app that can help improve productivity.",
    )

    # Create the workflow
    workflow = TreeOfThoughtWorkflow(
        task=task,
        agents=[],
        supervisor=supervisor_agent,
        start_agent=start_agent,
        end_agent=end_agent,
    )

    # Run the workflow
    final_output = workflow.run()

    # Optionally, print the message history


def test_fluid_chain_of_thought_workflow():

    # Configure logging
    configure_logging()

    # Define the task
    task_description = (
        """Create a roadmap to learn Python programming language on it's maximum."""
    )

    task = Task(
        agents=[
            "StartAgent",
            "EndAgent",
        ],
        prompt=task_description,
    )

    # Create a FluidAgent
    fluid_agent = FluidAgent(settings={"name": "FluidAgent"})

    # Define start and end agents
    start_agent = StartAgent(
        settings={
            "name": "StartAgent",
            "prompt": "You are responsible for improve the task clarity and return a task description.\nPlease provide an overview of the task. \n Please provide an step by step of how to finish the task, from beggining to end.",
        }
    )

    end_agent = EndAgent(
        settings={
            "name": "EndAgent",
            "prompt": f"Based on previous messages, you are responsible for providing the final output. \n Please provide the most complete and accurate answer. The final output contains All the information of the conversation that is related with the main task\n ",
            "input": "Based on our conversation give me the final output for the task.",
        }
    )

    supervisor_agent = SupervisorAgent(
        settings={
            "name": "SupervisorAgent",
            "prompt": "Oversee the workflow and ensure all aspects are covered.",
        },
        start_agent=start_agent,
        end_agent=end_agent,
    )

    # Create the FluidChainOfThoughtWorkflow
    workflow = FluidChainOfThoughtWorkflow(
        task=task,
        fluid_agent=fluid_agent,
        supervisor=supervisor_agent,
    )

    # Run the workflow
    final_output, id = workflow.run()

    with open("fluid_result.txt", "w") as f:
        f.write(final_output)
        f.close()
    # Optionally, print the message history


def main():
    # # Test ChainOfThoughtWorkflow
    # test_chain_of_thought_workflow()

    # # Test TreeOfThoughtWorkflow
    # test_tree_of_thought_workflow()

    # Test FluidChainOfThoughtWorkflow
    test_fluid_chain_of_thought_workflow()


if __name__ == "__main__":
    main()
