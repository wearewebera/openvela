# __main__.py

import argparse
import getpass
import logging
from typing import Optional

# Import classes from your library
from agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
from llms import GroqModel, Model, OllamaModel, OpenAIModel  # Ensure Model is imported
from logs import configure_logging
from memory import JsonReader
from tasks import Task
from workflows import FluidChainOfThoughtWorkflow


def get_model_instance(
    model_choice: str, api_key: Optional[str] = None, url: Optional[str] = None
) -> Model:
    """
    Instantiate and return the selected model with provided credentials.

    Args:
        model_choice (str): The model to instantiate. Choices are 'openai', 'groq', or 'ollama'.
        api_key (Optional[str]): API key for OpenAI or Groq models. If None, prompts the user securely.
        url (Optional[str]): URL for Ollama model. Defaults to 'http://localhost:11434/' if None.

    Returns:
        Model: An instance of the selected model.

    Raises:
        ValueError: If an unsupported model_choice is provided.
    """
    if model_choice.lower() == "openai":
        if not api_key:
            api_key = getpass.getpass(prompt="Enter your OpenAI API Key: ")
        return OpenAIModel(api_key=api_key)
    elif model_choice.lower() == "groq":
        if not api_key:
            api_key = getpass.getpass(prompt="Enter your Groq API Key: ")
        return GroqModel(api_key=api_key)
    elif model_choice.lower() == "ollama":
        if not url:
            url = (
                input("Enter Ollama URL (default: http://localhost:11434/): ")
                or "http://localhost:11434/"
            )
        return OllamaModel(
            host=url.split("://")[-1].split(":")[0]
        )  # Extract host from URL
    else:
        raise ValueError(
            "Unsupported model choice. Please select from OpenAI, Groq, or Ollama."
        )


def test_fluid_chain_of_thought_workflow(
    model: Model, task_description: str, verbose: bool
):
    """
    Set up and run the FluidChainOfThoughtWorkflow with the selected model and task.

    Args:
        model (Model): The instantiated language model to use.
        task_description (str): The description of the task to execute.
        verbose (bool): Flag to enable verbose logging.
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)

    # Define the task
    task = Task(
        agents=[
            "StartAgent",
            "EndAgent",
        ],
        prompt=task_description,
    )

    # Create a FluidAgent
    fluid_agent = FluidAgent(
        settings={
            "name": "FluidAgent",
            "prompt": "You are a FluidAgent that generates specialized agents based on the task.",
        }
    )

    # Define start and end agents
    start_agent = StartAgent(
        settings={
            "name": "StartAgent",
            "prompt": "Please provide an overview of the task. \n Please provide a step-by-step guide on how to finish the task, from beginning to end.",
        }
    )

    end_agent = EndAgent(
        settings={
            "name": "EndAgent",
            "prompt": (
                "Based on previous messages, you are responsible for providing the final output. \n"
                "Please provide the most complete and accurate answer. The final output should contain all the information related to the main task."
            ),
            "input": "Based on our conversation, give me the final output for the task.",
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
        start_agent=start_agent,
        end_agent=end_agent,
    )

    # Assign the selected model to all agents
    for agent in [fluid_agent, supervisor_agent, start_agent, end_agent]:
        agent.model = model

    # Enable verbose logging within the workflow if needed
    workflow.verbose = verbose  # Assuming Workflow supports a verbose attribute

    # Run the workflow
    final_output = workflow.run()
    print("Final Strategy:\n")
    print(final_output)

    # Save the result to a file if needed
    with open("fluid_result.txt", "w") as f:
        f.write(final_output)

    # Optionally, print the message history
    print("\nMessage History:")
    for message in workflow.memory.messages:
        print(f"{message['role']}: {message['content']}")


def main():
    # Configure argument parser
    parser = argparse.ArgumentParser(
        description="Run FluidChainOfThoughtWorkflow with selected LLM model."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["openai", "groq", "ollama"],
        required=True,
        help="Choose the LLM model to use: openai, groq, or ollama.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="API key for OpenAI or Groq models.",
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="URL for Ollama model. Defaults to localhost if not provided.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="Task description. If not provided, you will be prompted to enter it.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging to see detailed agent inputs and outputs.",
    )

    args = parser.parse_args()

    # Configure logging
    configure_logging()

    # Get model instance
    try:
        model = get_model_instance(
            model_choice=args.model, api_key=args.api_key, url=args.url
        )
    except ValueError as e:
        logging.error(e)
        return

    # Get task description
    if not args.task:
        task_description = input("Enter the task description: ")
    else:
        task_description = args.task

    # Run the workflow
    test_fluid_chain_of_thought_workflow(
        model=model, task_description=task_description, verbose=args.verbose
    )


if __name__ == "__main__":
    main()
