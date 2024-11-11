# __main__.py

import json
import logging
import os
import sys

from openvela.agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
from openvela.llms import GroqModel, OllamaModel, OpenAIModel
from openvela.logs import configure_logging
from openvela.memory import JsonReader
from openvela.tasks import Task
from openvela.workflows import (
    ChainOfThoughtWorkflow,
    FluidChainOfThoughtWorkflow,
    TreeOfThoughtWorkflow,
)


def example_json_file():
    """
    Provides an example JSON structure for defining agents in Chain or Tree workflows.
    """
    example = {
        "agents": [
            {
                "name": "StartAgent",
                "prompt": "You are the StartAgent. Begin the task by introducing the main character.",
                "input": "Initial task description or user input to begin the workflow.",
            },
            {
                "name": "MiddleAgent",
                "prompt": "You are the MiddleAgent. Develop the plot by introducing challenges.",
                "input": "Simulated question or directive that MiddleAgent uses to enhance the workflow.",
            },
            {
                "name": "EndAgent",
                "prompt": "You are the EndAgent. Conclude the story by resolving the challenges.",
                "input": "Final directive to synthesize and deliver the completed task outcome.",
            },
        ]
    }
    print("\nExample JSON file structure for Chain or Tree of Thought workflows:")
    print(json.dumps(example, indent=4))


def main():
    """
    The main entry point for the OpenVela interactive workflow interface.
    Guides the user through selecting workflow types, providers, setting configurations,
    loading agent definitions, inputting tasks, and executing workflows.
    """
    # Configure logging
    configure_logging()

    print("========================================")
    print("      Welcome to OpenVela Interface     ")
    print("========================================\n")

    # Step 1: Select Workflow Type
    workflow_types = {
        "1": "Chain of Thought",
        "2": "Tree of Thought",
        "3": "Fluid Chain of Thought",
    }
    print("Select the type of workflow:")
    for key, value in workflow_types.items():
        print(f"{key}. {value}")

    while True:
        workflow_choice = input(
            "\nEnter the number corresponding to your choice: "
        ).strip()
        if workflow_choice in workflow_types:
            workflow_type = workflow_types[workflow_choice]
            break
        else:
            print("Invalid choice. Please try again.")

    # Step 2: Select Provider
    providers = {"1": "groq", "2": "ollama", "3": "openai"}
    print("\nSelect the language model provider:")
    for key, value in providers.items():
        print(f"{key}. {value.capitalize()}")

    while True:
        provider_choice = input(
            "\nEnter the number corresponding to your choice: "
        ).strip()
        if provider_choice in providers:
            provider = providers[provider_choice]
            break
        else:
            print("Invalid choice. Please try again.")

    # Step 3: Set API Key or Host URL
    if provider in ["groq", "openai"]:
        while True:
            api_key = input(
                f"\nEnter your API key for {provider.capitalize()}: "
            ).strip()
            if api_key:
                break
            else:
                print("API key cannot be empty. Please try again.")
        if provider == "openai":
            model_instance = OpenAIModel(api_key=api_key)
        elif provider == "groq":
            model_instance = GroqModel(api_key=api_key)
    elif provider == "ollama":
        while True:
            host_url = input(
                "\nEnter the host URL for Ollama (e.g., http://localhost:11434): "
            )
            if host_url:
                break
            else:
                print("Host URL cannot be empty. Please try again.")
        model_instance = OllamaModel(base_url=host_url)
    else:
        print("Unsupported provider selected.")
        sys.exit(1)

    print(f"\nSelected Model Provider: {provider.capitalize()}")

    # Step 4: If Chain or Tree, input path to JSON file
    agents_definitions = []
    fluid_agent = None  # Will be initialized if needed
    if workflow_type in ["Chain of Thought", "Tree of Thought"]:
        print(
            "\nProvide the path to the JSON file defining the agents for the workflow."
        )
        print("If you need an example of the JSON structure, type 'example'.")
        while True:
            json_path = input(
                "Enter the path to the agents JSON file (or type 'example'): "
            ).strip()
            if json_path.lower() == "example":
                example_json_file()
                continue
            if not os.path.isfile(json_path):
                print(
                    "File not found. Please enter a valid file path or type 'example' for a sample structure."
                )
                continue
            try:
                with open(json_path, "r") as f:
                    agents_data = json.load(f)
                    agents_definitions = agents_data.get("agents", [])
                    if not agents_definitions:
                        print(
                            "No agents found in the JSON file. Please check the file and try again."
                        )
                        continue
                break
            except json.JSONDecodeError:
                print(
                    "Invalid JSON file. Please ensure the file is properly formatted and try again."
                )
            except Exception as e:
                print(f"An error occurred while reading the file: {e}")

    # Step 5: Create Agents and Supervisor
    agents = []
    start_agent = None
    end_agent = None
    supervisor_agent = None

    if workflow_type in ["Chain of Thought", "Tree of Thought"]:
        # Expecting a list of agents from the JSON
        for agent_def in agents_definitions:
            name = agent_def.get("name")
            prompt = agent_def.get("prompt")
            input_prompt = agent_def.get("input", "")
            if not name or not prompt:
                print(
                    f"Agent definition missing 'name' or 'prompt'. Skipping agent: {agent_def}"
                )
                continue
            if name.lower() == "startagent":
                start_agent = StartAgent(settings=agent_def, model=model_instance)
            elif name.lower() == "endagent":
                end_agent = EndAgent(settings=agent_def, model=model_instance)
            else:
                agents.append(Agent(settings=agent_def, model=model_instance))

        if not start_agent or not end_agent:
            print("\nJSON file must include 'StartAgent' and 'EndAgent'. Exiting.")
            sys.exit(1)

        # Initialize SupervisorAgent
        supervisor_agent = SupervisorAgent(
            settings={
                "name": "SupervisorAgent",
                "prompt": "Oversee the workflow and ensure all aspects are covered.",
            },
            start_agent=start_agent,
            end_agent=end_agent,
            agents=agents,
            model=model_instance,
        )
    elif workflow_type == "Fluid Chain of Thought":
        # For Fluid workflows, dynamically generate agents, skip to next step
        fluid_agent = FluidAgent(settings={"name": "FluidAgent"}, model=model_instance)
        # Initialize SupervisorAgent with no predefined agents
        supervisor_agent = SupervisorAgent(
            settings={
                "name": "SupervisorAgent",
                "prompt": "Oversee the workflow and ensure all aspects are covered.",
            },
            start_agent=None,
            end_agent=None,
            agents=[],
            model=model_instance,
        )
    else:
        print("Unsupported workflow type selected.")
        sys.exit(1)

    # Step 6: Input Task
    print("\nEnter the task description:")
    while True:
        task_description = input(">> ").strip()
        if task_description:
            break
        else:
            print("Task description cannot be empty. Please enter a valid task.")

    # Step 7: Create Task instance
    if workflow_type in ["Chain of Thought", "Tree of Thought"]:
        agent_names = [agent.name for agent in agents] + [
            start_agent.name,
            end_agent.name,
        ]
        task = Task(
            agents=agent_names,
            prompt=task_description,
            agents_path="agents",  # Assuming agents are in 'agents' directory
        )
    elif workflow_type == "Fluid Chain of Thought":
        task = Task(
            agents=[],  # Fluid workflows handle agents dynamically
            prompt=task_description,
            agents_path="agents",
        )

    # Step 8: Initialize Workflow
    if workflow_type == "Chain of Thought":
        workflow = ChainOfThoughtWorkflow(
            task=task,
            agents=agents,
            supervisor=supervisor_agent,
            start_agent=start_agent,
            end_agent=end_agent,
        )
    elif workflow_type == "Tree of Thought":
        workflow = TreeOfThoughtWorkflow(
            task=task,
            agents=agents,
            supervisor=supervisor_agent,
            start_agent=start_agent,
            end_agent=end_agent,
        )
    elif workflow_type == "Fluid Chain of Thought":
        workflow = FluidChainOfThoughtWorkflow(
            task=task,
            fluid_agent=fluid_agent,
            supervisor=supervisor_agent,
        )
    else:
        print("Unsupported workflow type selected.")
        sys.exit(1)

    # Step 9: Run the Workflow
    print("\nRunning the workflow. Please wait...\n")
    try:
        if workflow_type == "Fluid Chain of Thought":
            final_output, memory_id = workflow.run()
            print("========================================")
            print("               Final Output             ")
            print("========================================")
            print(final_output)
            print(f"\nMemory ID: {memory_id}")
            # Optionally, save to a file
            while True:
                save_choice = (
                    input("\nWould you like to save the output to a file? (y/n): ")
                    .strip()
                    .lower()
                )
                if save_choice in ["y", "n"]:
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            if save_choice == "y":
                while True:
                    filename = input(
                        "Enter the filename to save the output (e.g., 'output.txt'): "
                    ).strip()
                    if filename:
                        break
                    else:
                        print(
                            "Filename cannot be empty. Please enter a valid filename."
                        )
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(final_output)
                print(f"Output saved to {filename}")
        else:
            final_output = workflow.run()
            print("========================================")
            print("               Final Output             ")
            print("========================================")
            print(final_output)
            # Optionally, save to a file
            while True:
                save_choice = (
                    input("\nWould you like to save the output to a file? (y/n): ")
                    .strip()
                    .lower()
                )
                if save_choice in ["y", "n"]:
                    break
                else:
                    print("Please enter 'y' for yes or 'n' for no.")
            if save_choice == "y":
                while True:
                    filename = input(
                        "Enter the filename to save the output (e.g., 'output.txt'): "
                    ).strip()
                    if filename:
                        break
                    else:
                        print(
                            "Filename cannot be empty. Please enter a valid filename."
                        )
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(final_output)
                print(f"Output saved to {filename}")
    except Exception as e:
        logging.error(f"An error occurred while running the workflow: {e}")
        print(
            "\nAn error occurred while running the workflow. Please check the logs for more details."
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
