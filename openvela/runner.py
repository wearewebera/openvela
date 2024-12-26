# openvela/runner.py

import json
import logging
import os
import sys


def run_workflow_from_args(args):
    provider = args.provider
    workflow_type = args.workflow_type
    base_url_or_api_key = args.base_url_or_api_key
    model_name = args.model
    options = args.options
    agents_file = args.agents
    task_input = args.task

    # Parse options
    options_dict = {}
    if options:
        try:
            options_dict = json.loads(options)
        except json.JSONDecodeError:
            print("Invalid JSON string for options.")
            sys.exit(1)

    # Load agents if provided
    agents_definitions = []
    if agents_file:
        if not os.path.isfile(agents_file):
            print(f"Agents file {agents_file} not found.")
            sys.exit(1)
        with open(agents_file, "r") as f:
            agents_data = json.load(f)
            agents_definitions = agents_data.get("agents", [])

    # Load task
    if os.path.isfile(task_input):
        with open(task_input, "r") as f:
            task_description = f.read()
    else:
        task_description = task_input

    # Initialize the model based on provider
    if provider == "openai":
        from .llms import OpenAIModel

        model_instance = OpenAIModel(api_key=base_url_or_api_key, model=model_name)
    elif provider == "groq":
        from .llms import GroqModel

        model_instance = GroqModel(api_key=base_url_or_api_key, model=model_name)
    elif provider == "ollama":
        from .llms import OllamaModel

        model_instance = OllamaModel(host=base_url_or_api_key, model=model_name)
    else:
        print(f"Unsupported provider: {provider}")
        sys.exit(1)

    # Initialize the workflow based on workflow_type
    # You will need to implement this function
    output = run_workflow(
        {
            "provider": provider,
            "workflow_type": workflow_type,
            "base_url_or_api_key": base_url_or_api_key,
            "model": model_name, 
            "model_instance": model_instance,
            "options": options_dict,
            "agents_definitions": agents_definitions,
            "task_description": task_description,
        }
    )

    print("========================================")
    print("               Final Output             ")
    print("========================================")
    print(output)


from .llms import GroqModel, OllamaModel, OpenAIModel


def run_workflow(config):
    provider = config["provider"]
    workflow_type = config["workflow_type"]
    base_url_or_api_key = config["base_url_or_api_key"]
    model_name = config["model"]
    options = config["options"]
    agents_definitions = config["agents_definitions"]
    task_description = config["task_description"]

    # Initialize the model instance based on the provider
    if provider == "openai":
        model_instance = OpenAIModel(api_key=base_url_or_api_key, model=model_name)
    elif provider == "groq":
        model_instance = GroqModel(api_key=base_url_or_api_key, model=model_name)
    elif provider == "ollama":
        model_instance = OllamaModel(base_url=base_url_or_api_key, model=model_name)
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    # Implement similar logic as in __main__.py to initialize and run the workflow
    from .agents import Agent, EndAgent, FluidAgent, StartAgent, SupervisorAgent
    from .tasks import Task
    from .workflows import (
        ChainOfThoughtWorkflow,
        FluidChainOfThoughtWorkflow,
        TreeOfThoughtWorkflow,
    )

    if workflow_type in ["cot", "tot"] and not agents_definitions:
        raise ValueError(
            "Agents definitions are required for Chain of Thought and Tree of Thought workflows."
        )

    agents = []
    start_agent = None
    end_agent = None

    if workflow_type in ["cot", "tot"]:
        # Initialize agents
        for agent_def in agents_definitions:
            name = agent_def.get("name")
            if not name:
                continue
            if name.lower() == "startagent":
                start_agent = StartAgent(settings=agent_def, model=model_instance)
            elif name.lower() == "endagent":
                end_agent = EndAgent(settings=agent_def, model=model_instance)
            else:
                agents.append(Agent(settings=agent_def, model=model_instance))

        supervisor_agent = SupervisorAgent(
            settings={"name": "SupervisorAgent", "prompt": "Oversee the workflow."},
            start_agent=start_agent,
            end_agent=end_agent,
            agents=agents,
            model=model_instance,
        )

    task = Task(
        agents=[agent.name for agent in agents],
        prompt=task_description,
    )

    # Initialize and run the workflow
    if workflow_type == "cot":
        workflow = ChainOfThoughtWorkflow(
            task=task,
            agents=agents,
            supervisor=supervisor_agent,
            start_agent=start_agent,
            end_agent=end_agent,
        )
        output, memory_id = workflow.run(**options)
    elif workflow_type == "tot":
        workflow = TreeOfThoughtWorkflow(
            task=task,
            agents=agents,
            supervisor=supervisor_agent,
            start_agent=start_agent,
            end_agent=end_agent,
        )
        output, memory_id = workflow.run(**options)
    elif workflow_type == "fluid":
        fluid_agent = FluidAgent(settings={"name": "FluidAgent"}, model=model_instance)
        supervisor_agent = SupervisorAgent(
            settings={"name": "SupervisorAgent", "prompt": "Oversee the workflow."},
            start_agent=None,
            end_agent=None,
            agents=[],
            model=model_instance,
        )
        task = Task(agents=[], prompt=task_description)
        workflow = FluidChainOfThoughtWorkflow(
            task=task,
            fluid_agent=fluid_agent,
            supervisor=supervisor_agent,
        )
        output, memory_id = workflow.run(**options)
        print(f"\nMemory ID: {memory_id}")
    else:
        raise ValueError(f"Unsupported workflow type: {workflow_type}")
    workflow_infos = {
        "output": output,
        "memory": {
            "memory_id": memory_id,
            "workflow_memory": workflow.memory.load(),
        },
        "agents": {
            "start_agent": start_agent.settings,
            "middle_agents": [agent.settings for agent in agents],
            "end_agent": end_agent.settings,
        },
    }
    return workflow_infos
