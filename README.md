# OpenVela



Welcome to **OpenVela**, a versatile and extensible Python framework designed to simplify the creation and management of complex workflows involving language models (LLMs). OpenVela empowers developers and researchers to build intelligent agents that can process, analyze, and generate information in a structured manner using advanced LLMs.

**Introducing the Fluid Chain of Thoughts**, a groundbreaking mechanic created by **Augusto Izepon**, which allows for dynamic and adaptive workflows that evolve based on the task at hand. This innovative approach enhances the flexibility and efficiency of agent interactions within OpenVela.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Getting Started](#getting-started)
  - [Defining Tasks](#defining-tasks)
  - [Creating Agents](#creating-agents)
  - [Constructing Workflows](#constructing-workflows)
- [Workflow Types](#workflow-types)
  - [Chain of Thought Workflow](#chain-of-thought-workflow)
  - [Tree of Thought Workflow](#tree-of-thought-workflow)
  - [Fluid Chain of Thought Workflow](#fluid-chain-of-thought-workflow)
- [The Fluid Chain of Thoughts Mechanic](#the-fluid-chain-of-thoughts-mechanic)
- [Using the CLI](#using-the-cli)
  - [Available Commands](#available-commands)
  - [Workflow Execution via CLI](#workflow-execution-via-cli)
  - [Examples](#examples)
- [Running OpenVela as a Server](#running-openvela-as-a-server)
  - [Starting the Server](#starting-the-server)
  - [Making API Requests](#making-api-requests)
  - [API Request Examples](#api-request-examples)
- [Making Requests](#making-requests)
  - [Request Structure](#request-structure)
  - [Request Examples](#request-examples)
- [Language Model Providers](#language-model-providers)
- [Advanced Usage](#advanced-usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Features

- **Fluid Chain of Thoughts**: Implement dynamic and adaptive workflows using the innovative mechanic created by Augusto Izepon.
- **Modular Design**: Easily define tasks, agents, and workflows that can be customized and extended.
- **Multiple Workflow Types**: Support for Chain of Thought, Tree of Thought, and Fluid Chain of Thought workflows.
- **Agent Management**: Create agents with specific roles and prompts that can interact within workflows.
- **Memory Persistence**: Use JSON-based memory storage to keep track of conversations and agent states.
- **Language Model Integration**: Seamless integration with various LLM providers like OpenAI, Groq, and Ollama.
- **Interactive CLI and Server Mode**: User-friendly Command Line Interface for setting up and running workflows interactively, and the ability to run OpenVela as a server.
- **Extensibility**: Designed with beginners and advanced users in mind, allowing for simple use cases and complex customizations.

---

## Installation

OpenVela is compatible with Python 3.7 and above. You can install it using `pip`:

```bash
pip install openvela
```

Alternatively, clone the repository for the latest development version:

```bash
git clone https://github.com/weberaAI/openvela.git
cd openvela
pip install -e .
```

---

## Getting Started

This guide will help you get up and running with OpenVela, whether you're a beginner or an experienced programmer.

### Defining Tasks

A **Task** represents the main objective you want your agents to accomplish. It includes a prompt and a list of agents involved.

```python
from openvela.tasks import Task

task = Task(
    agents=['StartAgent', 'ProcessingAgent', 'EndAgent'],
    prompt="Analyze the impact of climate change on polar bear populations."
)
```

### Creating Agents

Agents are the entities that process information within workflows. Each agent has settings like `name`, `prompt`, and can be associated with a language model.

```python
from openvela.agents import Agent, StartAgent, EndAgent
from openvela.llms import OpenAIModel

# Initialize the language model
model_instance = OpenAIModel(api_key='your-openai-api-key')

# Define agents
start_agent = StartAgent(
    settings={
        "name": "StartAgent",
        "prompt": "You are the StartAgent. Begin by summarizing the task."
    },
    model=model_instance
)

processing_agent = Agent(
    settings={
        "name": "ProcessingAgent",
        "prompt": "You are the ProcessingAgent. Provide detailed analysis."
    },
    model=model_instance
)

end_agent = EndAgent(
    settings={
        "name": "EndAgent",
        "prompt": "You are the EndAgent. Conclude the findings."
    },
    model=model_instance
)
```

### Constructing Workflows

Workflows define how agents interact to complete the task. OpenVela supports various workflow types, including the innovative Fluid Chain of Thoughts.

```python
from openvela.workflows import FluidChainOfThoughtWorkflow

workflow = FluidChainOfThoughtWorkflow(
    task=task,
    fluid_agent=fluid_agent,
    supervisor=supervisor_agent
)
```

---

## Workflow Types

OpenVela provides flexibility with different workflow structures.

### Chain of Thought Workflow

**Chain of Thought Workflow** is a sequential processing pipeline where agents handle data one after another, each building upon the previous agent's output.

- **Use Case**: Best for linear tasks where each step depends on the outcome of the previous step.
- **Setup**:
  ```python
  from openvela.workflows import ChainOfThoughtWorkflow

  workflow = ChainOfThoughtWorkflow(
      task=task,
      agents=[agent1, agent2],
      supervisor=supervisor_agent,
      start_agent=start_agent,
      end_agent=end_agent
  )
  ```
- **Example**:
  ```python
  task = Task(
      agents=['StartAgent', 'AnalysisAgent', 'ConclusionAgent'],
      prompt="Evaluate the economic impact of renewable energy adoption."
  )
  ```

### Tree of Thought Workflow

**Tree of Thought Workflow** allows for parallel processing of multiple thoughts or ideas, which are then evaluated and the best paths are selected.

- **Use Case**: Suitable for brainstorming, creative tasks, or when multiple solutions need to be explored.
- **Setup**:
  ```python
  from openvela.workflows import TreeOfThoughtWorkflow

  workflow = TreeOfThoughtWorkflow(
      task=task,
      agents=[agent1, agent2],
      supervisor=supervisor_agent,
      start_agent=start_agent,
      end_agent=end_agent
  )
  ```
- **Example**:
  ```python
  task = Task(
      agents=['StartAgent', 'IdeaGeneratorAgent', 'EvaluatorAgent', 'EndAgent'],
      prompt="Generate innovative marketing strategies for a new product."
  )
  ```

### Fluid Chain of Thought Workflow

**Fluid Chain of Thought Workflow** is an innovative mechanic created by **Augusto Izepon**. It dynamically generates agents based on the task, enabling adaptive processing and flexible workflows that evolve during execution.

- **Use Case**: Ideal for complex or undefined tasks where the workflow structure can benefit from adaptation.
- **Setup**:
  ```python
  from openvela.agents import FluidAgent, SupervisorAgent
  from openvela.workflows import FluidChainOfThoughtWorkflow

  fluid_agent = FluidAgent(settings={"name": "FluidAgent"}, model=model_instance)
  supervisor_agent = SupervisorAgent(settings={"name": "SupervisorAgent"}, model=model_instance)

  workflow = FluidChainOfThoughtWorkflow(
      task=task,
      fluid_agent=fluid_agent,
      supervisor=supervisor_agent
  )
  ```
- **Example**:
  ```python
  task = Task(
      agents=[],
      prompt="Design a comprehensive plan to improve urban transportation efficiency."
  )
  ```

---

## The Fluid Chain of Thoughts Mechanic

The **Fluid Chain of Thoughts** mechanic is a groundbreaking approach introduced by **Augusto Izepon**. It revolutionizes how workflows are constructed and executed in OpenVela by allowing agents to be dynamically generated and organized based on the complexity and requirements of the task.

### Key Features

- **Dynamic Agent Generation**: Agents are not pre-defined but created on-the-fly according to the task's needs.
- **Adaptive Workflows**: The workflow can adjust its structure during execution, adding or modifying agents as necessary.
- **Scalability**: Suitable for tasks of varying complexity, from simple queries to intricate problem-solving scenarios.
- **Enhanced Collaboration**: Agents can learn from previous interactions, leading to progressively improved responses.

### How It Works

The Fluid Agent analyzes the task description and generates a set of agents with specific roles and prompts. These agents then process the task in a sequence determined by the supervisor agent, ensuring that each aspect of the task is thoroughly addressed.

### Benefits

- **Flexibility**: Accommodates changes in task requirements without needing to redesign the workflow.
- **Efficiency**: Optimizes agent interactions to focus on relevant parts of the task, reducing redundancy.
- **Innovation**: Introduces a new paradigm in workflow management within LLM frameworks.

### Example Usage

```python
from openvela.agents import FluidAgent, SupervisorAgent

# Initialize the fluid agent
fluid_agent = FluidAgent(
    settings={"name": "FluidAgent"},
    model=model_instance
)

# Initialize the supervisor agent
supervisor_agent = SupervisorAgent(
    settings={"name": "SupervisorAgent", "prompt": "Oversee the workflow."},
    model=model_instance
)

# Create the workflow
workflow = FluidChainOfThoughtWorkflow(
    task=task,
    fluid_agent=fluid_agent,
    supervisor=supervisor_agent
)

# Run the workflow
final_output, memory_id = workflow.run()
print("Final Output:\n", final_output)
print("Memory ID:", memory_id)
```

---

## Using the CLI

OpenVela provides a powerful Command Line Interface (CLI) that allows you to interactively set up and run workflows without writing code.

### Available Commands

- **Start OpenVela Interface**: Launch the interactive CLI.
  ```bash
  openvela
  ```
- **Run Workflows**: Execute predefined workflows.
  ```bash
  openvela run [workflow-type] [options]
  ```
- **Start Server Mode**: Run OpenVela as a server to accept API requests.
  ```bash
  openvela serve [options]
  ```
- **Configure Providers**: Set up language model providers like OpenAI, Groq, or Ollama.
- **Load Agents**: Input paths to agent definition JSON files.
- **Save Outputs**: Option to save workflow outputs to files.

### Workflow Execution via CLI

When you start the OpenVela interface, you will be guided through the following steps:

1. **Select Workflow Type**: Choose between Chain of Thought, Tree of Thought, or Fluid Chain of Thought workflows.
2. **Select Provider**: Configure your preferred language model provider.
3. **Set API Key or Host URL**: Provide necessary credentials for the selected provider.
4. **Load Agents (if applicable)**: Input the path to your agents JSON file for Chain or Tree workflows.
5. **Input Task**: Describe the task you want the agents to perform.
6. **Run Workflow**: Execute the workflow and view the output.
7. **Save Output**: Optionally save the output to a file.

### Examples

#### Starting the CLI

```bash
openvela
```

#### Example Session

```plaintext
========================================
      Welcome to OpenVela Interface
========================================

Select the type of workflow:
1. Chain of Thought
2. Tree of Thought
3. Fluid Chain of Thought

Enter the number corresponding to your choice: 3

Select the language model provider:
1. Groq
2. Ollama
3. OpenAI

Enter the number corresponding to your choice: 3

Enter your API key for OpenAI: sk-...

Enter the task description:
>> Develop a strategic plan to reduce carbon emissions in urban areas.

Running the workflow. Please wait...

========================================
               Final Output
========================================
[Final output generated by the workflow]

Would you like to save the output to a file? (y/n): y
Enter the filename to save the output (e.g., 'output.txt'): strategy_plan.txt
Output saved to strategy_plan.txt
```

---

## Running OpenVela as a Server

OpenVela can run in server mode, allowing you to send API requests to execute workflows programmatically. This is useful for integrating OpenVela into other applications or services.

### Starting the Server

To start the OpenVela server, use the `openvela serve` command:

```bash
openvela serve --host 0.0.0.0 --port 8000
```

**Options:**

- `--host`: Specify the host IP address (default is `127.0.0.1`).
- `--port`: Specify the port number (default is `8000`).

### Making API Requests

Once the server is running, you can make API requests to execute workflows.

**Endpoint:**

- `POST /api/v1/workflow`

**Request Body:**

- `workflow_type`: Type of the workflow (`chain`, `tree`, `fluid`).
- `provider`: Language model provider (`openai`, `groq`, `ollama`).
- `api_key`: API key or credentials for the provider.
- `task_description`: Description of the task.
- `agents_definitions` (optional): Agent definitions for Chain or Tree workflows.
- `options` (optional): Additional options for the workflow.

### API Request Examples

#### Example Request using `curl`

```bash
curl -X POST http://localhost:8000/api/v1/workflow \
-H "Content-Type: application/json" \
-d '{
  "workflow_type": "fluid",
  "provider": "openai",
  "api_key": "your-openai-api-key",
  "task_description": "Create a detailed project plan for developing a new mobile application."
}'
```

#### Example Response

```json
{
  "status": "success",
  "final_output": "[Generated project plan]",
  "memory_id": "123e4567-e89b-12d3-a456-426614174000"
}
```

#### Example with Agents Definitions

For Chain or Tree workflows, include `agents_definitions`:

```bash
curl -X POST http://localhost:8000/api/v1/workflow \
-H "Content-Type: application/json" \
-d '{
  "workflow_type": "chain",
  "provider": "openai",
  "api_key": "your-openai-api-key",
  "task_description": "Analyze market trends for renewable energy.",
  "agents_definitions": [
    {
      "name": "StartAgent",
      "prompt": "You are the StartAgent. Begin the analysis.",
      "input": "Initiate market trend analysis."
    },
    {
      "name": "AnalysisAgent",
      "prompt": "You are the AnalysisAgent. Provide detailed insights.",
      "input": "Analyze the data provided by StartAgent."
    },
    {
      "name": "EndAgent",
      "prompt": "You are the EndAgent. Summarize the findings.",
      "input": "Conclude the analysis."
    }
  ]
}'
```

---

## Making Requests

OpenVela allows you to make requests to language models via agents within workflows. You can interact programmatically or through the CLI.

### Request Structure

A request in OpenVela typically involves:

- **Messages**: A list of conversation messages between the user and agents.
- **Files (Optional)**: Any files that need to be processed.
- **Tools (Optional)**: Tools that agents might use during processing.
- **Format (Optional)**: Desired response format (e.g., JSON).

### Request Examples

#### Programmatic Request

```python
from openvela.agents import Agent
from openvela.llms import OpenAIModel

# Initialize the language model
model_instance = OpenAIModel(api_key='your-openai-api-key')

# Create an agent
agent = Agent(
    settings={"name": "Assistant", "prompt": "You are a helpful assistant."},
    model=model_instance
)

# Define messages
messages = [
    {"role": "user", "content": "Can you explain the concept of photosynthesis?"}
]

# Generate response
response = agent.model.generate_response(messages)
print(response)
```

#### Request with Files and Tools

```python
from openvela.agents import Agent
from openvela.llms import OpenAIModel
from openvela.tools import AIFunctionTool

# Initialize the model
model_instance = OpenAIModel(api_key='your-openai-api-key')

# Define a tool
summarizer_tool = AIFunctionTool(
    type="summarizer",
    function={
        "name": "SummarizeText",
        "description": "Summarizes the provided text.",
        "parameters": {"text": "str"}
    }
)

# Create an agent
agent = Agent(
    settings={"name": "Assistant", "prompt": "You are a summarization assistant."},
    model=model_instance,
    tools=[summarizer_tool]
)

# Define messages and files
messages = [
    {"role": "user", "content": "Please summarize the content of this document."}
]
files = [{"type": "text", "path": "document.txt"}]

# Generate response using the tool
response = agent.model.generate_response(messages, files=files, tools=[summarizer_tool], tool_choice="SummarizeText")
print(response)
```

---

## Language Model Providers

OpenVela supports integration with various LLM providers:

- **OpenAI**: Utilize models like GPT-3 and GPT-4.
- **Groq**: Interface with Groq models for advanced processing.
- **Ollama**: Connect with Ollama models for specialized tasks.

You can select and configure providers based on your requirements.

**Setting Up Providers:**

- **OpenAI**:
  ```python
  from openvela.llms import OpenAIModel

  model_instance = OpenAIModel(api_key='your-openai-api-key')
  ```
- **Groq**:
  ```python
  from openvela.llms import GroqModel

  model_instance = GroqModel(api_key='your-groq-api-key')
  ```
- **Ollama**:
  ```python
  from openvela.llms import OllamaModel

  model_instance = OllamaModel(host='localhost', port=11434)
  ```

---

## Advanced Usage

- **Custom Agents**: Extend the `Agent` class to create agents with custom behaviors.
- **Memory Management**: Utilize `WorkflowMemory` and `AgentMemory` for complex state management.
- **Tool Integration**: Define and use tools within agents to perform specific actions.
- **Error Handling**: Implement robust error handling in workflows and agents.
- **Logging Configuration**: Customize logging settings for debugging and monitoring.
- **Deep Dive into Fluid Chain of Thoughts**: Explore the mechanics of dynamic agent generation and adaptive workflows.

---

## Contributing

We welcome contributions from the community! If you're interested in improving OpenVela, please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Write clear commit messages and test your changes.
4. Submit a pull request with a detailed description.

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

---

## License

OpenVela is licensed under the [MIT License](LICENSE). You are free to use, modify, and distribute this software in accordance with the license terms.

---

## Contact

For questions, suggestions, or feedback, please reach out to us:

- **Email**: hello@webera.com
- **GitHub Issues**: [Create an Issue](https://github.com/weberaAI/openvela/issues)

We'd love to hear from you!

---

Thank you for choosing OpenVela. We hope this framework helps you create powerful and intelligent workflows with ease. Experience the innovation of the Fluid Chain of Thoughts and elevate your projects to new heights. Happy coding!
