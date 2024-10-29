# OpenVela Documentation

Welcome to **OpenVela**, a powerful and flexible Python library designed to streamline the creation and management of intelligent workflows using various Language Learning Models (LLMs). Whether you're a beginner looking to get started or an advanced user aiming to customize and extend functionalities, this documentation will guide you through every aspect of OpenVela.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Modules Overview](#modules-overview)
    - [Files Module](#files-module)
    - [Tasks Module](#tasks-module)
    - [Memory Module](#memory-module)
    - [Tools Module](#tools-module)
    - [Agents Module](#agents-module)
    - [Logs Module](#logs-module)
    - [Messages Module](#messages-module)
    - [Workflows Module](#workflows-module)
    - [LLMs Module](#llms-module)
6. [Advanced Usage](#advanced-usage)
    - [Customizing Agents](#customizing-agents)
    - [Creating Subworkflows](#creating-subworkflows)
    - [Integrating Additional LLMs](#integrating-additional-llms)
    - [Extending Memory Functionalities](#extending-memory-functionalities)
7. [Examples and Use Cases](#examples-and-use-cases)
    - [Basic Workflow Execution](#basic-workflow-execution)
    - [Advanced Workflow with Custom Agents](#advanced-workflow-with-custom-agents)
8. [How to Publish a New Release](#how-to-publish-a-new-release)
9. [Troubleshooting](#troubleshooting)
10. [FAQs](#faqs)
11. [Contributing](#contributing)
12. [License](#license)

---

## Introduction

**OpenVela** is a versatile library that facilitates the development of intelligent workflows by leveraging different Language Learning Models (LLMs) such as OpenAI, Groq, and Ollama. It provides a structured approach to defining tasks, managing agents, handling memory, and integrating various tools to create sophisticated AI-driven processes.

### Key Features

- **Modular Architecture**: Easily extend and customize different components like agents, memory systems, and workflows.
- **Multi-LLM Support**: Seamlessly integrate with multiple LLM providers including OpenAI, Groq, and Ollama.
- **Memory Management**: Utilize short-term and workflow memory systems to maintain context and state.
- **Flexible Workflow Designs**: Implement various workflow patterns such as Chain of Thought, Tree of Thought, and Fluid Chain of Thought.
- **Comprehensive Logging**: Detailed logging capabilities for monitoring and debugging workflows.

---

## Objectives

The primary objectives of **OpenVela** are to:

1. **Simplify Workflow Creation**: Provide an intuitive framework for designing and executing complex AI-driven workflows.
2. **Ensure Flexibility and Extensibility**: Allow developers to customize and extend functionalities to suit diverse use cases.
3. **Promote Modular Design**: Encourage the use of modular components for better maintainability and scalability.
4. **Integrate Multiple LLMs**: Support various Language Learning Models to offer flexibility in choosing the best-suited model for specific tasks.
5. **Facilitate Memory Management**: Implement robust memory systems to maintain context and state across different workflow stages.
6. **Enhance Debugging and Monitoring**: Offer comprehensive logging to aid in monitoring workflow execution and troubleshooting issues.
7. **Support Advanced Workflow Patterns**: Enable the implementation of sophisticated workflow designs like Chain of Thought and Tree of Thought.

---

## Installation

To get started with OpenVela, ensure you have Python 3.7 or higher installed. You can install OpenVela using `pip`:

```bash
pip install openvela
```

*Note: If OpenVela is not available on PyPI, clone the repository and install it manually:*

```bash
git clone https://github.com/wearewebera/openvela.git
cd openvela
pip install -e .
```

---

## Quick Start Guide

This section provides a simple example to help you get up and running with OpenVela.

### Step 1: Configure Logging

Before running any workflows, configure the logging to monitor the process.

```python
from openvela.logs import configure_logging

configure_logging()
```

### Step 2: Define a Task

Create a task with a prompt and specify the agents involved.

```python
from openvela.tasks import Task

task = Task(
    agents=["StartAgent", "EndAgent"],
    prompt="Analyze the impact of renewable energy on global economies."
)
```

### Step 3: Select and Initialize an LLM Model

Choose an LLM provider (e.g., OpenAI) and instantiate the model.

```python
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="your-openai-api-key")
```

### Step 4: Run the Workflow

Use the provided `__main__.py` script or create a custom workflow.

```bash
python openvela --model openai --api_key your-openai-api-key --task "Analyze the impact of renewable energy on global economies." --verbose
```

*The above command initializes the workflow and prints the final output along with message history.*

---

## Modules Overview

OpenVela is organized into several modules, each responsible for specific functionalities. Below is an overview of each module:

### Files Module

**Path:** `openvela/files.py`

**Purpose:** Handles reading different types of files such as audio and images.

**Key Classes:**

- **File (Abstract Base Class):**
  - `read(path: str) -> bytes`: Abstract method to read files.

- **OpenVelaAudioFile:**
  - Reads `.wav` audio files and other binary files.

- **OpenVelaImageFile:**
  - Reads image files using PIL and returns their byte content.

**Usage Example:**

```python
from openvela.files import OpenVelaAudioFile, OpenVelaImageFile

audio_reader = OpenVelaAudioFile()
audio_bytes = audio_reader.read("path/to/audio.wav")

image_reader = OpenVelaImageFile()
image_bytes = image_reader.read("path/to/image.png")
```

### Tasks Module

**Path:** `openvela/tasks.py`

**Purpose:** Defines tasks and manages agent descriptions.

**Key Classes:**

- **Task:**
  - Initializes with a list of agent names and a prompt.
  - `read_agents(agents: List[str])`: Reads agent descriptions from JSON files.
  - `__str__() -> str`: Returns a string representation of the task.

**Usage Example:**

```python
from openvela.tasks import Task

task = Task(
    agents=["Agent1", "Agent2"],
    prompt="Describe the benefits of AI in healthcare."
)
print(task)
```

### Memory Module

**Path:** `openvela/memory.py`

**Purpose:** Manages different types of memory systems for agents and workflows.

**Key Classes:**

- **MemoryFormat (Abstract Base Class):**
  - `save(data: dict, file_path: str)`: Abstract method to save data.
  - `load(file_path: str) -> dict`: Abstract method to load data.

- **JsonMemoryFormat:**
  - Implements `MemoryFormat` to handle JSON-based memory storage.

- **ShortTermMemory (Abstract Base Class):**
  - Manages short-term memory for agents.

- **JsonShortTermMemory:**
  - Implements `ShortTermMemory` using JSON files.

- **WorkflowMemory:**
  - Manages memory specific to workflows.

- **AgentMemory:**
  - Stores information about agents.

- **JsonReader:**
  - Reads JSON files into dictionaries.

**Usage Example:**

```python
from openvela.memory import JsonShortTermMemory

memory = JsonShortTermMemory(prompt="You are an AI assistant.")
memory.remember("user", "Hello!")
memory.remember("assistant", "Hi there!")
messages = memory.recall()
print(messages)
```

### Tools Module

**Path:** `openvela/tools.py`

**Purpose:** Defines abstract tools and AI function tools.

**Key Classes:**

- **Tool (Abstract Base Class):**
  - `use()`: Abstract method to use the tool.

- **OpenAIFunction (TypedDict):**
  - Defines the structure for OpenAI functions.

- **AIFunctionTool (TypedDict):**
  - Defines the structure for AI function tools.

**Usage Example:**

```python
from openvela.tools import AIFunctionTool

tool = AIFunctionTool(
    type="function",
    function={
        "name": "calculate",
        "description": "Performs calculations",
        "parameters": {"expression": "str"},
        "strict": True
    }
)
```

### Agents Module

**Path:** `openvela/agents.py`

**Purpose:** Defines various types of agents that interact within workflows.

**Key Classes:**

- **Agent:**
  - Represents a generic agent with methods like `process`, `observe`, `understand`, etc.

- **SupervisorAgent:**
  - Manages the flow between multiple agents.

- **StartAgent & EndAgent:**
  - Represent the starting and ending points of a workflow.

- **FluidAgent:**
  - Dynamically generates agents based on task descriptions.

**Usage Example:**

```python
from openvela.agents import Agent, SupervisorAgent, StartAgent, EndAgent

start_agent = StartAgent(
    settings={
        "name": "StartAgent",
        "prompt": "Begin the analysis."
    }
)

end_agent = EndAgent(
    settings={
        "name": "EndAgent",
        "prompt": "Provide the final report."
    }
)

supervisor = SupervisorAgent(
    settings={
        "name": "SupervisorAgent",
        "prompt": "Oversee the workflow."
    },
    start_agent=start_agent,
    end_agent=end_agent
)
```

### Logs Module

**Path:** `openvela/logs.py`

**Purpose:** Configures logging for the OpenVela application.

**Key Functions:**

- **configure_logging():**
  - Sets up logging with both console and file handlers.

**Usage Example:**

```python
from openvela.logs import configure_logging

configure_logging()
```

### Messages Module

**Path:** `openvela/messages.py`

**Purpose:** Defines the structure of messages exchanged between agents.

**Key Classes:**

- **Message (TypedDict):**
  - Base structure for messages with `role` and `content`.

- **UserMessage, AssistantMessage, SystemMessage:**
  - Specific message types inheriting from `Message`.

**Usage Example:**

```python
from openvela.messages import UserMessage, AssistantMessage

user_msg = UserMessage(content="Hello, how can I assist you?")
assistant_msg = AssistantMessage(content="I need help with my project.")
```

### Workflows Module

**Path:** `openvela/workflows.py`

**Purpose:** Defines different workflow patterns for orchestrating agents.

**Key Classes:**

- **Workflow (Abstract Base Class):**
  - Base class for all workflows with abstract `run` method.

- **ChainOfThoughtWorkflow:**
  - Implements a linear chain of agent processing.

- **TreeOfThoughtWorkflow:**
  - Implements a tree-like structure for parallel thought processing.

- **FluidChainOfThoughtWorkflow:**
  - Dynamically adjusts the chain of thought based on task requirements.

**Usage Example:**

```python
from openvela.workflows import ChainOfThoughtWorkflow

workflow = ChainOfThoughtWorkflow(
    task=task,
    agents=[agent1, agent2],
    supervisor=supervisor_agent,
    start_agent=start_agent,
    end_agent=end_agent
)
final_output = workflow.run()
print(final_output)
```

### LLMs Module

**Path:** `openvela/llms.py`

**Purpose:** Integrates different Language Learning Models (LLMs) such as OpenAI, Groq, and Ollama.

**Key Classes:**

- **Model (Abstract Base Class):**
  - Defines the interface for LLM models with `generate_response` method.

- **OllamaModel, OpenAIModel, GroqModel:**
  - Implementations of `Model` for specific LLM providers.

**Usage Example:**

```python
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="your-openai-api-key")
response = model.generate_response(messages=[...])
print(response)
```

---

## Advanced Usage

For users looking to leverage the full potential of OpenVela, the following sections delve into advanced customization and extensions.

### Customizing Agents

Agents are the backbone of OpenVela workflows. You can create custom agents by extending the `Agent` class and overriding its methods.

**Example: Creating a Custom Agent**

```python
from openvela.agents import Agent
import logging

class CustomAgent(Agent):
    def process(self, input_data: str) -> str:
        logging.debug(f"{self.name} processing input data.")
        # Custom processing logic
        output = f"Processed: {input_data}"
        return output

# Instantiate the custom agent
custom_agent = CustomAgent(
    settings={
        "name": "CustomAgent",
        "prompt": "Process the given data."
    }
)
```

### Creating Subworkflows

Subworkflows allow you to break down complex tasks into manageable sub-tasks, each handled by its own workflow.

**Example: Defining a Subworkflow**

```python
from openvela.workflows import ChainOfThoughtWorkflow
from openvela.tasks import Task

subtask = Task(
    agents=["SubAgent1", "SubAgent2"],
    prompt="Analyze the marketing data."
)

subworkflow = ChainOfThoughtWorkflow(
    task=subtask,
    agents=[sub_agent1, sub_agent2],
    supervisor=supervisor_agent,
    start_agent=start_agent,
    end_agent=end_agent
)

# Integrate subworkflow into the main workflow
main_workflow = ChainOfThoughtWorkflow(
    task=main_task,
    agents=[agent1, agent2],
    supervisor=supervisor_agent,
    start_agent=start_agent,
    end_agent=end_agent,
    subworkflows=[subworkflow]
)
```

### Integrating Additional LLMs

OpenVela supports multiple LLM providers. To integrate a new provider, create a new class that inherits from `Model` and implements the `generate_response` method.

**Example: Adding a Hypothetical `NewLLMModel`**

```python
from openvela.llms import Model

class NewLLMModel(Model):
    def generate_response(self, messages, files=None, tools=None, tool_choice=None, format=None, options=None):
        # Implement API calls to the new LLM provider
        response = "Response from NewLLM"
        return response

# Usage
new_llm = NewLLMModel()
response = new_llm.generate_response(messages=[...])
print(response)
```

### Extending Memory Functionalities

You can create custom memory formats by extending the `MemoryFormat` abstract class.

**Example: Creating a YAML-Based Memory Format**

```python
import yaml
from openvela.memory import MemoryFormat

class YamlMemoryFormat(MemoryFormat):
    def save(self, data: dict, file_path: str):
        with open(file_path, "w") as f:
            yaml.dump(data, f)

    def load(self, file_path: str) -> dict:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

# Usage
from openvela.memory import JsonShortTermMemory

yaml_memory_format = YamlMemoryFormat()
yaml_memory = JsonShortTermMemory(
    prompt="You are an AI assistant.",
    memory_format=yaml_memory_format,
    file_path="memory.yaml"
)
yaml_memory.remember("user", "Hello!")
yaml_memory.remember("assistant", "Hi there!")
messages = yaml_memory.recall()
print(messages)
```

---

## Examples and Use Cases

### Basic Workflow Execution

This example demonstrates a simple workflow using the default StartAgent and EndAgent with the OpenAI model.

**Code Example:**

```python
from openvela.logs import configure_logging
from openvela.tasks import Task
from openvela.llms import OpenAIModel
from openvela.agents import StartAgent, EndAgent, SupervisorAgent
from openvela.workflows import FluidChainOfThoughtWorkflow

# Configure logging
configure_logging()

# Define the task
task = Task(
    agents=["StartAgent", "EndAgent"],
    prompt="Evaluate the effectiveness of remote work policies."
)

# Initialize agents
start_agent = StartAgent(
    settings={
        "name": "StartAgent",
        "prompt": "Initiate the evaluation of remote work policies."
    }
)

end_agent = EndAgent(
    settings={
        "name": "EndAgent",
        "prompt": "Conclude the evaluation and provide the final report.",
        "input": "Please summarize the findings."
    }
)

supervisor_agent = SupervisorAgent(
    settings={
        "name": "SupervisorAgent",
        "prompt": "Ensure the workflow progresses smoothly."
    },
    start_agent=start_agent,
    end_agent=end_agent
)

# Initialize the model
model = OpenAIModel(api_key="your-openai-api-key")

# Create and run the workflow
workflow = FluidChainOfThoughtWorkflow(
    task=task,
    fluid_agent=None,  # No fluid agent in this basic example
    supervisor=supervisor_agent,
    start_agent=start_agent,
    end_agent=end_agent
)

# Assign the model to agents
for agent in [supervisor_agent, start_agent, end_agent]:
    agent.model = model

# Run the workflow
final_output = workflow.run()
print("Final Output:", final_output)
```

### Advanced Workflow with Custom Agents

This example showcases a more complex workflow involving custom agents and dynamic subworkflows.

**Code Example:**

```python
from openvela.logs import configure_logging
from openvela.tasks import Task
from openvela.llms import OpenAIModel
from openvela.agents import Agent, SupervisorAgent, StartAgent, EndAgent
from openvela.workflows import ChainOfThoughtWorkflow
from openvela.memory import WorkflowMemory

# Configure logging
configure_logging()

# Define a custom agent
class DataAnalysisAgent(Agent):
    def process(self, input_data: str) -> str:
        # Custom data analysis logic
        analysis = f"Analyzed data: {input_data}"
        return analysis

# Initialize agents
start_agent = StartAgent(
    settings={
        "name": "StartAgent",
        "prompt": "Begin the data analysis workflow."
    }
)

data_agent = DataAnalysisAgent(
    settings={
        "name": "DataAnalysisAgent",
        "prompt": "Analyze the provided data."
    }
)

end_agent = EndAgent(
    settings={
        "name": "EndAgent",
        "prompt": "Compile the analysis results into a report.",
        "input": "Provide the final analysis report."
    }
)

supervisor_agent = SupervisorAgent(
    settings={
        "name": "SupervisorAgent",
        "prompt": "Manage the workflow and oversee the agents."
    },
    start_agent=start_agent,
    end_agent=end_agent
)

# Define the task
task = Task(
    agents=["StartAgent", "DataAnalysisAgent", "EndAgent"],
    prompt="Analyze sales data for Q1 2024."
)

# Initialize the model
model = OpenAIModel(api_key="your-openai-api-key")

# Create the workflow
workflow = ChainOfThoughtWorkflow(
    task=task,
    agents=[data_agent],
    supervisor=supervisor_agent,
    start_agent=start_agent,
    end_agent=end_agent
)

# Assign the model to agents
for agent in [supervisor_agent, start_agent, data_agent, end_agent]:
    agent.model = model

# Run the workflow
final_output = workflow.run()
print("Final Analysis Report:", final_output)
```

---

## How to Publish a New Release

Publishing a new release of OpenVela involves updating the version and pushing the changes to the remote repository with appropriate tagging. Follow the steps below to ensure a smooth release process.

### Step 1: Change the Version in `pyproject.toml`

Locate the `pyproject.toml` file in the root directory of your project. Update the `version` field to reflect the new release version.

```toml
[tool.poetry]
name = "openvela"
version = "1.2.0"  # Update this line with the new version
description = "Your project description"
# ... other configurations ...
```

Ensure that the version follows [Semantic Versioning](https://semver.org/) (e.g., `MAJOR.MINOR.PATCH`).

### Step 2: Commit and Tag the Release

Execute the following Git commands to commit your changes and create a tagged release. Replace `TAG` with your desired version tag (e.g., `v1.2.0`) and provide a meaningful commit message.

```bash
git add .
git commit -m "Release version TAG: Description of the release"
git tag -a TAG -m "Release version TAG: Description of the release"
git push origin TAG
```

**Example:**

```bash
git add .
git commit -m "Release version v1.2.0: Added new features and bug fixes"
git tag -a v1.2.0 -m "Release version v1.2.0: Added new features and bug fixes"
git push origin v1.2.0
```

### Step 3: Publish to PyPI (Optional)

If you wish to publish the new release to PyPI, ensure you have the necessary configurations and permissions. Use `twine` to upload the package.

```bash
pip install twine
poetry build
twine upload dist/*
```

*Note: Make sure your `pyproject.toml` is properly configured for PyPI publishing.*

---

## Troubleshooting

### Common Issues

1. **Invalid API Key:**
   - **Symptom:** Authentication errors when connecting to LLM providers.
   - **Solution:** Ensure that the API key provided is correct and has the necessary permissions.

2. **File Not Found:**
   - **Symptom:** Errors when reading agent description files or other resources.
   - **Solution:** Verify that the file paths are correct and the files exist.

3. **JSON Decode Errors:**
   - **Symptom:** Issues parsing JSON files for agents or memory.
   - **Solution:** Check the JSON files for proper syntax and structure.

4. **Model Initialization Errors:**
   - **Symptom:** Errors when initializing LLM models.
   - **Solution:** Ensure that all required parameters (like API keys or URLs) are provided and correct.

### Logging for Debugging

Enable verbose logging to gain insights into the workflow execution and identify issues.

```bash
python openvela --model openai --api_key your-api-key --task "Your task" --verbose
```

---

## FAQs

**Q1: What LLM providers are supported by OpenVela?**

*A1: Currently, OpenVela supports OpenAI, Groq, and Ollama. You can integrate additional providers by extending the `Model` class.*

---

**Q2: How can I add new agents to a workflow?**

*A2: Define new agent classes by extending the `Agent` class, implement the required methods, and include them in your workflow's agent list.*

---

**Q3: Can I use OpenVela with other file types besides audio and images?**

*A3: Yes. You can extend the `File` class to handle additional file types as needed.*

---

**Q4: How does OpenVela manage memory across workflows?**

*A4: OpenVela uses `WorkflowMemory` to maintain context and state across different parts of a workflow. You can customize memory handling by extending memory classes.*

---

**Q5: Is it possible to run multiple workflows simultaneously?**

*A5: Yes. You can instantiate and run multiple workflow objects independently, ensuring that their memory and resources are appropriately managed.*

---

## Contributing

We welcome contributions to OpenVela! Whether it's reporting bugs, suggesting features, or submitting pull requests, your input is valuable.

### How to Contribute

1. **Fork the Repository:**
   - Click the "Fork" button on the [OpenVela GitHub repository](https://github.com/wearewebera/openvela).

2. **Clone Your Fork:**
   ```bash
   git clone https://github.com/your-username/openvela.git
   cd openvela
   ```

3. **Create a New Branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Make Your Changes:**
   - Implement your feature or bug fix.

5. **Commit Your Changes:**
   ```bash
   git commit -m "Add feature: your-feature-description"
   ```

6. **Push to Your Fork:**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request:**
   - Navigate to the original repository and create a pull request from your fork.

---

# Conclusion

OpenVela is a comprehensive library designed to empower developers and AI enthusiasts to create intelligent, dynamic workflows with ease. By following this documentation, you can effectively utilize OpenVela's features, customize its components, and integrate it seamlessly into your projects. For further assistance or to contribute to the project, please visit our [GitHub repository](https://github.com/wearewebera/openvela).

Happy Coding!
```