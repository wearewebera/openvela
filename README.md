# OpenVela Documentation

Welcome to **OpenVela**, a robust framework designed to orchestrate intelligent workflows using various Language Models (LLMs) such as Groq, Ollama, and OpenAI. OpenVela empowers developers to create, manage, and execute complex workflows involving multiple agents, each tailored to perform specific tasks within a workflow.

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Architecture Overview](#architecture-overview)
    - [Core Components](#core-components)
    - [Memory Management](#memory-management)
    - [Agents](#agents)
    - [Workflows](#workflows)
    - [Language Models Integration](#language-models-integration)
6. [Using OpenVela](#using-openvela)
    - [Defining Tasks and Agents](#defining-tasks-and-agents)
    - [Configuring and Running Workflows](#configuring-and-running-workflows)
7. [Advanced Usage](#advanced-usage)
    - [Custom Agents](#custom-agents)
    - [Extending Memory Formats](#extending-memory-formats)
8. [Examples](#examples)
    - [Chain of Thought Workflow](#chain-of-thought-workflow)
    - [Tree of Thought Workflow](#tree-of-thought-workflow)
    - [Fluid Chain of Thought Workflow](#fluid-chain-of-thought-workflow)
9. [Configuration](#configuration)
10. [Logging](#logging)
11. [Contributing](#contributing)
12. [License](#license)
13. [Contact](#contact)

## Introduction

OpenVela is a versatile framework that facilitates the creation of intelligent workflows by leveraging multiple agents and integrating various LLMs. Whether you're building complex data processing pipelines, automating tasks, or developing AI-driven applications, OpenVela provides the tools and flexibility needed to streamline and enhance your workflows.

## Features

- **Multi-LLM Support**: Integrate with Groq, Ollama, OpenAI, and more.
- **Agent-Based Architecture**: Define and manage multiple agents, each with specialized roles.
- **Flexible Workflows**: Implement Chain of Thought, Tree of Thought, and Fluid Chain of Thought workflows.
- **Memory Management**: Robust memory systems to maintain context and state across workflows.
- **Extensible Tools**: Incorporate custom tools and functions to extend agent capabilities.
- **Interactive Interface**: User-friendly CLI for configuring and executing workflows.
- **Logging and Debugging**: Comprehensive logging for monitoring and troubleshooting.

## Installation

### Prerequisites

- **Python 3.8+**: Ensure you have Python installed. You can download it from [Python's official website](https://www.python.org/downloads/).
- **Git**: Required for cloning the repository. Download it from [Git's official website](https://git-scm.com/downloads).

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/wearewebera/openvela.git
   cd openvela
   ```

2. **Set Up a Virtual Environment**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables**

   Depending on the LLM provider you intend to use (e.g., OpenAI), set the necessary API keys as environment variables.

   ```bash
   export OPENAI_API_KEY='your-openai-api-key'
   export GROQ_API_KEY='your-groq-api-key'
   # Add other necessary environment variables
   ```

## Quick Start

1. **Define Agents**

   Create a JSON file (e.g., `agents.json`) defining the agents involved in your workflow.

   ```json
   {
     "agents": [
       {
         "name": "StartAgent",
         "prompt": "You are the StartAgent. Begin the task by introducing the main topic.",
         "input": "Provide an overview of the task."
       },
       {
         "name": "ResearchAgent",
         "prompt": "You are the ResearchAgent. Gather relevant information on the topic.",
         "input": "Retrieve the latest data and insights."
       },
       {
         "name": "EndAgent",
         "prompt": "You are the EndAgent. Summarize the findings and conclude the task.",
         "input": "Provide a comprehensive summary."
       }
     ]
   }
   ```

2. **Run the OpenVela Interface**

   Execute the main script to interactively set up and run your workflow.

   ```bash
   python -m openvela
   ```

   Follow the on-screen prompts to select workflow types, providers, configure agents, and execute tasks.

## Architecture Overview

OpenVela's architecture is modular, allowing for flexibility and scalability. Here's an overview of its core components:

### Core Components

- **Agents**: Represent individual units of work or intelligence within a workflow. Each agent has a specific role and can interact with other agents.
- **Workflows**: Define the sequence and structure in which agents operate to accomplish tasks.
- **Memory**: Manages the state and context, enabling agents to recall previous interactions and maintain continuity.
- **Language Models (LLMs)**: Integrate with external AI models to generate responses, process data, and perform tasks.

### Memory Management

OpenVela employs robust memory systems to maintain context and state:

- **ShortTermMemory**: Handles transient data relevant to ongoing tasks.
- **WorkflowMemory**: Maintains the state and context of entire workflows.
- **AgentMemory**: Stores information specific to individual agents.

These memory systems are implemented using different formats, with JSON being the default for persistence.

### Agents

Agents are the building blocks of workflows. Key types include:

- **StartAgent**: Initiates the workflow by setting the context and objectives.
- **EndAgent**: Concludes the workflow by synthesizing results and providing final outputs.
- **SupervisorAgent**: Oversees the workflow, managing the sequence of agents and evaluating outputs.
- **FluidAgent**: Dynamically generates and manages a set of agents based on task descriptions, enabling adaptive workflows.

Each agent is defined with a name, prompt, and input, allowing customization of roles and responsibilities.

### Workflows

OpenVela supports various workflow structures:

- **Chain of Thought Workflow**: Sequential processing where each agent handles the output of the previous one.
- **Tree of Thought Workflow**: Parallel processing with multiple thoughts or paths that are evaluated and combined.
- **Fluid Chain of Thought Workflow**: Dynamic workflows where agents are generated on-the-fly based on task requirements.

### Language Models Integration

OpenVela integrates with multiple LLM providers:

- **GroqModel**: Interface for Groq LLMs.
- **OllamaModel**: Interface for Ollama LLMs.
- **OpenAIModel**: Interface for OpenAI's models, including support for text and audio processing.

Each model inherits from an abstract `Model` class, ensuring a consistent interface for generating responses.

## Using OpenVela

### Defining Tasks and Agents

Tasks are central to workflows, encapsulating the main objectives and the agents involved. Here's how to define them:

1. **Create an Agents Definition File**

   Define your agents in a JSON file, specifying their roles, prompts, and inputs.

   ```json
   {
     "agents": [
       {
         "name": "StartAgent",
         "prompt": "You are the StartAgent. Begin the task by introducing the main topic.",
         "input": "Provide an overview of the task."
       },
       {
         "name": "ResearchAgent",
         "prompt": "You are the ResearchAgent. Gather relevant information on the topic.",
         "input": "Retrieve the latest data and insights."
       },
       {
         "name": "EndAgent",
         "prompt": "You are the EndAgent. Summarize the findings and conclude the task.",
         "input": "Provide a comprehensive summary."
       }
     ]
   }
   ```

2. **Initialize a Task**

   A `Task` instance requires a list of agents and a prompt.

   ```python
   from openvela.tasks import Task

   task = Task(
       agents=["StartAgent", "ResearchAgent", "EndAgent"],
       prompt="Analyze the impact of renewable energy adoption in urban areas."
   )
   ```

### Configuring and Running Workflows

1. **Select Workflow Type and Provider**

   When running the main interface (`__main__.py`), you'll be prompted to select the workflow type (e.g., Chain of Thought) and the LLM provider (e.g., OpenAI).

2. **Provide Agents Definition**

   Supply the path to your agents' JSON file or type 'example' to view a sample structure.

3. **Input Task Description**

   Enter the task description that the workflow will address.

4. **Execute Workflow**

   The workflow will process the task through the defined agents, generating outputs at each stage.

5. **View and Save Output**

   After execution, view the final output in the console and optionally save it to a file.

## Advanced Usage

### Custom Agents

To extend OpenVela with custom agents:

1. **Create a New Agent Class**

   Subclass the `Agent` class and override necessary methods.

   ```python
   from openvela.agents import Agent
   from openvela.memory import WorkflowMemory

   class CustomAgent(Agent):
       def process(self, input_data: str) -> str:
           # Implement custom processing logic
           return f"Custom processing of: {input_data}"
   ```

2. **Integrate with Workflows**

   Include your custom agent in the agents' JSON definition and utilize it within your workflows.

### Extending Memory Formats

OpenVela allows for custom memory formats:

1. **Create a New MemoryFormat Subclass**

   ```python
   from openvela.memory import MemoryFormat

   class CustomMemoryFormat(MemoryFormat):
       def save(self, data: dict, file_path: str):
           # Implement custom save logic
           pass

       def load(self, file_path: str) -> dict:
           # Implement custom load logic
           return {}
   ```

2. **Use the Custom Format**

   Assign your custom memory format when initializing memory components.

   ```python
   from openvela.memory import ShortTermMemory

   memory = ShortTermMemory(
       prompt="Your prompt here",
       memory_format=CustomMemoryFormat()
   )
   ```

## Examples

### Chain of Thought Workflow

A sequential workflow where each agent processes the output of the previous one.

1. **Agents Definition (`chain_agents.json`)**

   ```json
   {
     "agents": [
       {
         "name": "StartAgent",
         "prompt": "You are the StartAgent. Begin by introducing the main topic.",
         "input": "Provide an overview of the renewable energy task."
       },
       {
         "name": "ResearchAgent",
         "prompt": "You are the ResearchAgent. Gather relevant data on renewable energy adoption.",
         "input": "Retrieve the latest statistics and trends."
       },
       {
         "name": "EndAgent",
         "prompt": "You are the EndAgent. Summarize the findings and conclude the analysis.",
         "input": "Provide a comprehensive summary."
       }
     ]
   }
   ```

2. **Running the Workflow**

   ```bash
   python -m openvela
   ```

   - Select **Chain of Thought** as the workflow type.
   - Choose **OpenAI** as the provider.
   - Provide the path to `chain_agents.json`.
   - Enter the task description: "Analyze the impact of renewable energy adoption in urban areas."
   - Review and save the final output.

### Tree of Thought Workflow

A parallel workflow allowing multiple agents to explore different aspects before combining results.

1. **Agents Definition (`tree_agents.json`)**

   ```json
   {
     "agents": [
       {
         "name": "StartAgent",
         "prompt": "You are the StartAgent. Initiate the analysis of renewable energy impact.",
         "input": "Begin by outlining the key areas of impact."
       },
       {
         "name": "EconomicAgent",
         "prompt": "You are the EconomicAgent. Analyze the economic effects of renewable energy adoption.",
         "input": "Provide insights on job creation and market growth."
       },
       {
         "name": "EnvironmentalAgent",
         "prompt": "You are the EnvironmentalAgent. Assess the environmental benefits of renewable energy.",
         "input": "Detail the reductions in carbon emissions and pollution."
       },
       {
         "name": "SocialAgent",
         "prompt": "You are the SocialAgent. Explore the social implications of renewable energy adoption.",
         "input": "Discuss public perception and societal changes."
       },
       {
         "name": "EndAgent",
         "prompt": "You are the EndAgent. Combine all insights to provide a final comprehensive analysis.",
         "input": "Synthesize the economic, environmental, and social findings."
       }
     ]
   }
   ```

2. **Running the Workflow**

   ```bash
   python -m openvela
   ```

   - Select **Tree of Thought** as the workflow type.
   - Choose **OpenAI** as the provider.
   - Provide the path to `tree_agents.json`.
   - Enter the task description: "Assess the multifaceted impact of renewable energy adoption in urban settings."
   - Review and save the final combined analysis.

### Fluid Chain of Thought Workflow

A dynamic workflow where agents are generated based on the task description.

1. **Running the Workflow**

   ```bash
   python -m openvela
   ```

   - Select **Fluid Chain of Thought** as the workflow type.
   - Choose **OpenAI** as the provider.
   - The system will dynamically generate agents based on your task.
   - Enter the task description: "Develop a marketing strategy for a new eco-friendly product."
   - Review and save the dynamically generated strategy.

## Configuration

OpenVela can be customized via configuration files and environment variables:

- **API Keys**: Set API keys for LLM providers as environment variables (e.g., `OPENAI_API_KEY`).
- **Workflow Definitions**: Define agents and workflows using JSON files.
- **Memory Settings**: Configure memory formats and storage paths within the code or via configuration files.

## Logging

OpenVela employs a robust logging system to facilitate monitoring and debugging.

- **Console Logging**: Outputs INFO level logs to the console.
- **File Logging**: Saves DEBUG level logs to `app.log` for detailed analysis.

### Customizing Logging

Modify the `configure_logging` function in `logs.py` to adjust logging levels, formats, or handlers as needed.

```python
def configure_logging():
    # Modify logging settings here
    pass
```

## Contributing

Contributions are welcome! To contribute to OpenVela:

1. **Fork the Repository**

   Click the "Fork" button on the [GitHub repository](https://github.com/wearewebera/openvela) to create your own copy.

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature description"
   ```

4. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

5. **Open a Pull Request**

   Navigate to your forked repository and click "New Pull Request" to propose your changes.

Please ensure that your contributions adhere to the project's coding standards and include relevant tests and documentation.

## License

OpenVela is licensed under the [MIT License](./LICENSE). You are free to use, modify, and distribute this software, provided that you include the original license and copyright notice.

## Contact

For questions, suggestions, or support, please reach out to the OpenVela team at [hello@webera.com](mailto:hello@webera.com).

---

*This documentation is maintained and updated regularly. For the latest information, refer to the [official repository](https://github.com/wearewebera/openvela).*