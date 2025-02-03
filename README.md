
# OpenVela

OpenVela is an open‐source Python library for designing and executing **agentic workflows**. It leverages modern language models to create dynamic, interactive, and multi‐agent systems that collaboratively solve complex tasks. OpenVela supports multiple workflow paradigms—including Chain of Thought, Tree of Thought, Fluid Chain of Thought, and AutoSelect workflows—as well as a variety of agents, memory management techniques, and integrations with popular language model providers such as OpenAI, Groq, and Ollama.

> **Note:** When you run `pip install openvela`, you install the OpenVela CLI, which provides the following commands:
> - `openvela serve` – Run the API server.
> - `openvela run [parameters]` – Execute a workflow with custom parameters.
> - `openvela interface` – Launch the interactive workflow interface.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Features](#features)
3. [Installation](#installation)
4. [Quick Start Guide](#quick-start-guide)
5. [Workflows](#workflows)
    - [Chain of Thought Workflow](#chain-of-thought-workflow)
    - [Tree of Thought Workflow](#tree-of-thought-workflow)
    - [Fluid Chain of Thought Workflow](#fluid-chain-of-thought-workflow)
    - [AutoSelectWorkflow](#autoselectworkflow)
6. [SupervisorAgent: Types and Roles](#supervisoragent-types-and-roles)
7. [Language Model Integrations](#language-model-integrations)
8. [Agents](#agents)
    - [Basic Agent](#basic-agent)
    - [StartAgent & EndAgent](#startagent--endagent)
    - [SupervisorAgent](#supervisoragent)
    - [FluidAgent & FluidValidator](#fluidagent--fluidvalidator)
    - [SQLAgent](#sqlagent)
9. [Memory Management](#memory-management)
10. [Command Line Interface (CLI)](#command-line-interface-cli)
11. [Server API](#server-api)
12. [Advanced Examples Using All Parameters](#advanced-examples-using-all-parameters)
13. [Contributing](#contributing)
14. [License](#license)

---

## Introduction

OpenVela is designed to streamline the creation of multi-agent systems using state-of-the-art language models. Whether you are building a storytelling engine, a data analysis tool, or an interactive chatbot network, OpenVela provides a modular framework that allows you to define tasks, instantiate agents, manage conversation history, and integrate with various language model providers.

---

## Features

- **Agentic Workflows:** Supports sequential (Chain of Thought), branching (Tree of Thought), dynamic (Fluid Chain of Thought), and adaptive (AutoSelectWorkflow) modes.
- **Multi-Agent Support:** Create, manage, and orchestrate different types of agents (e.g., StartAgent, EndAgent, SupervisorAgent, FluidAgent).  
  **Agent Type Parameter:** Many agents, particularly the `SupervisorAgent`, accept an `agent_type` parameter (`"simple"` or `"selector"`) to control how the next agent is selected.
- **Memory Management:** Built-in support for short-term and workflow memory using JSON-based formats.
- **LLM Integrations:** Integrates with OpenAI, Groq, and Ollama language models.
- **CLI & Server API:** Run workflows via command line (`openvela run`, `openvela serve`, `openvela interface`) or expose them as RESTful endpoints.
- **File Processing:** Handle audio and image files for advanced interactions.

---

## Installation

Install OpenVela via pip:

```bash
pip install openvela
```

This installs the OpenVela CLI with the following commands:
- `openvela serve` – Run the API server.
- `openvela run [parameters]` – Execute a workflow.
- `openvela interface` – Launch the interactive workflow interface.

---

## Quick Start Guide

Here’s a minimal example to get you started with a Chain of Thought workflow using OpenVela with the OpenAI provider:

1. **Prepare Your Agents’ Definitions**

   Create a JSON file (e.g., `agents.json`) with your agent definitions:

   ```json
   {
     "agents": [
       {
         "name": "StartAgent",
         "prompt": "You are the StartAgent. Begin by outlining the main objectives of the task.",
         "input": "What are the primary goals for this task?"
       },
       {
         "name": "MiddleAgent",
         "prompt": "You are the MiddleAgent. Expand on the ideas presented by the StartAgent.",
         "input": "Provide further details on the objectives."
       },
       {
         "name": "EndAgent",
         "prompt": "You are the EndAgent. Conclude by synthesizing the information into a final answer.",
         "input": "Integrate the insights and deliver the final response."
       }
     ]
   }
   ```

2. **Run the Interactive Interface**

   Launch the interactive workflow interface:

   ```bash
   openvela interface
   ```

   Follow the prompts to select:
   - Workflow type (e.g., Chain of Thought)
   - Provider (e.g., OpenAI)
   - API Key
   - Agents JSON file path
   - Task description

3. **View the Output**

   After the workflow completes, the final output and memory ID will be displayed on screen and optionally saved to a file.

---

## Workflows

OpenVela provides several workflow types to suit different use cases.

### Chain of Thought Workflow

Agents process data sequentially. The output of one agent becomes the input of the next. This workflow supports an optional output validation loop.

#### Example Usage

```python
from openvela.agents import Agent, StartAgent, EndAgent, SupervisorAgent, FluidValidator
from openvela.tasks import Task
from openvela.workflows import ChainOfThoughtWorkflow
from openvela.llms import OpenAIModel

# Initialize model instance
model_instance = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

# Define agent settings with all parameters
start_settings = {
    "name": "StartAgent",
    "prompt": "You are the StartAgent. Begin by outlining the main objectives in detail.",
    "input": "What are the primary goals for this task? Please be specific."
}
middle_settings = {
    "name": "MiddleAgent",
    "prompt": "You are the MiddleAgent. Expand on the objectives by providing detailed insights.",
    "input": "Add further details to the objectives."
}
end_settings = {
    "name": "EndAgent",
    "prompt": "You are the EndAgent. Conclude by synthesizing all information into a final answer.",
    "input": "Summarize and integrate all insights."
}

# Instantiate agents
start_agent = StartAgent(settings=start_settings, model=model_instance)
middle_agent = Agent(settings=middle_settings, model=model_instance)
end_agent = EndAgent(settings=end_settings, model=model_instance)

# Create a FluidValidator instance for output validation
validator = FluidValidator(settings={"name": "FluidValidator"}, model=model_instance)

# Prepare intermediary agents and the supervisor (using simple mode by default)
supervisor = SupervisorAgent(
    settings={"name": "SupervisorAgent", "prompt": "Oversee the workflow and select the next agent in order."},
    start_agent=start_agent,
    end_agent=end_agent,
    agents=[middle_agent],
    model=model_instance
)

# Define the task
task = Task(prompt="Tell a creative story about a hero overcoming challenges with unexpected twists.")

# Initialize and run the workflow with output validation enabled
workflow = ChainOfThoughtWorkflow(
    task=task,
    agents=[middle_agent],
    supervisor=supervisor,
    start_agent=start_agent,
    end_agent=end_agent,
    validate_output=True,      # Enable output validation
    max_attempts=3,            # Maximum attempts for valid output
    validator=validator        # Provide a custom validator
)
final_output, memory_id = workflow.run()
print("Chain of Thought Final Output:", final_output)
print("Memory ID:", memory_id)
```

### Tree of Thought Workflow

This workflow generates multiple parallel thoughts, evaluates them, and may combine the best outputs.

#### Example Usage

```python
from openvela.agents import Agent, StartAgent, EndAgent, SupervisorAgent
from openvela.tasks import Task
from openvela.workflows import TreeOfThoughtWorkflow
from openvela.llms import OpenAIModel

# Initialize model instance
model_instance = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

# Define agents
start_agent = StartAgent(settings={
    "name": "StartAgent",
    "prompt": "You are the StartAgent. Initiate the task by outlining key points.",
    "input": "What is the main challenge?"
}, model=model_instance)

end_agent = EndAgent(settings={
    "name": "EndAgent",
    "prompt": "You are the EndAgent. Synthesize all insights into a final answer.",
    "input": "Provide the final synthesis."
}, model=model_instance)

middle_agent = Agent(settings={
    "name": "MiddleAgent",
    "prompt": "You are the MiddleAgent. Develop multiple ideas from the StartAgent's input.",
    "input": "Expand on the challenge with several possibilities."
}, model=model_instance)

supervisor = SupervisorAgent(
    settings={"name": "SupervisorAgent", "prompt": "Evaluate the generated thoughts and choose the best ones."},
    start_agent=start_agent,
    end_agent=end_agent,
    agents=[middle_agent],
    model=model_instance
)

# Define task
task = Task(prompt="Outline multiple innovative solutions for climate change mitigation.")

# Run Tree of Thought workflow
workflow = TreeOfThoughtWorkflow(
    task=task,
    agents=[middle_agent],
    supervisor=supervisor,
    start_agent=start_agent,
    end_agent=end_agent
)
final_output, memory_id = workflow.run()
print("Tree of Thought Final Output:", final_output)
print("Memory ID:", memory_id)
```

### Fluid Chain of Thought Workflow

This workflow dynamically generates agents based on the task description and allows parameters such as `max_attempts` and `max_previous_messages` for adaptive processing.

#### Example Usage

```python
from openvela.agents import FluidAgent, SupervisorAgent
from openvela.tasks import Task
from openvela.workflows import FluidChainOfThoughtWorkflow
from openvela.llms import OpenAIModel

# Initialize model instance
model_instance = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

# FluidAgent generates agents on the fly
fluid_agent = FluidAgent(settings={"name": "FluidAgent"}, model=model_instance)

# Supervisor for fluid workflows (without predefined agents)
supervisor = SupervisorAgent(
    settings={"name": "SupervisorAgent", "prompt": "Guide the fluid workflow and choose the best agent dynamically."},
    start_agent=None,
    end_agent=None,
    agents=[],   # Empty list as FluidAgent generates agents dynamically
    model=model_instance
)

# Define task
task = Task(prompt="Generate an innovative business strategy for a tech startup in a competitive market.")

# Run Fluid Chain of Thought workflow with additional parameters
workflow = FluidChainOfThoughtWorkflow(
    task=task,
    fluid_agent=fluid_agent,
    supervisor=supervisor,
    max_attempts=3,            # Maximum iterations for valid output
    max_previous_messages=5    # Limit on previous messages passed to agents
)
final_output, memory_id = workflow.run()
print("Fluid Chain of Thought Final Output:", final_output)
print("Memory ID:", memory_id)
```

### AutoSelectWorkflow

The **AutoSelectWorkflow** dynamically selects the next agent based on the supervisor’s evaluation of the latest output. It is adaptive and supports integrated output validation. In this workflow, the **SupervisorAgent** plays a key role in choosing the next agent using either a sequential or an LLM-driven dynamic approach.  
Below, we explore how the `agent_type` parameter in the SupervisorAgent affects this selection process.

#### Example Usage

```python
from openvela.agents import Agent, StartAgent, EndAgent, SupervisorAgent, FluidValidator
from openvela.tasks import Task
from openvela.workflows import AutoSelectWorkflow
from openvela.llms import OpenAIModel

# Initialize model instance
model_instance = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

# Define agents
start_agent = StartAgent(settings={
    "name": "StartAgent",
    "prompt": "You are the StartAgent. Begin by asking an open-ended question.",
    "input": "What is your initial thought on the task?"
}, model=model_instance)

middle_agent = Agent(settings={
    "name": "MiddleAgent",
    "prompt": "You are the MiddleAgent. Process the previous output and add further detail.",
    "input": "Expand on the idea."
}, model=model_instance)

end_agent = EndAgent(settings={
    "name": "EndAgent",
    "prompt": "You are the EndAgent. Conclude by summarizing all information.",
    "input": "Provide the final synthesis."
}, model=model_instance)

# Create a SupervisorAgent in selector mode for dynamic selection
supervisor = SupervisorAgent(
    settings={
      "name": "SupervisorAgent",
      "prompt": "Dynamically select the next agent based on the current output.",
      "agent_type": "selector"  # Use "selector" for LLM-driven decision-making
    },
    start_agent=start_agent,
    end_agent=end_agent,
    agents=[middle_agent],
    model=model_instance
)

# Create a FluidValidator for output validation
validator = FluidValidator(settings={"name": "FluidValidator"}, model=model_instance)

# Define task
task = Task(prompt="Discuss the impact of artificial intelligence on modern society.")

# Run AutoSelectWorkflow with adaptive agent selection and integrated validation
workflow = AutoSelectWorkflow(
    task=task,
    agents=[middle_agent],
    supervisor=supervisor,
    validate_output=True,   # Enable validation loop
    max_attempts=3,         # Maximum number of iterations
    validator=validator     # Custom validator instance
)
auto_output = workflow.run()
print("AutoSelectWorkflow Final Output:", auto_output)
```

---

## SupervisorAgent: Types and Roles

The **SupervisorAgent** orchestrates the workflow by deciding which agent should process the next piece of information. It supports two primary modes, determined by the `agent_type` parameter:

### Simple Mode (`agent_type: "simple"`)

- **Sequential Selection:**  
  The supervisor selects the next agent based on the defined order. For example, if the current agent is the StartAgent, it will choose the first intermediary agent; if not, it selects the agent immediately following the current one.
  
- **Deterministic Behavior:**  
  This mode is ideal for workflows with a clear, fixed sequence. It does not involve additional LLM-based reasoning.

- **Example Usage:**
  ```python
  supervisor = SupervisorAgent(
      settings={
        "name": "SupervisorAgent",
        "prompt": "Select the next agent in sequence.",
        "agent_type": "simple"  # Sequential selection mode
      },
      start_agent=start_agent,
      end_agent=end_agent,
      agents=[middle_agent],
      model=model_instance
  )
  ```
  
### Selector Mode (`agent_type: "selector"`)

- **Dynamic, LLM-Driven Decision Making:**  
  In this mode, the supervisor leverages the language model to analyze the conversation history, the latest output, and the overall task. It then produces a JSON response indicating:
  - **next_agent:** The selected agent's name (or `"FINISH"` to conclude the workflow).
  - **next_input:** Detailed instructions or context for the selected agent.
  - **thinking (optional):** The reasoning behind the decision.
  
- **Adaptive and Flexible:**  
  Selector mode is particularly useful for complex or ambiguous tasks where the optimal next step is not obvious. The dynamic decision-making enables the workflow to adapt by choosing the agent best suited to handle the current context.

- **Example Usage:**
  ```python
  supervisor = SupervisorAgent(
      settings={
        "name": "SupervisorAgent",
        "prompt": "Analyze the conversation and select the next agent based on the current output.",
        "agent_type": "selector"  # Use dynamic, LLM-driven selection
      },
      start_agent=start_agent,
      end_agent=end_agent,
      agents=[middle_agent],
      model=model_instance
  )
  ```

In summary, while **Simple Mode** offers a straightforward, deterministic selection of agents in a fixed sequence, **Selector Mode** provides a dynamic and adaptive mechanism that leverages the language model to guide the workflow based on the evolving context.

---

## Language Model Integrations

OpenVela abstracts LLM integrations via a common interface. Currently supported models include:

### OpenAIModel

```python
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")
messages = [{"role": "user", "content": "Hello, world!"}]
response = model.generate_response(messages)
print(response)
```

### GroqModel

```python
from openvela.llms import GroqModel

model = GroqModel(api_key="YOUR_API_KEY", model="llama-3.3-70b-versatile")
messages = [{"role": "user", "content": "Explain quantum physics."}]
response = model.generate_response(messages, format="json")
print(response)
```

### OllamaModel

```python
from openvela.llms import OllamaModel

model = OllamaModel(base_url="http://localhost:11434", model="llama3.2")
messages = [{"role": "user", "content": "Summarize the latest news."}]
response = model.generate_response(messages)
print(response)
```

---

## Agents

Agents are the building blocks of your workflows. They encapsulate prompts, process inputs, and interact with the language model.

### Basic Agent

```python
from openvela.agents import Agent
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")
agent_settings = {
    "name": "BasicAgent",
    "prompt": "You are a basic agent that processes user input.",
    "input": "What information do you need?"
}
agent = Agent(settings=agent_settings, model=model)
response = agent.generate(max_previous_messages=3)  # Passing extra kwargs
print(response)
```

### StartAgent & EndAgent

These agents mark the beginning and end of a workflow.

```python
from openvela.agents import StartAgent, EndAgent
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

start_agent = StartAgent(settings={
    "name": "StartAgent",
    "prompt": "You are the StartAgent. Begin by asking a clarifying question.",
    "input": "What is the task?"
}, model=model)

end_agent = EndAgent(settings={
    "name": "EndAgent",
    "prompt": "You are the EndAgent. Conclude the conversation with a summary.",
    "input": "Summarize the discussion."
}, model=model)
```

### SupervisorAgent

Orchestrates the workflow by selecting the next agent based on the latest output. See the section above for details on the `agent_type` parameter.

```python
from openvela.agents import SupervisorAgent
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")
supervisor = SupervisorAgent(
    settings={
      "name": "SupervisorAgent",
      "prompt": "Select the next best agent based on the latest output.",
      "agent_type": "selector"  # or "simple" for sequential selection
    },
    start_agent=start_agent,
    end_agent=end_agent,
    agents=[{"name": "DummyAgent", "prompt": "Dummy", "input": "Dummy"}],  # Example placeholder
    model=model
)
```

### FluidAgent & FluidValidator

FluidAgent dynamically generates agents based on the task. FluidValidator is used to validate the output.

```python
from openvela.agents import FluidAgent, FluidValidator
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

fluid_agent = FluidAgent(settings={"name": "FluidAgent"}, model=model)
validator = FluidValidator(settings={"name": "FluidValidator"}, model=model)

# FluidAgent can generate agent definitions based on task description
agents_definitions = fluid_agent.generate_agents_from_task(
    "Describe a new marketing strategy for a product launch, considering competitive analysis and market trends."
)
print("Generated Agent Definitions:", agents_definitions)
```

### SQLAgent

A specialized agent for generating read-only SQL queries and executing them via SQLAlchemy.

```python
from openvela.agents import SQLAgent
from openvela.llms import OpenAIModel

model = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

sql_agent = SQLAgent(
    settings={
        "name": "SQLAgent",
        "prompt": "Generate a SELECT query based on the user's input.",
        "input": "Show me the top 10 customers by revenue."
    },
    model=model,
    example_queries=[{"question": "Get all users", "sql_query": "SELECT * FROM users;"}],
    sql_dialect="postgresql",
    sqlalchemy_engine_url="postgresql://user:password@localhost/dbname",
    database_structure="Tables: users, orders, products",
    formatter_prompt="Format the SQL query results in a neat JSON output."
)

result = sql_agent.generate()
print("SQL Query Result:", result)
```

---

## Memory Management

OpenVela provides a flexible memory management system to store and recall messages across workflows.

### JSON Memory Format & Short-Term Memory

```python
from openvela.memory import JsonMemoryFormat, JsonShortTermMemory

# Create a JSON memory format instance
memory_format = JsonMemoryFormat()

# Initialize a short-term memory with a system prompt
memory = JsonShortTermMemory(prompt="System: This is the base prompt for the workflow.")

# Remember and recall messages
memory.remember("user", "Hello, how can I help?")
messages = memory.recall()
print("Short-Term Memory Messages:", messages)
```

### Workflow Memory

Keeps track of the conversation history within a workflow.

```python
from openvela.memory import WorkflowMemory

workflow_memory = WorkflowMemory(memory_id="workflow-123")
workflow_memory.add_message("AgentName", "assistant", "This is an output message.")
loaded_messages = workflow_memory.load()
print("Workflow Memory:", loaded_messages)
```

---

## Command Line Interface (CLI)

After installing OpenVela via pip, the CLI commands are available:

### Running a Workflow

```bash
openvela run \
  --provider openai \
  --workflow_type cot \
  --base_url_or_api_key YOUR_API_KEY \
  --model gpt-4o-mini \
  --agents path/to/agents.json \
  --task "Draft a creative story about adventure and discovery." \
  --options '{"max_attempts": 3, "other_option": "value"}'
```

### Running the Server

```bash
openvela serve --host 0.0.0.0 --port 8000
```

### Launching the Interactive Interface

```bash
openvela interface
```

---

## Server API

The server (built with FastAPI) exposes two endpoints:

- **`/run_workflow`**: Accepts workflow configuration and returns the workflow output.
- **`/completion`**: Accepts messages and returns a language model completion.

### Example Request

```bash
curl -X POST "http://localhost:8000/run_workflow" \
  -H "Content-Type: application/json" \
  -d '{
    "provider": "openai",
    "workflow_type": "cot",
    "base_url_or_api_key": "YOUR_API_KEY",
    "model": "gpt-4o-mini",
    "options": {"max_attempts": 3},
    "agents": [ ... ],  // JSON array of agent definitions
    "task": "Explain the theory of relativity."
  }'
```

---

## Advanced Examples Using All Parameters

This section combines various parameters from workflows, agents, memory, CLI, and LLM integrations to demonstrate a comprehensive use case.

### Comprehensive Workflow Example

```python
from openvela.agents import (
    Agent, StartAgent, EndAgent, SupervisorAgent, FluidAgent, FluidValidator, SQLAgent
)
from openvela.tasks import Task
from openvela.workflows import (
    ChainOfThoughtWorkflow, TreeOfThoughtWorkflow, FluidChainOfThoughtWorkflow, AutoSelectWorkflow
)
from openvela.memory import JsonShortTermMemory, WorkflowMemory
from openvela.llms import OpenAIModel

# Initialize model with all parameters
model_instance = OpenAIModel(api_key="YOUR_API_KEY", model="gpt-4o-mini")

# Initialize memory for agents and workflow
short_term_memory = JsonShortTermMemory(prompt="System: Base prompt for conversation.")
workflow_memory = WorkflowMemory(memory_id="advanced-workflow-001")

# Define detailed agent settings with extra options
start_settings = {
    "name": "StartAgent",
    "prompt": "You are the StartAgent. Begin by outlining the objectives in detail.",
    "input": "What is the initial strategy?"
}
middle_settings = {
    "name": "MiddleAgent",
    "prompt": "You are the MiddleAgent. Develop ideas based on the initial strategy.",
    "input": "Provide deeper analysis."
}
end_settings = {
    "name": "EndAgent",
    "prompt": "You are the EndAgent. Conclude by synthesizing all information.",
    "input": "Summarize the final output."
}

# Instantiate agents with memory and additional options
start_agent = StartAgent(settings=start_settings, model=model_instance)
middle_agent = Agent(settings=middle_settings, model=model_instance, options={"extra_detail": True})
end_agent = EndAgent(settings=end_settings, model=model_instance)

# Create a supervisor in selector mode (LLM-driven decision making)
supervisor = SupervisorAgent(
    settings={"name": "SupervisorAgent", "prompt": "Analyze conversation history and select the next agent.", "agent_type": "selector"},
    start_agent=start_agent,
    end_agent=end_agent,
    agents=[middle_agent],
    model=model_instance
)

# Create a FluidValidator for workflows that support output validation
validator = FluidValidator(settings={"name": "FluidValidator"}, model=model_instance)

# Define the task with a detailed description
task = Task(prompt="Develop a comprehensive marketing strategy for a new product launch, considering market trends and competitor analysis.")

# Example 1: Run a Chain of Thought Workflow with validation
cot_workflow = ChainOfThoughtWorkflow(
    task=task,
    agents=[middle_agent],
    supervisor=supervisor,
    start_agent=start_agent,
    end_agent=end_agent,
    validate_output=True,
    max_attempts=3,
    validator=validator
)
cot_output, cot_memory_id = cot_workflow.run()
print("Chain of Thought Output:", cot_output)

# Example 2: Run a Fluid Chain of Thought Workflow with dynamic agent generation
fluid_workflow = FluidChainOfThoughtWorkflow(
    task=task,
    fluid_agent=FluidAgent(settings={"name": "FluidAgent"}, model=model_instance),
    supervisor=supervisor,
    max_attempts=3,
    max_previous_messages=5
)
fluid_output, fluid_memory_id = fluid_workflow.run()
print("Fluid Chain of Thought Output:", fluid_output)

# Example 3: Run an AutoSelectWorkflow with adaptive agent selection and integrated validation
autoselect_workflow = AutoSelectWorkflow(
    task=task,
    agents=[middle_agent],
    supervisor=supervisor,
    validate_output=True,
    max_attempts=3,
    validator=validator
)
auto_output = autoselect_workflow.run()
print("AutoSelectWorkflow Output:", auto_output)

# Example 4: Use SQLAgent to generate a read-only SQL query
sql_agent = SQLAgent(
    settings={
        "name": "SQLAgent",
        "prompt": "Generate a SELECT query based on the user's input.",
        "input": "Retrieve the top 5 products by sales."
    },
    model=model_instance,
    example_queries=[{"question": "List all products", "sql_query": "SELECT * FROM products;"}],
    sql_dialect="postgresql",
    sqlalchemy_engine_url="postgresql://user:password@localhost/dbname",
    database_structure="Tables: products, sales, customers",
    formatter_prompt="Format the SQL query results in a well-structured JSON format."
)
sql_result = sql_agent.generate()
print("SQLAgent Result:", sql_result)
```

---

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines on how to contribute to OpenVela.

---

## License

OpenVela is released under the [MIT License](LICENSE).

---

Happy coding and enjoy building agentic workflows with OpenVela!
