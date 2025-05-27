# Suggested Package Structure

```
openvela/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── protocols.py          # Interface definitions
│   ├── exceptions.py         # Custom exceptions
│   └── types.py             # Common types and enums
├── agents/
│   ├── __init__.py
│   ├── base.py              # Abstract Agent class
│   ├── basic.py             # Agent, StartAgent, EndAgent
│   ├── supervisor.py        # SupervisorAgent
│   ├── fluid.py             # FluidAgent, FluidValidator
│   ├── sql.py               # SQLAgent
│   └── factory.py           # Agent creation factory
├── workflows/
│   ├── __init__.py
│   ├── base.py              # Abstract Workflow class
│   ├── chain_of_thought.py  # ChainOfThoughtWorkflow
│   ├── tree_of_thought.py   # TreeOfThoughtWorkflow
│   ├── fluid.py             # FluidChainOfThoughtWorkflow
│   ├── auto_select.py       # AutoSelectWorkflow
│   └── factory.py           # Workflow creation factory
├── llms/
│   ├── __init__.py
│   ├── base.py              # Abstract Model class
│   ├── openai.py            # OpenAIModel
│   ├── groq.py              # GroqModel
│   └── ollama.py            # OllamaModel
├── memory/
│   ├── __init__.py
│   ├── base.py              # Memory protocols
│   ├── json_memory.py       # JSON-based memory implementations
│   └── workflow_memory.py   # Workflow-specific memory
├── config/
│   ├── __init__.py
│   ├── models.py            # Pydantic config models
│   └── validation.py        # Configuration validation
└── cli/
    ├── __init__.py
    ├── commands.py          # CLI command implementations
    └── interface.py         # Interactive interface
```