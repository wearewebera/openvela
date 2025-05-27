# Example: Modern CLI using Typer for better developer experience
from pathlib import Path
from typing import Optional, Dict, Any
import json

import typer
from rich.console import Console
from rich.table import Table
from pydantic import ValidationError

from .config import WorkflowConfig, WorkflowType
from .factories import AgentFactory, WorkflowFactory
from .exceptions import OpenVelaError

app = typer.Typer(help="OpenVela - Agentic Workflows Library")
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
) -> None:
    """Start the OpenVela API server."""
    console.print(f"🚀 Starting OpenVela server on {host}:{port}")
    # Server implementation here


@app.command()
def run(
    workflow_type: WorkflowType = typer.Option(..., help="Type of workflow to execute"),
    agents_file: Path = typer.Option(..., help="Path to agents configuration file"),
    task: str = typer.Option(..., help="Task description"),
    provider: str = typer.Option("openai", help="LLM provider (openai, groq, ollama)"),
    model: str = typer.Option("gpt-4o-mini", help="Model name"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider"),
    max_attempts: int = typer.Option(3, help="Maximum validation attempts"),
    validate_output: bool = typer.Option(False, help="Enable output validation"),
    output_file: Optional[Path] = typer.Option(None, help="Save output to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """Execute a workflow with the given configuration."""
    
    try:
        # Load and validate configuration
        agents_config = _load_agents_config(agents_file)
        workflow_config = WorkflowConfig(
            type=workflow_type,
            max_attempts=max_attempts,
            validate_output=validate_output
        )
        
        if verbose:
            console.print("📋 Configuration:")
            console.print(f"  Workflow: {workflow_type}")
            console.print(f"  Agents: {len(agents_config)} agents loaded")
            console.print(f"  Provider: {provider}")
            console.print(f"  Model: {model}")
        
        # Execute workflow
        console.print("🔄 Executing workflow...")
        result, memory_id = _execute_workflow(
            workflow_config, agents_config, task, provider, model, api_key
        )
        
        # Output results
        console.print("✅ Workflow completed successfully!")
        console.print(f"📝 Memory ID: {memory_id}")
        
        if output_file:
            _save_output(output_file, result, memory_id)
            console.print(f"💾 Output saved to {output_file}")
        else:
            console.print("📄 Result:")
            console.print(result)
            
    except OpenVelaError as e:
        console.print(f"❌ OpenVela Error: {e}", style="red")
        if verbose and e.details:
            console.print(f"Details: {e.details}")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print(f"❌ Configuration Error: {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Unexpected Error: {e}", style="red")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def validate(
    agents_file: Path = typer.Argument(..., help="Path to agents configuration file"),
) -> None:
    """Validate an agents configuration file."""
    
    try:
        agents_config = _load_agents_config(agents_file)
        console.print("✅ Configuration is valid!")
        
        # Display summary table
        table = Table(title="Agents Summary")
        table.add_column("Name", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Has Prompt", style="yellow")
        
        for agent_config in agents_config:
            table.add_row(
                agent_config.get("name", "Unknown"),
                agent_config.get("agent_type", "simple"),
                "✓" if agent_config.get("prompt") else "✗"
            )
        
        console.print(table)
        
    except ValidationError as e:
        console.print(f"❌ Validation Error: {e}", style="red")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error: {e}", style="red")
        raise typer.Exit(1)


def _load_agents_config(agents_file: Path) -> list[dict]:
    """Load and validate agents configuration from file."""
    if not agents_file.exists():
        raise typer.BadParameter(f"Agents file not found: {agents_file}")
    
    try:
        with open(agents_file) as f:
            data = json.load(f)
        
        if "agents" in data:
            return data["agents"]
        return data
        
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON in agents file: {e}")


def _execute_workflow(
    workflow_config: WorkflowConfig,
    agents_config: list[dict],
    task: str,
    provider: str,
    model: str,
    api_key: Optional[str]
) -> tuple[str, str]:
    """Execute the workflow with given configuration."""
    # Implementation would go here
    # This is just a placeholder
    return "Workflow result", "memory-id-123"


def _save_output(output_file: Path, result: str, memory_id: str) -> None:
    """Save workflow output to file."""
    output_data = {
        "result": result,
        "memory_id": memory_id,
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)


if __name__ == "__main__":
    app()