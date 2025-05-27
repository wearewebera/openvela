"""Modern CLI using Typer for better developer experience."""

import json
from pathlib import Path
from typing import Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from pydantic import ValidationError

from ..core import (
    WorkflowConfig, 
    WorkflowType, 
    ModelConfig,
    ModelProvider,
    OpenVelaError,
    AgentFactory,
    WorkflowFactory,
    ModelFactory,
    auto_register_factories
)

# Initialize factories with existing implementations
auto_register_factories()

app = typer.Typer(
    help="OpenVela - Modern Agentic Workflows Library",
    rich_markup_mode="rich"
)
console = Console()


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", help="Host to bind the server to"),
    port: int = typer.Option(8000, help="Port to bind the server to"),
    workers: int = typer.Option(1, help="Number of worker processes"),
    reload: bool = typer.Option(False, help="Enable auto-reload for development"),
) -> None:
    """🚀 Start the OpenVela API server."""
    
    console.print(Panel(
        f"[bold green]Starting OpenVela Server[/bold green]\n\n"
        f"[cyan]Host:[/cyan] {host}\n"
        f"[cyan]Port:[/cyan] {port}\n"
        f"[cyan]Workers:[/cyan] {workers}\n"
        f"[cyan]Reload:[/cyan] {reload}",
        title="Server Configuration",
        border_style="green"
    ))
    
    try:
        # Import and start server
        from ..server import run_server
        run_server(host, port, workers=workers, reload=reload)
    except ImportError:
        console.print("[red]Error: Server module not available[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Failed to start server: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def run(
    workflow_type: WorkflowType = typer.Option(..., help="Type of workflow to execute"),
    agents_file: Path = typer.Option(..., help="Path to agents configuration file", exists=True),
    task: str = typer.Option(..., help="Task description"),
    provider: ModelProvider = typer.Option(ModelProvider.OPENAI, help="LLM provider"),
    model: str = typer.Option("gpt-4o-mini", help="Model name"),
    api_key: Optional[str] = typer.Option(None, help="API key for the provider"),
    base_url: Optional[str] = typer.Option(None, help="Base URL for the provider"),
    max_attempts: int = typer.Option(3, help="Maximum validation attempts", min=1, max=10),
    validate_output: bool = typer.Option(False, help="Enable output validation"),
    output_file: Optional[Path] = typer.Option(None, help="Save output to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    temperature: float = typer.Option(0.7, help="Model temperature", min=0.0, max=2.0),
) -> None:
    """⚡ Execute a workflow with the given configuration."""
    
    try:
        # Load and validate configuration
        console.print("[yellow]📋 Loading configuration...[/yellow]")
        agents_config = _load_agents_config(agents_file)
        
        workflow_config = WorkflowConfig(
            type=workflow_type,
            max_attempts=max_attempts,
            validate_output=validate_output
        )
        
        model_config = ModelConfig(
            provider=provider,
            model_name=model,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature
        )
        
        if verbose:
            _display_configuration(workflow_config, model_config, agents_config, task)
        
        # Execute workflow
        console.print("[yellow]🔄 Executing workflow...[/yellow]")
        result, memory_id = _execute_workflow(
            workflow_config, model_config, agents_config, task
        )
        
        # Output results
        _display_results(result, memory_id, output_file)
        
    except OpenVelaError as e:
        console.print(f"[red]❌ OpenVela Error: {e}[/red]")
        if verbose and hasattr(e, 'details') and e.details:
            console.print(f"[dim]Details: {e.details}[/dim]")
        raise typer.Exit(1)
    except ValidationError as e:
        console.print(f"[red]❌ Configuration Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Unexpected Error: {e}[/red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def validate(
    agents_file: Path = typer.Argument(..., help="Path to agents configuration file", exists=True),
    show_details: bool = typer.Option(False, "--details", help="Show detailed validation info"),
) -> None:
    """✅ Validate an agents configuration file."""
    
    try:
        agents_config = _load_agents_config(agents_file)
        
        console.print("[green]✅ Configuration is valid![/green]")
        
        # Display summary table
        table = Table(title="Agents Summary", show_header=True, header_style="bold blue")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Type", style="green")
        table.add_column("Has Prompt", style="yellow", justify="center")
        table.add_column("Has Input", style="magenta", justify="center")
        
        for agent_config in agents_config:
            table.add_row(
                agent_config.get("name", "Unknown"),
                agent_config.get("agent_type", "simple"),
                "✓" if agent_config.get("prompt") else "✗",
                "✓" if agent_config.get("input") else "✗"
            )
        
        console.print(table)
        
        if show_details:
            _display_agent_details(agents_config)
            
    except ValidationError as e:
        console.print(f"[red]❌ Validation Error: {e}[/red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]❌ Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def init(
    project_name: str = typer.Argument(..., help="Name of the project"),
    directory: Optional[Path] = typer.Option(None, help="Directory to create project in"),
) -> None:
    """🎯 Initialize a new OpenVela project with example configuration."""
    
    project_dir = directory or Path.cwd() / project_name
    
    if project_dir.exists():
        console.print(f"[red]Directory {project_dir} already exists[/red]")
        raise typer.Exit(1)
    
    try:
        _create_project_structure(project_dir, project_name)
        console.print(f"[green]✅ Project '{project_name}' created successfully![/green]")
        console.print(f"[cyan]📁 Location: {project_dir}[/cyan]")
        
    except Exception as e:
        console.print(f"[red]❌ Failed to create project: {e}[/red]")
        raise typer.Exit(1)


def _load_agents_config(agents_file: Path) -> list[dict]:
    """Load and validate agents configuration from file."""
    try:
        with open(agents_file) as f:
            data = json.load(f)
        
        if "agents" in data:
            return data["agents"]
        return data if isinstance(data, list) else [data]
        
    except json.JSONDecodeError as e:
        raise typer.BadParameter(f"Invalid JSON in agents file: {e}")


def _display_configuration(
    workflow_config: WorkflowConfig,
    model_config: ModelConfig, 
    agents_config: list[dict],
    task: str
) -> None:
    """Display configuration summary."""
    
    config_panel = Panel(
        f"[bold]Workflow:[/bold] {workflow_config.type}\n"
        f"[bold]Agents:[/bold] {len(agents_config)} loaded\n"
        f"[bold]Provider:[/bold] {model_config.provider}\n"
        f"[bold]Model:[/bold] {model_config.model_name}\n"
        f"[bold]Validation:[/bold] {'Enabled' if workflow_config.validate_output else 'Disabled'}\n"
        f"[bold]Max Attempts:[/bold] {workflow_config.max_attempts}",
        title="Configuration",
        border_style="blue"
    )
    
    task_panel = Panel(
        Text(task, style="italic"),
        title="Task", 
        border_style="green"
    )
    
    console.print(config_panel)
    console.print(task_panel)


def _execute_workflow(
    workflow_config: WorkflowConfig,
    model_config: ModelConfig,
    agents_config: list[dict],
    task: str
) -> tuple[str, str]:
    """Execute the workflow with given configuration."""
    
    # Create model instance
    model = ModelFactory.create_model(model_config)
    
    # Create agents
    agents = []
    supervisor = None
    start_agent = None
    end_agent = None
    
    for agent_config in agents_config:
        agent = AgentFactory.create_from_dict(agent_config, model)
        
        # Categorize agents based on type/name
        if "supervisor" in agent.name.lower():
            supervisor = agent
        elif "start" in agent.name.lower():
            start_agent = agent
        elif "end" in agent.name.lower(): 
            end_agent = agent
        else:
            agents.append(agent)
    
    # Create workflow
    workflow = WorkflowFactory.create_workflow(
        workflow_config,
        agents=agents,
        supervisor=supervisor or agents[0] if agents else None,
        start_agent=start_agent,
        end_agent=end_agent,
        task_prompt=task
    )
    
    # Execute workflow
    return workflow.run()


def _display_results(result: str, memory_id: str, output_file: Optional[Path]) -> None:
    """Display workflow results."""
    
    console.print("[green]✅ Workflow completed successfully![/green]")
    console.print(f"[dim]📝 Memory ID: {memory_id}[/dim]")
    
    if output_file:
        _save_output(output_file, result, memory_id)
        console.print(f"[green]💾 Output saved to {output_file}[/green]")
    else:
        result_panel = Panel(
            Text(result),
            title="Result",
            border_style="green"
        )
        console.print(result_panel)


def _display_agent_details(agents_config: list[dict]) -> None:
    """Display detailed agent information."""
    
    for i, agent_config in enumerate(agents_config, 1):
        details_panel = Panel(
            f"[bold]Name:[/bold] {agent_config.get('name', 'Unknown')}\n"
            f"[bold]Type:[/bold] {agent_config.get('agent_type', 'simple')}\n"
            f"[bold]Prompt:[/bold] {agent_config.get('prompt', 'No prompt')[:100]}...\n"
            f"[bold]Input:[/bold] {agent_config.get('input', 'No input')[:100]}...",
            title=f"Agent {i}",
            border_style="cyan"
        )
        console.print(details_panel)


def _save_output(output_file: Path, result: str, memory_id: str) -> None:
    """Save workflow output to file."""
    output_data = {
        "result": result,
        "memory_id": memory_id,
        "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
    }
    
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)


def _create_project_structure(project_dir: Path, project_name: str) -> None:
    """Create a new OpenVela project structure."""
    
    project_dir.mkdir(parents=True)
    
    # Create example agents configuration
    agents_config = {
        "agents": [
            {
                "name": "StartAgent",
                "prompt": "You are the StartAgent. Begin by outlining the main objectives of the task.",
                "input": "What are the primary goals for this task?",
                "agent_type": "simple"
            },
            {
                "name": "MiddleAgent", 
                "prompt": "You are the MiddleAgent. Expand on the ideas presented by the StartAgent.",
                "input": "Provide further details on the objectives.",
                "agent_type": "simple"
            },
            {
                "name": "EndAgent",
                "prompt": "You are the EndAgent. Conclude by synthesizing the information into a final answer.",
                "input": "Integrate the insights and deliver the final response.",
                "agent_type": "simple"
            }
        ]
    }
    
    with open(project_dir / "agents.json", "w") as f:
        json.dump(agents_config, f, indent=2)
    
    # Create example task file
    with open(project_dir / "example_task.txt", "w") as f:
        f.write("Tell a creative story about a hero overcoming challenges with unexpected twists.")
    
    # Create README
    readme_content = f"""# {project_name}

An OpenVela project for agentic workflows.

## Quick Start

1. Install OpenVela:
   ```bash
   pip install openvela
   ```

2. Run the example workflow:
   ```bash
   openvela run --workflow_type cot --agents_file agents.json --task "$(cat example_task.txt)" --provider openai --api_key YOUR_API_KEY
   ```

3. Validate your agents configuration:
   ```bash
   openvela validate agents.json
   ```

## Configuration Files

- `agents.json`: Defines the agents for your workflow
- `example_task.txt`: Example task description

## Learn More

Visit the OpenVela documentation for more information on creating powerful agentic workflows.
"""
    
    with open(project_dir / "README.md", "w") as f:
        f.write(readme_content)


if __name__ == "__main__":
    app()