import json
from pathlib import Path
from typing import Dict, List


class Task:
    def __init__(self, agents: List[str], prompt: str, agents_path: str = "agents"):
        self.prompt: str = prompt
        self.agents_path: Path = Path(agents_path)
        self.agents: Dict[str, str] = {}  # Dictionary to store agent descriptions
        self.read_agents(agents)

    def read_agents(self, agents: List[str]) -> None:
        """Read agent descriptions from JSON files."""
        for agent in agents:
            agent_file = self.agents_path / f"{agent}.json"
            try:
                with open(agent_file, "r") as file:
                    agent_json = json.load(file)
                    self.agents[agent] = agent_json.get("description", "")
            except FileNotFoundError:
                print(f"Warning: Agent file {agent_file} not found.")
                self.agents[agent] = ""
            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON in {agent_file}.")
                self.agents[agent] = ""
            except Exception as e:
                print(f"Error reading agent {agent}: {e}")
                self.agents[agent] = ""

    def __str__(self) -> str:
        """Return a string representation of the task."""
        task_summary = f"Prompt: {self.prompt}\n"
        for agent, description in self.agents.items():
            task_summary += f"Agent: {agent}\n"
            task_summary += f"{description}\n"
        return task_summary
