import json
from pathlib import Path
from typing import Dict, List


class Task:
    """
    Represents a task within the OpenVela framework, encapsulating the prompt and associated agents.
    Responsible for reading agent descriptions from JSON files and providing a string representation of the task.
    """

    def __init__(self, agents: List[str], prompt: str, agents_path: str = "agents"):
        """
        Initializes the Task instance.

        Args:
            agents (List[str]): A list of agent names involved in the task.
            prompt (str): The main prompt or description of the task.
            agents_path (str, optional): The directory path where agent JSON files are stored. Defaults to "agents".
        """
        self.prompt: str = prompt
        self.agents_path: Path = Path(agents_path)
        self.agents: Dict[str, str] = {}  # Dictionary to store agent descriptions
        self.read_agents(agents)

    def read_agents(self, agents: List[str]) -> None:
        """
        Reads agent descriptions from JSON files and populates the `agents` dictionary.

        Each agent's JSON file is expected to contain a "description" field.

        Args:
            agents (List[str]): A list of agent names to read descriptions for.

        Logs warnings and errors if agent files are missing or contain invalid JSON.
        """
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
        """
        Provides a string representation of the task, including the prompt and agent descriptions.

        Returns:
            str: A formatted string summarizing the task.
        """
        task_summary = f"Prompt: {self.prompt}\n"
        for agent, description in self.agents.items():
            task_summary += f"Agent: {agent}\n"
            task_summary += f"{description}\n"
        return task_summary
