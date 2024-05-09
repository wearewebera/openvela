import json


class Task:
    def __init__(self, agents: list, prompt: str):
        self.agents: dict = dict.fromkeys(agents, "prompt")
        self.prompt: str = prompt
        self.agents_path: str = "agents"
        self.read_agents()

    def read_agents(self) -> None:
        """Read agent descriptions from JSON files."""
        for agent in self.agents.keys():
            with open(f"{self.agents_path}/{agent}.json", "r") as file:
                agent_json = json.load(file)
                self.agents[agent] = agent_json["description"]

    def __str__(self) -> str:
        """Return a string representation of the task."""
        task_summary = f"Prompt: {self.prompt}\n"
        for agent in self.agents.keys():
            task_summary += f"Agent: {agent}\n"
            task_summary += f"{self.agents[agent]}"
        return task_summary
