import json
from pathlib import Path
from typing import Dict, List


class Task:
    """
    Represents a task within the OpenVela framework, encapsulating the prompt and associated agents.
    Responsible for reading agent descriptions from JSON files and providing a string representation of the task.
    """

    def __init__(self, prompt: str):
        """
        Initializes the Task instance.

        Args:
            agents (List[str]): A list of agent names involved in the task.
            prompt (str): The main prompt or description of the task.
            agents_path (str, optional): The directory path where agent JSON files are stored. Defaults to "agents".
        """
        self.prompt: str = prompt
