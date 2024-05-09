from abc import ABC, abstractmethod


class Tool(ABC):
    @abstractmethod
    def use(self):
        pass
