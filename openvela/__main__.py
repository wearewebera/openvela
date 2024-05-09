import logging

from logs import configure_logging
from agents import Agent


def main() -> None:
    configure_logging()
    agent = Agent()
    while True:
        message = input("You: ")
        if message.lower() == "exit":
            break
        response = agent.respond(message)
        print(f"Agent: {response}")
        logging.info(f"Memory: {agent.memory()}")


if __name__ == "__main__":
    main()
