import logging
import sys


def configure_logging():
    """
    Configures the logging settings for the OpenVela application.

    Sets up two handlers:
        - StreamHandler for stdout with INFO level.
        - FileHandler for logging detailed debug information to 'app.log'.

    Both handlers use distinct formatters for console and file outputs.
    """
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Create a stream handler for stdout
    stdout_handler = logging.StreamHandler(stream=sys.stdout)
    stdout_handler.setLevel(logging.INFO)

    # Create a file handler
    file_handler = logging.FileHandler("app.log")
    file_handler.setLevel(logging.DEBUG)

    # Create formatters
    console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Set formatters for handlers
    stdout_handler.setFormatter(console_formatter)
    file_handler.setFormatter(file_formatter)

    # Add handlers to the logger
    logger.addHandler(stdout_handler)
    logger.addHandler(file_handler)
