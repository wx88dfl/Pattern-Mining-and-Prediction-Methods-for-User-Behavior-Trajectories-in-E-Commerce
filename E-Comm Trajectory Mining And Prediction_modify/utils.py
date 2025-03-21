import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("results/logs.txt"),
        logging.StreamHandler()
    ]
)

def log_message(message, level="info"):
    """
    Log messages to console and a log file.

    :param message: Log message
    :param level: Log level (info, warning, error)
    """
    if level == "info":
        logging.info(message)
    elif level == "warning":
        logging.warning(message)
    elif level == "error":
        logging.error(message)

def create_directories():
    """Ensure necessary directories exist before execution."""
    dirs = ["data", "models", "results"]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
