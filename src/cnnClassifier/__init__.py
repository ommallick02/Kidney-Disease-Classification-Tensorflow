import os
import sys
import logging

# Define the format for log messages
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Directory to store log files
log_dir = "logs"

# Full path to the log file
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create the logs directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging settings
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format=logging_str,  # Specify the log message format

    # Specify handlers for logging: write to a file and output to the console
    handlers=[
        logging.FileHandler(log_filepath),  # Write log messages to a file
        logging.StreamHandler(sys.stdout)  # Print log messages to the console
    ]
)

# Create a logger instance with a custom name
logger = logging.getLogger("cnnClassifierLogger")
