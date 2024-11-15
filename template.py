import os
import logging
from pathlib import Path

# Configure logging format and set the logging level to INFO
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# Define the main project name
project_name = 'cnnClassifier'

# List of files and directories to create for project structure
list_of_files = [
    ".github/workflows/.gitkeep",                         # Placeholder to keep the directory structure in Git
    f"src/{project_name}/__init__.py",                    # Initialize the main module
    f"src/{project_name}/components/__init__.py",         # Initialize components submodule
    f"src/{project_name}/utils/__init__.py",              # Initialize utils submodule
    f"src/{project_name}/config/__init__.py",             # Initialize config submodule
    f"src/{project_name}/config/configuration.py",        # Configuration settings for the project
    f"src/{project_name}/pipeline/__init__.py",           # Initialize pipeline submodule
    f"src/{project_name}/entity/__init__.py",             # Initialize entity submodule
    f"src/{project_name}/constants/__init__.py",          # Initialize constants submodule
    "config/config.yaml",                                 # YAML config file for project settings
    "dvc.yaml",                                           # DVC pipeline configuration
    "params.yaml",                                        # Parameters file for training models
    "requirements.txt",                                   # Dependencies for the project
    "setup.py",                                           # Setup script for packaging
    ".gitignore",                                         # Git ignore file
    "research/trials.ipynb",                              # Notebook for research and trials
    "templates/index.html"                                # HTML template file
]

# Create each file and directory in the specified project structure
for filepath in list_of_files:
    filepath = Path(filepath)  # Convert to Path object for easier handling
    filedir, filename = os.path.split(filepath)  # Split directory and file name

    # Create directories if they don't exist
    if filedir:
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    # Create empty file if it doesn't exist or if it is empty
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass  # Create the file
        logging.info(f"Creating empty file: {filepath}")

    else:
        logging.info(f"{filename} already exists")
