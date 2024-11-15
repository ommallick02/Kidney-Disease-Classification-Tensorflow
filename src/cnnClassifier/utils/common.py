import os
import json
import yaml
import joblib
import base64
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from cnnClassifier import logger
from box import ConfigBox
from pathlib import Path
from typing import Any


# Function to read a YAML file and return its contents as a ConfigBox object
@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its contents.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Raises:
        ValueError: If the YAML file is empty.
        Exception: For any other exceptions during file reading.

    Returns:
        ConfigBox: YAML file contents as a ConfigBox object.
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("YAML file is empty")
    except Exception as e:
        raise e

# Function to create directories from a list of paths
@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """Creates directories from a list of paths.

    Args:
        path_to_directories (list): List of directory paths to create.
        verbose (bool, optional): Logs creation details if True. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Created directory at: {path}")

# Function to save data as a JSON file
@ensure_annotations
def save_json(path: Path, data: dict):
    """Saves data to a JSON file.

    Args:
        path (Path): Path to the JSON file.
        data (dict): Data to save in JSON format.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")

# Function to load data from a JSON file
@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """Loads data from a JSON file.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Data as class attributes instead of a dictionary.
    """
    with open(path) as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully from: {path}")
    return ConfigBox(content)

# Function to save data as a binary file
@ensure_annotations
def save_bin(data: Any, path: Path):
    """Saves data to a binary file.

    Args:
        data (Any): Data to save in binary format.
        path (Path): Path to the binary file.
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"Binary file saved at: {path}")

# Function to load data from a binary file
@ensure_annotations
def load_bin(path: Path) -> Any:
    """Loads data from a binary file.

    Args:
        path (Path): Path to the binary file.

    Returns:
        Any: Object stored in the binary file.
    """
    data = joblib.load(path)
    logger.info(f"Binary file loaded from: {path}")
    return data

# Function to get the size of a file in KB
@ensure_annotations
def get_size(path: Path) -> str:
    """Gets the size of a file in KB.

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size in KB.
    """
    size_in_kb = round(os.path.getsize(path) / 1024)
    return f"~ {size_in_kb} KB"

# Function to decode a Base64 string into an image file
def decodeImage(imgstring, fileName):
    """Decodes a Base64 string into an image file.

    Args:
        imgstring (str): Base64-encoded image string.
        fileName (str): Name of the file to save the decoded image.
    """
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()

# Function to encode an image file into a Base64 string
def encodeImageIntoBase64(croppedImagePath):
    """Encodes an image file into a Base64 string.

    Args:
        croppedImagePath (str): Path to the image file.

    Returns:
        str: Base64-encoded string of the image.
    """
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())
