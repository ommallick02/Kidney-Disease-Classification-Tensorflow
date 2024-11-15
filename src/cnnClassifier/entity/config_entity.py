from dataclasses import dataclass
from pathlib import Path

# Configuration for data ingestion
@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion.

    Attributes:
        root_dir (Path): Root directory for data ingestion.
        source_URL (str): URL to download the source data.
        local_data_file (Path): Path to the local data file.
        unzip_dir (Path): Directory where the downloaded data will be unzipped.
    """
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path

# Configuration for preparing the base model
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    """Configuration for preparing the base model.

    Attributes:
        root_dir (Path): Root directory for model preparation.
        base_model_path (Path): Path to the pre-trained base model.
        updated_base_model_path (Path): Path to save the updated base model.
        params_image_size (list): List specifying the input image size.
        params_learning_rate (float): Learning rate for training.
        params_include_top (bool): Whether to include the top layers of the model.
        params_weights (str): Pre-trained weights to use (e.g., 'imagenet').
        params_classes (int): Number of output classes for the model.
        params_dropout (float): Dropout rate for regularization.
    """
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_image_size: list
    params_learning_rate: float
    params_include_top: bool
    params_weights: str
    params_classes: int
    params_dropout: float

# Configuration for training the model
@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for training the model.

    Attributes:
        root_dir (Path): Root directory for training outputs.
        trained_model_path (Path): Path to save the trained model.
        updated_base_model_path (Path): Path to the updated base model.
        training_data (Path): Path to the training dataset.
        params_epochs (int): Number of training epochs.
        params_batch_size (int): Batch size for training.
        params_is_augmentation (bool): Whether to apply data augmentation.
        params_image_size (list): Input image size for training.
    """
    root_dir: Path
    trained_model_path: Path
    updated_base_model_path: Path
    training_data: Path
    params_epochs: int
    params_batch_size: int
    params_is_augmentation: bool
    params_image_size: list

# Configuration for model evaluation
@dataclass(frozen=True)
class EvaluationConfig:
    """Configuration for evaluating the model.

    Attributes:
        path_of_model (Path): Path to the trained model to be evaluated.
        training_data (Path): Path to the evaluation dataset.
        all_params (dict): Dictionary containing all evaluation parameters.
        mlflow_uri (str): MLflow tracking URI for logging metrics and results.
        params_image_size (list): Input image size for evaluation.
        params_batch_size (int): Batch size for evaluation.
    """
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_image_size: list
    params_batch_size: int
