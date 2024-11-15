import os
from pathlib import Path
from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories, save_json
from cnnClassifier.entity.config_entity import (
    DataIngestionConfig,
    PrepareBaseModelConfig,
    TrainingConfig,
    EvaluationConfig,
)


class ConfigurationManager:
    """
    Configuration Manager class for handling configuration retrieval
    and directory creation for various stages of the project.
    """

    def __init__(self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize the ConfigurationManager by reading YAML configuration files.

        Args:
            config_filepath (Path): Path to the configuration YAML file.
            params_filepath (Path): Path to the parameters YAML file.
        """
        self.config = read_yaml(config_filepath)  # Load configurations from YAML
        self.params = read_yaml(params_filepath)  # Load parameters from YAML

        # Create the artifacts root directory
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the Data Ingestion configuration.

        Returns:
            DataIngestionConfig: Configuration for data ingestion.
        """
        config = self.config.data_ingestion

        # Ensure the root directory for data ingestion exists
        create_directories([config.root_dir])

        # Create and return the DataIngestionConfig object
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        """
        Get the configuration for preparing the base model.

        Returns:
            PrepareBaseModelConfig: Configuration for base model preparation.
        """
        config = self.config.prepare_base_model

        # Ensure the root directory for base model preparation exists
        create_directories([config.root_dir])

        # Create and return the PrepareBaseModelConfig object
        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_image_size=self.params.IMAGE_SIZE,
            params_learning_rate=self.params.LEARNING_RATE,
            params_include_top=self.params.INCLUDE_TOP,
            params_weights=self.params.WEIGHTS,
            params_classes=self.params.CLASSES,
            params_dropout=self.params.DROPOUT,
        )

        return prepare_base_model_config

    def get_training_config(self) -> TrainingConfig:
        """
        Get the Training configuration.

        Returns:
            TrainingConfig: Configuration for training the model.
        """
        training = self.config.training
        prepare_base_model = self.config.prepare_base_model
        params = self.params

        # Path to the training data
        training_data = os.path.join(
            self.config.data_ingestion.unzip_dir, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone"
        )

        # Ensure the root directory for training exists
        create_directories([Path(training.root_dir)])

        # Create and return the TrainingConfig object
        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            updated_base_model_path=Path(prepare_base_model.updated_base_model_path),
            training_data=Path(training_data),
            params_epochs=params.EPOCHS,
            params_batch_size=params.BATCH_SIZE,
            params_is_augmentation=params.AUGMENTATION,
            params_image_size=params.IMAGE_SIZE,
        )

        return training_config

    def get_evaluation_config(self) -> EvaluationConfig:
        """
        Get the Evaluation configuration.

        Returns:
            EvaluationConfig: Configuration for evaluating the model.
        """
        # Create and return the EvaluationConfig object
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/model.h5",
            training_data="artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone",
            mlflow_uri="https://dagshub.com/om.mallick02/Kidney-Disease-Classification-Tensorflow.mlflow",
            all_params=self.params,
            params_image_size=self.params.IMAGE_SIZE,
            params_batch_size=self.params.BATCH_SIZE,
        )
        return eval_config
