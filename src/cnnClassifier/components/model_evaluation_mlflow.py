from pathlib import Path
import tensorflow as tf
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from cnnClassifier.entity.config_entity import EvaluationConfig
from cnnClassifier.utils.common import read_yaml, create_directories, save_json


class Evaluation:
    """
    A class to handle model evaluation and logging using TensorFlow and MLflow.
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initializes the Evaluation class with the provided configuration.

        :param config: EvaluationConfig object containing configuration parameters.
        """
        self.config = config
        self.valid_generator = None
        self.model = None
        self.score = None

    def _valid_generator(self):
        """
        Prepares the validation data generator for evaluation.
        """
        # Arguments for the data generator
        datagenerator_kwargs = dict(
            rescale=1.0 / 255,  # Rescale pixel values
            validation_split=0.30  # Use 30% of data for validation
        )

        # Arguments for data flow
        dataflow_kwargs = dict(
            target_size=self.config.params_image_size[:-1],  # Exclude channel dimension
            batch_size=self.config.params_batch_size,
            interpolation="bilinear"
        )

        # Validation data generator
        valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(**datagenerator_kwargs)
        self.valid_generator = valid_datagenerator.flow_from_directory(
            directory=self.config.training_data,
            subset="validation",
            shuffle=False,  # No shuffling for evaluation
            **dataflow_kwargs
        )

    @staticmethod
    def load_model(path: Path) -> tf.keras.Model:
        """
        Loads a trained model from the specified path.

        :param path: Path to the model file.
        :return: Loaded TensorFlow model.
        """
        return tf.keras.models.load_model(path)

    def evaluation(self):
        """
        Evaluates the model using the validation generator.
        """
        # Load the trained model
        self.model = self.load_model(self.config.path_of_model)

        # Prepare the validation data generator
        self._valid_generator()

        # Evaluate the model on the validation set
        self.score = self.model.evaluate(self.valid_generator)

        # Save the evaluation scores
        self.save_score()

    def save_score(self):
        """
        Saves the evaluation scores (loss and accuracy) to a JSON file.
        """
        scores = {"loss": self.score[0], "accuracy": self.score[1]}
        save_json(path=Path("scores.json"), data=scores)

    def log_into_mlflow(self):
        """
        Logs parameters, metrics, and the trained model into MLflow.

        - If the MLflow tracking URI is not a file store, the model will be registered.
        """
        mlflow.set_registry_uri(self.config.mlflow_uri)
        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        with mlflow.start_run():
            # Log configuration parameters
            mlflow.log_params(self.config.all_params)

            # Log evaluation metrics
            mlflow.log_metrics({"loss": self.score[0], "accuracy": self.score[1]})

            # Handle model registry based on tracking URI type
            if tracking_url_type_store != "file":
                # Register the model with a specific name
                mlflow.keras.log_model(self.model, "model", registered_model_name="VGG19Model")
            else:
                # Log the model without registering
                mlflow.keras.log_model(self.model, "model")
