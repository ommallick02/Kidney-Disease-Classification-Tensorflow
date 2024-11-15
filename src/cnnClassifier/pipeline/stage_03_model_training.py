# Import necessary modules and classes
from cnnClassifier.config.configuration import ConfigurationManager  # Handles configuration management
from cnnClassifier.components.model_training import Training  # Component for training the model
from cnnClassifier import logger  # Logger for tracking and debugging

# Define the name of the pipeline stage for logging purposes
STAGE_NAME = "Training"

class ModelTrainingPipeline:
    """
    A pipeline class to handle the training process of the model.
    Includes loading the base model, preparing data generators, and training the model.
    """
    def __init__(self):
        # Constructor - initializes the pipeline
        pass

    def main(self):
        """
        Main method to execute the steps for training the model:
        - Fetch training configuration
        - Load the base model
        - Prepare data generators for training and validation
        - Train the model using the prepared data
        """
        # Initialize configuration manager and fetch the training configuration
        config = ConfigurationManager()
        training_config = config.get_training_config()

        # Create a Training object with the fetched configuration
        training = Training(config=training_config)

        # Perform the training steps
        training.get_base_model()  # Load the updated base model
        training.train_valid_generator()  # Prepare data generators for training and validation
        training.train()  # Train the model


# Main execution block
if __name__ == '__main__':
    try:
        # Log the start of the pipeline stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the pipeline and execute it
        obj = ModelTrainingPipeline()
        obj.main()

        # Log the successful completion of the pipeline stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception if an error occurs and re-raise it
        logger.exception(e)
        raise e
