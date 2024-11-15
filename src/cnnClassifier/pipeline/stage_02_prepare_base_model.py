# Import necessary modules and classes
from cnnClassifier.config.configuration import ConfigurationManager  # Handles configuration management
from cnnClassifier.components.prepare_base_model import PrepareBaseModel  # Component for preparing the base model
from cnnClassifier import logger  # Logger for tracking and debugging

# Define the name of the pipeline stage for logging purposes
STAGE_NAME = "Prepare Base Model"

class PrepareBaseModelTrainingPipeline:
    """
    A pipeline class to handle the preparation of the base model,
    including downloading, configuring, and updating the base model architecture.
    """
    def __init__(self):
        # Constructor - initializes the pipeline
        pass

    def main(self):
        """
        Main method to execute the steps for preparing the base model:
        - Fetch configuration for preparing the base model
        - Load and configure the base model
        - Update the base model with additional layers and compile it
        """
        # Initialize configuration manager and fetch the base model preparation configuration
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()

        # Create a PrepareBaseModel object with the fetched configuration
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)

        # Perform the base model preparation steps
        prepare_base_model.get_base_model()  # Load the base model and save its architecture
        prepare_base_model.update_base_model()  # Update the model by adding custom layers and compiling it


# Main execution block
if __name__ == '__main__':
    try:
        # Log the start of the pipeline stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the pipeline and execute it
        obj = PrepareBaseModelTrainingPipeline()
        obj.main()

        # Log the successful completion of the pipeline stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception if an error occurs and re-raise it
        logger.exception(e)
        raise e
