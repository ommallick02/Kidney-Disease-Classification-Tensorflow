# Import necessary modules and classes
from cnnClassifier.config.configuration import ConfigurationManager  # Handles configuration management
from cnnClassifier.components.model_evaluation_mlflow import Evaluation  # Evaluation component for the model
from cnnClassifier import logger  # Logger for tracking and debugging

# Define the name of the pipeline stage for logging purposes
STAGE_NAME = "Evaluation"

class EvaluationPipeline:
    """
    A pipeline class to handle the evaluation process of the model.
    Includes loading the trained model, evaluating it on validation data,
    saving the scores, and logging metrics to MLflow.
    """
    def __init__(self):
        # Constructor - initializes the pipeline
        pass

    def main(self):
        """
        Main method to execute the steps for evaluating the model:
        - Fetch evaluation configuration
        - Perform model evaluation
        - Save evaluation scores
        - Log metrics to MLflow (commented out for AWS deployment)
        """
        # Initialize configuration manager and fetch evaluation configuration
        config = ConfigurationManager()
        eval_config = config.get_evaluation_config()

        # Create an Evaluation object with the fetched configuration
        evaluation = Evaluation(eval_config)

        # Perform the evaluation steps
        evaluation.evaluation()  # Evaluate the model on validation data
        evaluation.save_score()  # Save evaluation scores

        # Log metrics into MLflow (comment this line before deploying on AWS)
        evaluation.log_into_mlflow()


# Main execution block
if __name__ == '__main__':
    try:
        # Log the start of the pipeline stage
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the pipeline and execute it
        obj = EvaluationPipeline()
        obj.main()

        # Log the successful completion of the pipeline stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception if an error occurs and re-raise it
        logger.exception(e)
        raise e
