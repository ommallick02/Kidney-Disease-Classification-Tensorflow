# Import necessary modules and classes
from cnnClassifier.config.configuration import ConfigurationManager  # Handles configuration management
from cnnClassifier.components.data_ingestion import DataIngestion  # Data ingestion component
from cnnClassifier import logger  # Logger for tracking and debugging

# Define the name of the pipeline stage for logging purposes
STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    """
    A pipeline class to handle the data ingestion process, 
    which includes downloading and extracting data.
    """
    def __init__(self):
        # Constructor - initializes the pipeline
        pass

    def main(self):
        """
        Main method to execute the data ingestion steps:
        - Fetch configuration for data ingestion
        - Download the dataset
        - Extract the dataset
        """
        # Initialize configuration manager and fetch data ingestion configuration
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()

        # Create a DataIngestion object with the fetched configuration
        data_ingestion = DataIngestion(config=data_ingestion_config)

        # Perform the data ingestion steps
        data_ingestion.download_file()  # Download the data file
        data_ingestion.extract_zip_file()  # Extract the downloaded zip file


# Main execution block
if __name__ == '__main__':
    try:
        # Log the start of the pipeline stage
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")

        # Create an instance of the pipeline and execute it
        obj = DataIngestionTrainingPipeline()
        obj.main()

        # Log the successful completion of the pipeline stage
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        # Log the exception if an error occurs and re-raise it
        logger.exception(e)
        raise e
