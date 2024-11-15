import os
import gdown
import zipfile
from cnnClassifier import logger
from cnnClassifier.utils.common import get_size
from cnnClassifier.entity.config_entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        """
        Initializes the DataIngestion class with a configuration object.
        :param config: DataIngestionConfig containing source URL, local file path, and extraction directory.
        """
        self.config = config

    def download_file(self) -> str:
        """
        Downloads a dataset from the specified Google Drive URL into the configured local file path.

        :return: None
        """
        try:
            # Extract source URL and local destination from the configuration
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file
            
            # Ensure the destination directory exists
            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)

            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            # Build the Google Drive download URL
            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # Use gdown to download the file
            gdown.download(prefix + file_id, zip_download_dir)

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")

        except Exception as e:
            logger.error("Error occurred while downloading the file.", exc_info=True)
            raise e

    def extract_zip_file(self) -> None:
        """
        Extracts the downloaded zip file into the configured directory.

        :return: None
        """
        try:
            unzip_path = self.config.unzip_dir
            
            # Ensure the extraction directory exists
            os.makedirs(unzip_path, exist_ok=True)

            logger.info(f"Extracting {self.config.local_data_file} to {unzip_path}")

            # Extract contents of the zip file
            with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
                zip_ref.extractall(unzip_path)

            logger.info(f"Extraction completed successfully into {unzip_path}")

        except zipfile.BadZipFile as e:
            logger.error("The zip file is corrupted or invalid.", exc_info=True)
            raise e
        except Exception as e:
            logger.error("An error occurred during extraction.", exc_info=True)
            raise e
