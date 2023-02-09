import os
import sys
from shutil import unpack_archive
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifacts
from src.configuration.s3_syncer import S3Sync
from src.exception import CustomException
from src.logger import logging
from src.constants import *
from sklearn.model_selection import train_test_split

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        
        self.s3 = S3Sync()

    def get_data_from_s3(self) -> None:
        try:
            logging.info("Entered the get_data_from_s3 method of Data ingestion class")
            os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)

            self.s3.sync_folder_from_s3(folder=self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,bucket_name=self.data_ingestion_config.BUCKET_NAME,bucket_folder_name=self.data_ingestion_config.S3_DATA_DIR)

            logging.info("Exited the get_data_from_s3 method of Data ingestion class")
        except Exception as e:
            raise CustomException(e, sys) from e

    def unzip_and_clean(self):
        logging.info("Entered the unzip_and_clean method of Data ingestion class")
        try:
            unpack_archive(filename=self.data_ingestion_config.ZIP_FILE_PATH,extract_dir=self.data_ingestion_config.ZIP_FILE_DIR,format="zip")

            logging.info("Exited the unzip_and_clean method of Data ingestion class")

            return self.data_ingestion_config.UNZIPPED_FILE_PATH
        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the initiate_data_ingestion method of Data ingestion class")
        try:

            self.get_data_from_s3()

            logging.info("Fetched the data from S3 bucket")
            unzipped_data= self.unzip_and_clean()
            logging.info("Unzipped file and splited into train, test and valid")

            data_ingestion_artifact = DataIngestionArtifacts(all_dataset_file_path=self.data_ingestion_config.ALL_DATASET_FILE_PATH)

            logging.info("Exited the initiate_data_ingestion method of Data ingestion class")

            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")

            return data_ingestion_artifact

        except Exception as e:
            raise CustomException(e, sys) from e