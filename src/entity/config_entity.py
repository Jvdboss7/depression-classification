from dataclasses import dataclass
from src.constants import *
import os 

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME:str = ZIP_FILE_NAME
        self.S3_DATA_DIR = DATA_DIR
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR,DATA_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        self.UNZIPPED_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)
        self.ALL_DATASET_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        
@dataclass
class DataTransformationConfig: 
    def __init__(self): 
        self.ROOT_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                DATA_TRANSFORMATION_TRAIN_FILE_NAME)
        self.TEST_TRANSFORM_OBJECT_FILE_PATH = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                DATA_TRANSFORMATION_TEST_FILE_NAME)
        self.TRAIN_SPLIT = DATA_TRANSFORMATION_TRAIN_SPLIT
        self.TEST_SPLIT = DATA_TRANSFORMATION_TEST_SPLIT