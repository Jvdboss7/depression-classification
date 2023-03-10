from dataclasses import dataclass
from src.constants import *
import os 

@dataclass
class DataIngestionConfig:
    def __init__(self):
        self.BUCKET_NAME: str = BUCKET_NAME
        self.ZIP_FILE_NAME:str = ZIP_FILE_NAME
        self.S3_DATA_DIR = DATA_DIR
        self.DATA_INGESTION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_DIR = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR)
        self.ZIP_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, self.ZIP_FILE_NAME)
        self.UNZIPPED_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR, RAW_FILE_NAME)
        self.ALL_DATASET_FILE_PATH = os.path.join(self.DATA_INGESTION_ARTIFACTS_DIR,DATASET_DIR)
        
@dataclass
class DataTransformationConfig: 
    def __init__(self): 
        self.ROOT_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR)
        self.DATA_TRANSFORMATION_ARTIFACTS_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,DATA_TRANSFORMATION_ARTIFACTS_DIR)
        self.TRAIN_TEXT_PAD = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                TRAIN_TEXT_PAD)
        self.TEST_TEXT_PAD = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                TEST_TEXT_PAD)
        self.TRAIN_OUTPUT = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                TRAIN_OUTPUT)
        self.TEST_OUTPUT = os.path.join(self.DATA_TRANSFORMATION_ARTIFACTS_DIR,
                                                                TEST_OUTPUT)

@dataclass
class ModelTrainerConfig:
     def __init__(self):
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.EMBEDDINGS = EMBEDDINGS
        self.EMBEDDINGS_PATH: str = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR,self.EMBEDDINGS)
        self.GLOVE_EMBEDDING_PATH = os.path.join(self.EMBEDDINGS_PATH,"glove.840B.300d.pkl")
@dataclass
class ModelEvaluationConfig: 
    def __init__(self):
        self.MODEL_EVALUATION_ARTIFACT_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR, MODEL_EVALUATION_ARTIFACTS_DIR)
        self.BUCKET_NAME = BUCKET_NAME 
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.MODEL_TRAINER_ARTIFACTS_DIR: str = os.path.join(os.getcwd(), ARTIFACTS_DIR, MODEL_TRAINER_ARTIFACTS_DIR)
        self.TRAINED_MODEL_PATH = os.path.join(self.MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.S3_MODEL_FOLDER = TRAINED_MODEL_DIR
        self.BUCKET_FOLDER_NAME = BUCKET_FOLDER_NAME
        self.S3_BUCKET_NAME = BUCKET_NAME
        # self.MODEL_DIR = MODEL_DIR

@dataclass
class ModelPusherConfig:
    def __init__(self):
        self.TRAINED_MODEL_DIR: str = os.path.join(os.getcwd(),ARTIFACTS_DIR,MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)
        self.BEST_MODEL_PATH: str = os.path.join(self.TRAINED_MODEL_DIR)
        self.BUCKET_NAME: str = BUCKET_NAME
        self.S3_MODEL_KEY_PATH: str = os.path.join(MODEL_TRAINER_ARTIFACTS_DIR,TRAINED_MODEL_DIR)

        