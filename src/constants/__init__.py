import os 
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

# Data Ingestion constants
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'depression-classification'
ZIP_FILE_NAME = 'dataset.zip'
DATA_DIR = "data"
RAW_FILE_NAME = 'dataset'
DATA_INGESTION_ARTIFACTS_DIR = 'DataIngestionArtifacts'
DATASET_DIR = "Suicide_Detection.csv"

# Data transformation constants 
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTransformationArtifacts'
DATA_TRANSFORMATION_TRAIN_DIR = 'train_data'
DATA_TRANSFORMATION_TEST_DIR = 'test_data'
TRAIN_TEXT_PAD = "train_text_pad"
TEST_TEXT_PAD = "test_text_pad"
TRAIN_OUTPUT = 'train_output'
TEST_OUTPUT = 'test_output'

# AWS CONSTANTS
AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "ap-south-1"