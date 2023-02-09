from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    all_dataset_file_path:str

# Data Transformation artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_train_object: str 
    transformed_test_object: str
    train_data_path: str
    test_data_path: str