from dataclasses import dataclass

# Data Ingestion Artifacts
@dataclass
class DataIngestionArtifacts:
    all_dataset_file_path:str

# Data Transformation artifacts
@dataclass
class DataTransformationArtifacts:
    train_text_pad_path: str 
    test_text_pad_path: str
    train_output_path: str
    test_output_path: str
    tokenizer: object

# Model Trainer artifacts
@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str

# Model evaluation artifacts
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool
        
# Model Pusher Artifacts
@dataclass
class ModelPusherArtifacts:
    bucket_name: str
    s3_model_path: str