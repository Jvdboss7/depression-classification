import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from keras.utils import pad_sequences
from src.constants import *
from src.configuration.s3_syncer import S3Sync
from sklearn.metrics import confusion_matrix
from src.entity.config_entity import ModelEvaluationConfig
from src.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts



class ModelEvaluation:
    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):

        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.s3 = S3Sync()


    def get_model_from_s3(self) -> str:
        """
        Method Name :   predict
        Description :   This method predicts the image.

        Output      :   Predictions
        """
        logging.info("Entered the get_model_from_s3 method of PredictionPipeline class")
        try:
            logging.info(f"Checking the s3_key path{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            print(f"s3_key_path:{self.model_evaluation_config.TRAINED_MODEL_PATH}")
            best_model = self.s3.s3_key_path_available(bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,s3_key="ModelTrainerArtifacts/trained_model/")

            if best_model:
                self.s3.sync_folder_from_s3(folder=self.model_evaluation_config.TRAINED_MODEL_PATH,bucket_name=self.model_evaluation_config.S3_BUCKET_NAME,bucket_folder_name=self.model_evaluation_config.BUCKET_FOLDER_NAME)
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            best_model_path = os.path.join(self.model_evaluation_config.TRAINED_MODEL_PATH)
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self):

        try:
            logging.info("Entering into to the evaluate function of Model Evaluation class")

            test_text_pad = np.load(self.data_transformation_artifacts.test_text_pad_path)
            test_output = np.load(self.data_transformation_artifacts.test_output_path)

            load_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            accuracy = load_model.evaluate(test_text_pad,test_output)

            logging.info(f"the test accuracy is {accuracy}")

            lstm_prediction = load_model.predict(test_text_pad)
            res = []
            for prediction in lstm_prediction:
                if prediction[0] < 0.5:
                    res.append(0)
                else:
                    res.append(1)

            return accuracy
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
            Method Name :   initiate_model_evaluation
            Description :   This function is used to initiate all steps of the model evaluation

            Output      :   Returns model evaluation artifact
            On Failure  :   Write an exception log and then raise an exception
        """
        logging.info("Initiate Model Evaluation")
        try:

            logging.info("Loading currently trained model")
            trained_model=keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            
            test_text_pad = np.load(self.data_transformation_artifacts.test_text_pad_path)
            test_output = np.load(self.data_transformation_artifacts.test_output_path)

            trained_model_accuracy = trained_model.evaluate(test_text_pad,test_output)

            logging.info("Fetch best model from gcloud storage")
            best_model_path = self.get_model_from_s3()

            logging.info("Check is best model present in the gcloud storage or not ?")
            if os.path.isfile(best_model_path) is False:
                is_model_accepted = True
                logging.info("glcoud storage model is false and currently trained model accepted is true")

            else:
                logging.info("Load best model fetched from gcloud storage")
                best_model=keras.models.load_model(best_model_path)
                best_model_accuracy= best_model.evaluate(test_text_pad,test_output)
                logging.info("Comparing loss between best_model_loss and trained_model_loss ? ")
                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = True
                    logging.info("Trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("Trained model accepted")

            model_evaluation_artifacts = ModelEvaluationArtifacts(is_model_accepted=is_model_accepted)
            logging.info("Returning the ModelEvaluationArtifacts")
            return model_evaluation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e


