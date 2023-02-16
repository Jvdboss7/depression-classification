import os
import io
import sys
import keras
import pickle
import neattext.functions as nfx
from src.logger import logging
from src.constants import *
from src.exception import CustomException
from keras.utils import pad_sequences
from src.configuration.s3_syncer import S3Sync
from src.entity.config_entity import DataTransformationConfig,ModelEvaluationConfig

class PredictionPipeline:
    def __init__(self):
        self.bucket_name = BUCKET_NAME
        self.data_transformation_config = DataTransformationConfig()
        self.model_evaluation_config = ModelEvaluationConfig()
        self.model_path = os.path.join("artifacts", "PredictModel")
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
            best_model = self.s3.s3_key_path_available(
                                                        bucket_name = self.model_evaluation_config.S3_BUCKET_NAME, 
                                                        s3_key = "ModelTrainerArtifacts/trained_model/"
                                                    )

            if best_model:
                self.s3.sync_folder_from_s3(
                                            folder = self.model_evaluation_config.TRAINED_MODEL_PATH,
                                            bucket_name = self.model_evaluation_config.S3_BUCKET_NAME,
                                            bucket_folder_name = self.model_evaluation_config.BUCKET_FOLDER_NAME
                                            )
            logging.info("Exited the get_model_from_s3 method of PredictionPipeline class")
            best_model_path = os.path.join(self.model_evaluation_config.TRAINED_MODEL_PATH)
            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e
        
    def text_cleaning(self,text):
        """
        This function takes in a list of sentences and returns a list of cleaned sentences and a list of the
        length of each sentence
        
        :param text: The text to be cleaned
        :return: clean_text and text_len
        """
        try:
            logging.info("Entered into the text_cleaning function")
            text_len = []
            clean_text = []
            for sentance in text:
                sentance = sentance.lower()
                sentance = nfx.remove_special_characters(sentance)
                sentance = nfx.remove_stopwords(sentance)
                text_len.append(len(sentance.split()))
                clean_text.append(sentance)
            logging.info("Exited the text_cleaning function")
            return clean_text,text_len
        except Exception as e:
            raise CustomException(e,sys) from e

    def predict(self,best_model_path,text):
        """load image, returns cuda tensor"""
        logging.info("Running the predict function")
        try:
            best_model_path:str = self.get_model_from_s3()
            load_model=keras.models.load_model(best_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)
            
            text,text_len = self.text_cleaning(text)
            # text=self.data_transformation.concat_data_cleaning(text)
            text = [text]            
            print(text)
            seq = load_tokenizer.texts_to_sequences(text)
            padded = pad_sequences(seq, maxlen=50)
            print(seq)
            pred = load_model.predict(padded)
            pred
            print("pred", pred)
            if pred>0.4:
                print("depressive ")
                return "depressive"
            else:
                print("not depressive")
                return "not depressive"
        except Exception as e:
            raise CustomException(e, sys) from e

    def run_pipeline(self,text):
        logging.info("Entered the run_pipeline method of PredictionPipeline class")
        try:

            best_model_path: str = self.get_model_from_s3() 
            predicted_text = self.predict(best_model_path,text)
            logging.info("Exited the run_pipeline method of PredictionPipeline class")
            return predicted_text
        except Exception as e:
            raise CustomException(e, sys) from e


            

