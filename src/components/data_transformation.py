import os
import sys
import pandas as pd
import numpy as np
import pickle
from src.logger import logging
from src.exception import CustomException
from src.constants import *
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# from src.utils.main_utils import save_object
from sklearn.preprocessing import LabelEncoder
import neattext.functions as nfx
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifacts, DataTransformationArtifacts

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifact: DataIngestionArtifacts):
        """
        > This function takes in a data transformation configuration and a data ingestion artifact and
        returns a data transformation artifact
        
        :param data_transformation_config: This is the configuration object that contains the parameters for
        the data transformation job
        :type data_transformation_config: DataTransformationConfig
        :param data_ingestion_artifact: This is the object that contains the data that we want to transform
        :type data_ingestion_artifact: DataIngestionArtifacts
        """
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifact = data_ingestion_artifact

    def load_split_data(self):
        """
        This function loads the data from the file path and splits the data into train and test data
        :return: train_dataset,test_dataset
        """
        try:
            logging.info("Entered into the load_split_data function")
            print(f"-------------{self.data_ingestion_artifact.all_dataset_file_path}--------------")
            df = pd.read_csv(self.data_ingestion_artifact.all_dataset_file_path)
            train_dataset,test_dataset = train_test_split(df,test_size = 0.2, random_state=42)
            logging.info("Exited the load_split_data function")
            return train_dataset,test_dataset
        except Exception as e:
            raise CustomException(e,sys) from e

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


    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        """
        The function loads the data, cleans the text, tokenizes the text, pads the text, encodes the output
        and saves the artifacts
        :return: DataTransformationArtifacts object
        """

        try:
            logging.info("Entered the initiate_data_transformation method of Data transformation class")

            train_dataset,test_dataset= self.load_split_data()

            clean_train_text,  _ = self.text_cleaning(train_dataset.text)
            clean_test_text, _ = self.text_cleaning(test_dataset.text)

            tokenizer = Tokenizer()
            tokenizer.fit_on_texts(clean_train_text)

            train_text_seq=tokenizer.texts_to_sequences(clean_train_text)
            train_text_pad=pad_sequences(train_text_seq,maxlen=50)


            test_text_seq=tokenizer.texts_to_sequences(clean_test_text)
            test_text_pad=pad_sequences(test_text_seq,maxlen=50)

            label_target=LabelEncoder()
            train_output=label_target.fit_transform(train_dataset['class'])
            test_output=label_target.transform(test_dataset['class'])

            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR,exist_ok=True)
            np.save(self.data_transformation_config.TRAIN_TEXT_PAD,train_text_pad)
            np.save(self.data_transformation_config.TEST_TEXT_PAD,test_text_pad)
            np.save(self.data_transformation_config.TRAIN_OUTPUT,train_output)
            np.save(self.data_transformation_config.TEST_OUTPUT,test_output)

            data_transformation_artifact = DataTransformationArtifacts(
                train_text_pad_path=self.data_transformation_config.TRAIN_TEXT_PAD,
                test_text_pad_path=self.data_transformation_config.TEST_TEXT_PAD,
                train_output_path = self.data_transformation_config.TRAIN_OUTPUT,
                test_output_path = self.data_transformation_config.TEST_OUTPUT,
                tokenizer = tokenizer
                )

            logging.info(f'{data_transformation_artifact}')

            logging.info("Exited the initiate_data_transformation method of Data transformation class")

            return data_transformation_artifact

        except Exception as e:
            raise CustomException(e, sys) from e
