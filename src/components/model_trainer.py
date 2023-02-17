import os
import sys
import pickle
import pandas as pd
import numpy as np
from src.constants import *
import tensorflow as tf
import keras 
from keras.preprocessing.text import Tokenizer
from src.logger import logging
from src.exception import CustomException
from src.configuration.s3_syncer import S3Sync
from src.entity.config_entity import ModelTrainerConfig
from keras.layers import Embedding,Dense,LSTM,Bidirectional,GlobalMaxPooling1D,Input,Dropout,AveragePooling1D
from keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.models import Sequential
from src.entity.artifact_entity import DataTransformationArtifacts, ModelTrainerArtifacts

class ModelTrainer:
    
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                    model_trainer_config: ModelTrainerConfig):
        """
        `__init__` is a special function in Python that is called when an object is created from a class and
        allows the class to initialize the attributes of the class
        
        :param data_transformation_artifacts: This is the output of the data transformation step. It
        contains the transformed data and the metadata about the transformed data
        :type data_transformation_artifacts: DataTransformationArtifacts
        :param model_trainer_config: This is the configuration for the model trainer. It contains the
        following parameters:
        :type model_trainer_config: ModelTrainerConfig
        """

        self.data_transformation_artifacts = data_transformation_artifacts
        self.model_trainer_config = model_trainer_config
        self.s3 = S3Sync()


    def embedding_matrix(self) -> np.ndarray:
        """
        It takes the embedding file from the S3 bucket and creates an embedding matrix for the words in the
        vocabulary
        :return: The embedding matrix is being returned.
        """
        try:
            logging.info("Entered into the embedding_matrix function")
            self.s3.sync_folder_from_s3(folder = self.model_trainer_config.MODEL_TRAINER_ARTIFACTS_DIR,bucket_name = BUCKET_NAME,bucket_folder_name = self.model_trainer_config.EMBEDDINGS)
            with open(self.model_trainer_config.EMBEDDINGS_PATH, 'rb') as fp:
                glove_embedding = pickle.load(fp)
            tokenizer = self.data_transformation_artifacts.tokenizer
            v=len(tokenizer.word_index)
            embedding_matrix=np.zeros((v+1,300), dtype=float)
            for word,idx in tokenizer.word_index.items():
                embedding_vector=glove_embedding.get(word)
                if embedding_vector is not None:
                    embedding_matrix[idx]=embedding_vector
            logging.info("Exited the embedding matrix function")
            return embedding_matrix
        except Exception as e:
            raise CustomException(e,sys) from e 

    def model(self, embedding_matrix) -> Sequential:
        """
        It creates a model with an embedding layer, an LSTM layer, a global max pooling layer, a dense layer
        and an output layer
        :return: A model object is being returned.
        """
        try:
            logging.info("Creating the custom model")
            
            tokenizer = self.data_transformation_artifacts.tokenizer
            v=len(tokenizer.word_index)
            model=Sequential()
            model.add(Input(shape=(50,)))
            model.add(Embedding(v+1,300,weights=[embedding_matrix],trainable=False))
            model.add(LSTM(100,return_sequences=True))
            model.add(GlobalMaxPooling1D())
            model.add(Dropout(0.3))
            model.add(Dense(256,activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1,activation='sigmoid'))
            # model.compile(optimizer=keras.optimizers.SGD(0.1,momentum=0.09),loss='binary_crossentropy',metrics=['accuracy'])
            model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
            logging.info("Custom model is created")
            return model
        except Exception as e:
            raise CustomException(e,sys) from e
        
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered initiate_model_trainer method of ModelTrainer class")

        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            embedding_matrix = self.embedding_matrix()
            model = self.model(embedding_matrix)

            # early_stop=EarlyStopping(patience=10)
            # reducelr=ReduceLROnPlateau(patience=10)

            train_text_pad = np.load(self.data_transformation_artifacts.train_text_pad_path)
            train_output = np.load(self.data_transformation_artifacts.train_output_path)
            test_text_pad = np.load(self.data_transformation_artifacts.test_text_pad_path)
            test_output = np.load(self.data_transformation_artifacts.test_output_path)

            # history=model.fit(train_text_pad,train_output,validation_data=(test_text_pad,test_output),
            # epochs=30,batch_size=256,callbacks=[early_stop,reducelr])

            history=model.fit(train_text_pad,train_output,validation_data=(test_text_pad,test_output),
            epochs=20,batch_size=32)
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)

            logging.info(f"Saved the trained model")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifacts}")

            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e



