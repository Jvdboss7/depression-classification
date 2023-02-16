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