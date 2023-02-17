
import keras
import pickle
from keras_preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# text = input("Enter the sentance")

text = "i like humans they are idiots"

best_model_path:str = r"/home/jvdboss/workspace/ML_DL/Depression-Detection/depression-classification/artifacts/02_17_2023_16_36_39/ModelTrainerArtifacts/trained_model"

load_model=keras.models.load_model(best_model_path)

with open(r'/home/jvdboss/workspace/ML_DL/Depression-Detection/depression-classification/tokenizer.pickle', 'rb') as handle:
    load_tokenizer = pickle.load(handle)


print(text)

seq = load_tokenizer.texts_to_sequences([text])

print(seq)

padded = pad_sequences(seq, maxlen=50)
pred = load_model.predict(padded)
print("pred", pred)
if pred>0.5:
    print("depressive ")
else:
    print("not depressive")
