
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from azureml.core import Model
import joblib
import pickle as pkl

def init():
    global model, model_vocab, model_vocab_demog,N_VOCAB,N_DEMOG

    model_path =  Model.get_model_path(model_name='dan_d1_icu',version=1)
    model = keras.models.load_model(model_path)

def run(input_data):

    resultList = []


    num_rows, num_cols = input_data.shape
    print("input data shape:",input_data.shape)
    print("input data type:",type(input_data))
    print(input_data)
    #Read comma-delimited data into an array
    # data = np.expand_dims(input_data,axis=0)
    # Reshape into a 2-dimensional array for model input
    prediction = model.predict(input_data).reshape((num_rows, 1))
    print("prediction shape:", prediction.shape)
    # Append prediction to results
    resultList.append(prediction)

    return pd.DataFrame(prediction)
