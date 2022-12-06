import os
import time
from importlib import reload
import pandas as pd
import argparse
import numpy as np
import tensorflow as tf
from tensorflow import keras
from azureml.core import Model,Dataset
import joblib
import pickle as pkl
from azureml.core import Workspace, Experiment, Run, RunConfiguration

def preprocessing(inputs):

    run = Run.get_context()
    print("run name:",run.display_name)
    print("run details:",run.get_details())
    
    ws = run.experiment.workspace
        
    #model_vocab_path = Model.get_model_path(model_name='vocab_lstm_icu',_workspace=ws)
    #model_demog_vocab_path = model_vocab = Model.get_model_path(model_name='vocab_demog_lstm_icu',_workspace=ws)
    data_store = ws.get_default_datastore()
    
    vocab = Dataset.File.from_files(path=data_store.path('pkl/all_ftrs_dict.pkl'))
    demog_dict = Dataset.File.from_files(path=data_store.path('pkl/demog_dict.pkl'))
    
    pwd = os.path.dirname(__file__)
    output_dir = os.path.abspath(os.path.join(pwd,"output"))
    pkl_dir = os.path.join(output_dir, "pkl")

    os.makedirs(pkl_dir, exist_ok=True)
    vocab.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)
    demog_dict.download(target_path=pkl_dir,overwrite=True,ignore_not_found=True)


    with open(inputs, "rb") as f:
        inputs = pkl.load(f)

    with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
        model_vocab = pkl.load(f)

    with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
        model_vocab_demog = pkl.load(f)

    features = [l[0][-1] for l in inputs]
    N_VOCAB = len(model_vocab) + 1
    N_DEMOG = len(model_vocab_demog) + 1
    print(N_VOCAB,N_DEMOG)
    new_demog = [[i + N_VOCAB - 1 for i in l[1]] for l in inputs]
    features = [
                    features[i] + new_demog[i] for i in range(len(features))
                ]
    demog_vocab = {k: v + N_VOCAB - 1 for k, v in model_vocab_demog.items()}
    model_vocab.update(demog_vocab)
    N_VOCAB = np.max([np.max(l) for l in features]) + 1
    print(N_VOCAB,N_DEMOG)
    X = keras.preprocessing.sequence.pad_sequences(features,padding='post')
    print(X.shape)

    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser("feature")
    parser.add_argument("--inputs",type=str)
    parser.add_argument("--features_file",type=str)

    args = parser.parse_args()
    inputs = args.inputs
    print(inputs)
    print(args.features_file)

    features = preprocessing(inputs)
    #with open(os.path.join(args.features_file,"features.pkl"), "wb") as f:
    # pkl.dump(features, f)
    
    os.makedirs(args.features_file,exist_ok=True)
    np.savetxt(os.path.join(args.features_file,"features.csv"), features, delimiter=",")
    run = Run.get_context()
    ws = run.experiment.workspace
    
    data_store = ws.get_default_datastore()
    data_store.upload(src_dir=args.features_file,target_path=args.features_file,overwrite=True,show_progress=True)
    datastore_paths = [(data_store, args.features_file)]
    inputs = Dataset.Tabular.from_delimited_files(path=datastore_paths)
    inputs.register(name='premier_features',workspace=ws,create_new_version=True)


