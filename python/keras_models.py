"""
Starter Keras model
"""
# %%
import itertools
import os
import pickle as pkl
import random

import kerastuner.tuners as tuners
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

from tools import keras as tk

# %% Globals
SAMPLE_N = 10000
TIME_SEQ = 225
LSTM_DROPOUT = 0.1
LSTM_RECURRENT_DROPOUT = 0.2
N_LSTM = 128
HYPER_TUNING = False
BATCH_SIZE = 32

# %% Load in Data
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("data/data/") + "/"
pkl_dir = output_dir + "pkl/"

with open(pkl_dir + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "pat_data.pkl", "rb") as f:
    y_ = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

# %% Determining number of vocab entries
N_VOCAB = len(vocab)

# %% Create data generator for On-the-fly batch generation
dat_generator = tk.DataGenerator(inputs,
                                 max_time=TIME_SEQ,
                                 batch_size=BATCH_SIZE)

# %% Model

if HYPER_TUNING:
    # Generate Hyperparameter model
    hyper_model = tk.LSTMHyperModel(ragged=False,
                                    n_timesteps=TIME_SEQ,
                                    vocab_size=N_VOCAB,
                                    batch_size=BATCH_SIZE)
    tuner = tuners.Hyperband(
        hyper_model,
        objective="accuracy",
        max_epochs=5,
        project_name="hyperparameter-tuning",
        # NOTE: This could be in output as well if we don't want to track/version it
        directory="data/model_checkpoints/",
        distribution_strategy=tf.distribute.MirroredStrategy())

    # Announce the search space
    tuner.search_space_summary()

    # And search the space
    tuner.search(dat_generator, epochs=5)

    # Get results
    tuner.results_summary()
else:

    # Normal model, no hyperparameter tuning nonsense
    input_layer = keras.Input(
        shape=(TIME_SEQ, None),
        batch_size=BATCH_SIZE,
    )
    # Feature Embeddings
    emb1 = keras.layers.Embedding(N_VOCAB,
                                  output_dim=512,
                                  name="Feature_Embeddings")(input_layer)
    # Average weights of embedding
    emb2 = keras.layers.Embedding(N_VOCAB,
                                  output_dim=1,
                                  name="Average_Embeddings")(input_layer)

    # Multiply and average
    mult = keras.layers.Multiply(name="Embeddings_by_Average")([emb1, emb2])
    avg = keras.backend.mean(mult, axis=2)

    lstm_layer = keras.layers.LSTM(N_LSTM,
                                   dropout=LSTM_DROPOUT,
                                   recurrent_dropout=LSTM_RECURRENT_DROPOUT,
                                   name="Recurrent")(avg)

    output_dim = keras.layers.Dense(1, activation="sigmoid",
                                    name="Output")(lstm_layer)

    model = keras.Model(input_layer, output_dim)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")

    model.summary()

    # %% Train
    # NOTE: Multiprocessing is superfluous here with epochs=1, but we could use it
    model.fit(dat_generator)