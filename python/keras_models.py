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

import tools.keras as tk

# %% Globals
sample_n = 10000
time_seq = 225
lstm_dropout = 0.2
lstm_recurrent_dropout = 0.2
n_lstm = 128
HYPER_TUNING = True
BATCH_SIZE = 32
# %% Load in Data
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("data/data/") + "/"
pkl_dir = output_dir + "pkl/"

with open(pkl_dir + "int_seqs.pkl", "rb") as f:
    X_ = pkl.load(f)

with open(pkl_dir + "pat_data.pkl", "rb") as f:
    y_ = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

# %% Determining number of vocab entries
n_tok = len(vocab)
# %% HACK: Sampling only a few to prototype:
X = random.choices(X_, k=sample_n)

# HACK: Randomly generating labels to prototype, remove before moving on
y = np.random.randint(low=0, high=2, size=len(X))

# Determine largest bag size
# NOTE: There's probably an easier way to do this
lens = [[len(x) for x in y] for y in X]
n_bags = max(itertools.chain(*lens))

# %% Create data generator for On-the-fly batch generation
dat_generator = tk.DataGenerator(inputs=X,
                                 labels=y,
                                 dim=[time_seq, n_bags],
                                 batch_size=BATCH_SIZE)

# %% Model

if HYPER_TUNING:
    # Generate Hyperparameter model
    hyper_model = tk.LSTMHyperModel(ragged=False,
                                    n_timesteps=time_seq,
                                    n_tokens=n_tok,
                                    n_bags=n_bags,
                                    batch_size=BATCH_SIZE)
    tuner = tuners.Hyperband(
        hyper_model,
        objective="accuracy",
        max_epochs=5,
        project_name="hyperparameter-tuning",
        distribution_strategy=tf.distribute.MirroredStrategy())

    # Announce the search space
    tuner.search_space_summary()

    # And search the space
    tuner.search(dat_generator, epochs=5)

    # Get results
    tuner.results_summary()
else:
    input_layer = keras.Input(
        shape=(time_seq, n_bags),
        batch_size=BATCH_SIZE,
    )
    # NOTE: Not sure if we need to mask ragged or not, but if so, there should
    # be a masking layer here and the embedding layer should be set to ignore
    # 0 as a mask and input_dim=n_tok+1 accordingly
    emb_layer = keras.layers.Embedding(n_tok,
                                       output_dim=1,
                                       input_length=time_seq)(input_layer)

    # BUG: I think this is it? Maybe I need to look at the math again
    reshape = keras.layers.Reshape((time_seq, n_bags))(emb_layer)

    lstm_layer = keras.layers.LSTM(
        n_lstm, dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout)(reshape)

    output_dim = keras.layers.Dense(1, activation="softmax")(lstm_layer)

    model = keras.Model(input_layer, output_dim)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")

    model.summary()

    # %% Train
    # NOTE: Multiprocessing is superfluous here with epochs=1, but we could use it
    model.fit(dat_generator, use_multiprocessing=True, workers=4)
