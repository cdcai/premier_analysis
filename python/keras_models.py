"""
Starter Keras model
"""
# %%
import csv
import itertools
import os
import pickle as pkl
import random
from datetime import datetime

import kerastuner.tuners as tuners
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

from tools import keras as tk
from tools.analysis import grid_metrics

# %% Globals
TIME_SEQ = 2358
TARGET = "misa_pt"
RAGGED = True
LSTM_DROPOUT = 0.2
# NOTE: Recurrent dropout is advisable, but it also means
# you forgoe CuDNN-optimization for the LSTM, so it will train
# about 1/3 slower on GPU
LSTM_RECURRENT_DROPOUT = 0.4
N_LSTM = 128
HYPER_TUNING = False
# NOTE: I maxed out my GPU running 32, 20 ran ~7.8GB on GPU
BATCH_SIZE = 32
EPOCHS = 20
SUBSAMPLE = True
SAMPLE_FRAC = 0.01
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2020
TB_UPDATE_FREQ = 100
# %% Load in Data
output_dir = os.path.abspath("output/") + "/"
tensorboard_dir = os.path.abspath("data/model_checkpoints/") + "/"
data_dir = os.path.abspath("data/data/") + "/"
pkl_dir = output_dir + "pkl/"

with open(pkl_dir + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "pat_data.pkl", "rb") as f:
    y_ = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

# %% Save Embedding metadata
# We can use this with tensorboard to visualize the embeddings
with open(tensorboard_dir + "emb_metadata.tsv", "w") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerows(zip(["id"], ["word"], ["desc"]))
    writer.writerows(zip([0], ["OOV"], ["Padding/OOV"]))
    for key, value in vocab.items():
        writer.writerow([value, key, all_feats[key]])
# %% Determining number of vocab entries
N_VOCAB = len(vocab) + 1

# %% Subsampling if desired
if SUBSAMPLE:
    _, inputs, _, _ = train_test_split(
        inputs,
        [labs for _, labs in inputs],
        test_size=SAMPLE_FRAC,
        random_state=RAND,
        stratify=[labs for _, labs in inputs],
    )
# %% Split into test/train
train, test, _, _ = train_test_split(
    inputs,
    [labs for _, labs in inputs],
    test_size=TEST_SPLIT,
    random_state=RAND,
    stratify=[labs for _, labs in inputs],
)

# Further split into train/validation
train, validation, _, _ = train_test_split(
    train,
    [labs for _, labs in train],
    test_size=VAL_SPLIT,
    random_state=RAND,
    stratify=[labs for _, labs in train],
)

# %% Compute class weights
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique([labs for _, labs in train]),
    y=[labs for _, labs in train],
)

class_weights = dict(zip([0, 1], class_weights))

# %%
train_gen = tk.create_ragged_data(train,
                                  max_time=TIME_SEQ,
                                  epochs=EPOCHS,
                                  batch_size=BATCH_SIZE)

validation_gen = tk.create_ragged_data(validation,
                                       max_time=TIME_SEQ,
                                       epochs=EPOCHS,
                                       batch_size=BATCH_SIZE)

test_gen = tk.create_ragged_data(test,
                                 max_time=TIME_SEQ,
                                 epochs=1,
                                 batch_size=BATCH_SIZE)
# %% Create data generator for On-the-fly batch generation
# train_gen = tk.DataGenerator(
#     train,
#     max_time=TIME_SEQ,
#     ragged=RAGGED,
#     resampler=RandomOverSampler(sampling_strategy="minority"),
#     batch_size=BATCH_SIZE,
# )
# # %%
# validation_gen = tk.DataGenerator(validation,
#                                   max_time=TIME_SEQ,
#                                   ragged=RAGGED,
#                                   batch_size=BATCH_SIZE)

# test_gen = tk.DataGenerator(test,
#                             max_time=TIME_SEQ,
#                             ragged=RAGGED,
#                             batch_size=BATCH_SIZE)

# %%
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
    )

    # Announce the search space
    tuner.search_space_summary()

    # And search the space
    tuner.search(
        train_gen,
        validation_data=validation_gen,
        epochs=5,
    )

    # Get results
    tuner.results_summary()
else:

    input_layer = keras.Input(shape=(None if RAGGED else TIME_SEQ, None),
                              ragged=RAGGED,
                              batch_size=BATCH_SIZE)
    # Feature Embeddings
    emb1 = keras.layers.Embedding(N_VOCAB,
                                  output_dim=512,
                                  mask_zero=True,
                                  name="Feature_Embeddings")(input_layer)
    # Average weights of embedding
    emb2 = keras.layers.Embedding(N_VOCAB,
                                  output_dim=1,
                                  mask_zero=True,
                                  name="Average_Embeddings")(input_layer)

    # Multiply and average
    if RAGGED:
        # NOTE: I think these are the equivalent ragged-aware ops
        # but that could be incorrect
        mult = keras.layers.Multiply(name="Embeddings_by_Average")(
            [emb1, emb2])
        avg = keras.layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=2),
                                  name="Averaging")(mult)
    else:
        mult = keras.layers.Multiply(name="Embeddings_by_Average")(
            [emb1, emb2])
        avg = keras.backend.mean(mult, axis=2)

    lstm_layer = keras.layers.LSTM(
        N_LSTM,
        dropout=LSTM_DROPOUT,
        recurrent_dropout=LSTM_RECURRENT_DROPOUT,
        name="Recurrent",
    )(avg)

    output_dim = keras.layers.Dense(1, activation="sigmoid",
                                    name="Output")(lstm_layer)

    model = keras.Model(input_layer, output_dim)

    model.compile(optimizer="adam", loss="binary_crossentropy", metrics="acc")

    model.summary()

    # Create Tensorboard callback
    tb_callback = TensorBoard(
        log_dir=tensorboard_dir + "/" + TARGET + "/" +
        datetime.now().strftime("%Y%m%d-%H%M%S") + "/",
        histogram_freq=1,
        update_freq=TB_UPDATE_FREQ,
        # embeddings_freq=1,
        # embeddings_metadata=output_dir + "emb_metadata.tsv",
    )

    # Create model checkpoint callback
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=tensorboard_dir + "/" + TARGET + "/" +
        "weights.{epoch:02d}-{val_loss:.2f}.hdf5",
        save_weights_only=True,
        monitor="val_acc",
        mode="max",
        save_best_only=True,
    )

    # Create early stopping callback
    stopping_checkpoint = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0,
        patience=2,
        mode="auto",
        restore_best_weights=True,
    )

    # %% Train
    fitting = model.fit(
        train_gen,
        validation_data=validation_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=[
            tb_callback, model_checkpoint_callback, stopping_checkpoint
        ],
    )

    # Test
    test_loss, test_acc = model.evaluate(test_gen)

    print("Test Loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(test_acc))

    # %% F1, etc
    y_pred = model.predict(test_gen)
    # y_pred = (pred >= 0.5).astype(int)
    y_true = [lab for _, lab in test]

    # Resizing for output which is divisible by BATCH_SIZE
    y_true = y_true[0:y_pred.shape[0]]
    output = grid_metrics(y_true, y_pred)
    print(output)

    print(output)
    print("ROC-AUC: {}".format(roc_auc_score(y_true, y_pred)))

# %%
