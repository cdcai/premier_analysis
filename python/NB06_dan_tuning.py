# Databricks notebook source
# MAGIC %pip install keras-tuner

# COMMAND ----------

# Imports
import os
import pickle as pkl

import numpy as np
import tensorflow.keras as keras
import kerastuner
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from tensorflow.keras.callbacks import TensorBoard

from tools import keras as tk
import tools.preprocessing as tp

# COMMAND ----------

# GLOBALS   
DAY_ONE_ONLY = True
TIME_SEQ = 225
TARGET = "misa_pt"
BATCH_SIZE = 128
EPOCHS = 10
MAX_TRIALS = 500
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2021
TB_UPDATE_FREQ = 200
WEIGHTED_LOSS = False

# Paths
# BUG: This use to be a cool hack to alway return the root dir
# of the repo, but that sometimes fails, so just set your PWD here
# or leave as an empty string if that's where this is running.
# all paths to output/ and data/ are constructed relative to that
pwd = ""

output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'
tensorboard_dir = os.path.abspath(
    os.path.join(output_dir, "model_checkpoints"))
pkl_dir = os.path.join(output_dir, "pkl")
stats_dir = os.path.join(output_dir, "analysis")

# Create analysis dir if it doesn't exist
os.makedirs(stats_dir, exist_ok=True)
os.makedirs(tensorboard_dir, exist_ok=True)

# COMMAND ----------

# Data load
with open(os.path.join(pkl_dir + "/trimmed_seqs.pkl"), "rb") as f:
    inputs = pkl.load(f)

with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
    vocab = pkl.load(f)

with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
    all_feats = pkl.load(f)

with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
    demog_lookup = pkl.load(f)

# Determining number of vocab entries
N_VOCAB = len(vocab) + 1
N_DEMOG = len(demog_lookup) + 1
MAX_DEMOG = max(len(x) for _, x, _ in inputs)
#N_CLASS = max(x for _, _, x in inputs) + 1

# COMMAND ----------

# Model Metrics and callbacks
callbacks = [
    TensorBoard(
    log_dir=os.path.join(tensorboard_dir, "dan_hp_tune_tb", ""),
    histogram_freq=1,
    profile_batch=0,
    write_graph=False,
    update_freq=TB_UPDATE_FREQ
    ),
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                min_delta=0,
                                patience=3,
                                restore_best_weights=True,
                                mode="min")
]

# Create some metrics
metrics = [
    keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
    keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC")
]

# COMMAND ----------

# TTV
# Splitting the data
train, test = train_test_split(
    range(len(inputs)),
    test_size=TEST_SPLIT,
    stratify=[labs for _, _, labs in inputs],
    random_state=RAND)

train, validation = train_test_split(
    train,
    test_size=VAL_SPLIT,
    stratify=[samp[2] for i, samp in enumerate(inputs) if i in train],
    random_state=RAND)

# COMMAND ----------

if DAY_ONE_ONLY:
    # Optionally limiting the features to only those from the first day
    # of the actual COVID visit
    features = [l[0][-1] for l in inputs]
else:
    features = [tp.flatten(l[0]) for l in inputs]

new_demog = [[i + N_VOCAB - 1 for i in l[1]] for l in inputs]
features = [
    features[i] + new_demog[i] for i in range(len(features))
]
demog_vocab = {k: v + N_VOCAB - 1 for k, v in demog_lookup.items()}
vocab.update(demog_vocab)
N_VOCAB = np.max([np.max(l) for l in features]) + 1

# Making the variables
X = keras.preprocessing.sequence.pad_sequences(features, padding='post')
y = np.array([l[2] for l in inputs])

N_FEATS = X.shape[1]

# COMMAND ----------

y

# COMMAND ----------

classes = np.unique([labs for _, _, labs in inputs]).tolist()
classes

# COMMAND ----------

classes = np.unique([labs for _, _, labs in inputs]).tolist()

if WEIGHTED_LOSS:
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=[labs for _, _, labs in inputs],
    )

    class_weights = dict(zip(classes, class_weights))

    print(class_weights)

# COMMAND ----------

N_CLASS = 3


hyper_model = tk.DANHyper(
    vocab_size = N_VOCAB,
    input_size=N_FEATS,
    metrics = metrics,
    n_classes = N_CLASS
)

tuner = kerastuner.tuners.bayesian.BayesianOptimization(
    hyper_model,
    max_trials=MAX_TRIALS,
    objective="val_loss",
    project_name="dan_hp_tune",
    directory=tensorboard_dir
)

# COMMAND ----------

tuner.search_space_summary()

# COMMAND ----------

if N_CLASS > 2:
    # We have to pass one-hot labels for model fit, but CLF metrics
    # will take indices
    y_one_hot = np.eye(N_CLASS)[y]

    tuner.search(X[train], y_one_hot[train],
                validation_data=(X[validation], y_one_hot[validation]),
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                callbacks=callbacks
    )
else:
    tuner.search(X[train], y[train],
            validation_data=(X[validation], y[validation]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            #callbacks=callbacks
    )

# COMMAND ----------

y[train]

# COMMAND ----------

tuner.results_summary()



# COMMAND ----------

# Pull the best model
best_hp = tuner.get_best_hyperparameters()[0]
best_model = tuner.hypermodel.build(best_hp)

best_model.summary()

# COMMAND ----------

best_model.save(os.path.join(tensorboard_dir, "best", "dan"))
