'''Trains a deep averaging network (DAN) on the visit sequences'''

import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import scipy
import os

from importlib import reload
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score

import tools.keras as tk
import tools.analysis as ta


# Globals
DAY_ONE_ONLY = True
USE_DEMOG = True
OUTCOME = 'death'
MAX_SEQ = 225

# Setting the directories and importing the data
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("..data/data/") + "/"
tensorboard_dir = os.path.abspath("../data/model_checkpoints/") + "/"
pkl_dir = output_dir + "pkl/"
stats_dir = output_dir + 'analysis/'

with open(pkl_dir + OUTCOME + "_trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

with open(pkl_dir + "demog_dict.pkl", "rb") as f:
    demog_dict = pkl.load(f)
    demog_dict = {k:v for v, k in demog_dict.items()}

# Separating the inputs and labels
features = [t[0] for t in inputs]
demog = [t[1] for t in inputs]
labels = [t[2] for t in inputs]

# Counts to use for loops and stuff
n_patients = len(features)
n_features = np.max(list(vocab.keys()))

# Optionally limiting the features to only those from the first day
# of the actual COVID visit
if DAY_ONE_ONLY:
    features = [l[-1] for l in features]
else:
    features = [tp.flatten(l) for l in features]

# Optionally mixing in the demographic features
if USE_DEMOG:
    new_demog = [[i + n_features for i in l] for l in demog]
    features = [features[i] + new_demog[i] for i in range(n_patients)]
    demog_vocab = {k + n_features:v for k,v in demog_dict.items()}
    vocab.update(demog_vocab)
    n_features = np.max([np.max(l) for l in features])

# Making the variables
X = tf.keras.preprocessing.sequence.pad_sequences(features,
                                                  maxlen=225,
                                                  padding='post')
y = np.array(labels, dtype=np.uint8)

# Splitting the data
train, test = train_test_split(range(n_patients),
                               test_size=0.5,
                               stratify=y,
                               random_state=2020)

val, test = train_test_split(test,
                             test_size=0.5,
                             stratify=y[test],
                             random_state=2020)

# Setting up the callbacks
callbacks = [
    keras.callbacks.EarlyStopping(patience=2),
    keras.callbacks.TensorBoard(log_dir=output_dir + '/logs'),
]

metrics = [
    keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
    keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC"),
]

# Setting and training up the model
mod = tk.DAN(vocab_size=n_features + 1,
             ragged=False,
             input_length=MAX_SEQ)
mod.compile(optimizer='adam',
            loss='binary_crossentropy',
            metrics=metrics)
mod.fit(X[train], y[train],
        batch_size=32,
        epochs=20,
        validation_data=(X[val], y[val]),
        callbacks=callbacks)

# Getting the model's predictions
val_probs = mod.predict(X[val]).flatten()
val_gm = ta.grid_metrics(y[val], val_probs)
f1_cut = val_gm.cutoff.values[np.argmax(val_gm.f1)]
test_probs = mod.predict(X[test]).flatten()
test_preds = ta.threshold(test_probs, f1_cut)

# Writing the results to disk

