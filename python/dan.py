'''Trains a deep averaging network (DAN) on the visit sequences'''

import numpy as np
import pandas as pd
import pickle as pkl
import tensorflow as tf
import argparse
import scipy
import os

from importlib import reload
from tensorflow import keras as keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score
from scipy.sparse import lil_matrix

import tools.keras as tk
import tools.analysis as ta
import tools.preprocessing as tp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome',
                        type=str,
                        default='multi_class',
                        choices=['misa_pt', 'multi_class', 'death'],
                        help='which outcome to use as the prediction target')
    parser.add_argument('--history',
                        type=str,
                        default='target_visit_only',
                        choices=['all', 'target_visit_only'],
                        help='how much patient history to use in constructing\
                         the visit sequences')
    parser.add_argument('--use_demog',
                        type=bool,
                        default=True,
                        help='whether to iclude demographics in the features')
    parser.add_argument('--average',
                        type=str,
                        default='weighted',
                        choices=['micro', 'macro', 'weighted'],
                        help='how to average stats for multiclass predictions')
    parser.add_argument('--max_seq',
                        type=int,
                        default=225,
                        help='maximum number of days in a visit sequence')
    parser.add_argument('--out_dir',
                        type=str,
                        default='output/',
                        help='output directory')
    parser.add_argument('--data_dir',
                        type=str,
                        default='..data/data/',
                        help='path to the Premier data')
    args = parser.parse_args()
    
    # Setting the globals
    OUTCOME = args.outcome
    USE_DEMOG = args.use_demog
    AVERAGE = args.average
    MAX_SEQ = args.max_seq
    DAY_ONE_ONLY = True if args.history != 'all' else False

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
    n_classes = len(np.unique(labels))

    # Revert to single output for binary problems; kludge
    if n_classes == 2:
        n_classes = 1

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
                                   test_size=0.2,
                                   stratify=y,
                                   random_state=2021)
    
    # Doing a validation split for threshold-picking on binary problems
    train, val = train_test_split(train,
                                  test_size=0.2,
                                  stratify=y[train],
                                  random_state=2021)
    
    # Setting up the callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=2),
        keras.callbacks.TensorBoard(log_dir=output_dir + '/logs'),
    ]

    metrics = [
        keras.metrics.AUC(num_thresholds=int(1e5), name="ROC-AUC"),
        keras.metrics.AUC(num_thresholds=int(1e5), curve="PR", name="PR-AUC"),
    ]

    # Settnig the loss
    if n_classes > 1:
        loss = 'categorical_crossentropy'
        y_mat = ta.onehot_matrix(y)
    else:
        loss = 'binary_crossentropy'
        y_mat = y

    # Setting and training up the model
    mod = tk.DAN(vocab_size=n_features + 1,
                 n_classes=n_classes,
                 ragged=False,
                 input_length=MAX_SEQ)
    mod.compile(optimizer='adam',
                loss=loss,
                metrics=metrics)
    mod.fit(X[train], y_mat[train],
            batch_size=32,
            epochs=30,
            validation_data=(X[val], y_mat[val]),
            callbacks=callbacks)

    # Getting the model's predictions
    mod_name = 'dan'
    if DAY_ONE_ONLY:
        mod_name += '_d1'

    # Getting predictions in the binary case
    if n_classes < 2:
        val_probs = mod.predict(X[val]).flatten()
        val_gm = ta.grid_metrics(y[val], val_probs)
        cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
        test_probs = mod.predict(X[test]).flatten()
        out_probs = test_probs
        test_preds = ta.threshold(test_probs, cutpoint)
        stats = ta.clf_metrics(y[test],
                               test_probs,
                               cutpoint=cutpoint,
                               mod_name=mod_name)
    else:
        cutpoint = 0.5
        test_probs = mod.predict(X[test])
        test_preds = np.argmax(test_probs, axis=1)
        out_probs = ta.max_probs(test_probs, maxes=test_preds)
        stats = ta.clf_metrics(y[test],
                               test_probs,
                               mod_name=mod_name)
    
    probs_file = 'probs/' + mod_name + '_' + OUTCOME + '.pkl'
    prob_out = {'cutpoint': cutpoint, 'probs': test_probs}
    pkl.dump(prob_out, open(stats_dir + probs_file, 'wb'))

    # Writing the results to disk
    ta.write_stats(stats, OUTCOME)
    ta.write_preds(preds=test_preds,
                   probs=out_probs,
                   outcome=OUTCOME,
                   mod_name=mod_name,
                   test_idx=test)
