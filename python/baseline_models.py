'''Runs some baseline prediction models on day-1 predictors'''
import numpy as np
import pandas as pd
import pickle
import scipy
import os
import sys

from importlib import reload
from scipy.sparse import lil_matrix
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score

import tools.preprocessing as tp
import tools.analysis as ta


# Setting the directories
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("data/data/") + "/"
pkl_dir = output_dir + "pkl/"

with open(pkl_dir + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

# Separating the inputs and labels
features = [t[0] for t in inputs]
labels = [t[1] for t in inputs]

# Dropping anything with missing features
good = np.where([len(doc) > 0 for doc in features])

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)[good]

# Converting the features to a sparse matrix
mat = lil_matrix((len(features), len(vocab.keys()) + 1))
for row, cols in features:
    mat[row, cols] = 1

X = mat[good]

# Splitting the data
train, test = train_test_split(range(X.shape[0]),
                               test_size=0.25,
                               stratify=y))

