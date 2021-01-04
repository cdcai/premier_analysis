'''Runs some baseline prediction models on day-1 predictors'''
import numpy as np
import pandas as pd
import pickle as pkl
import scipy
import os

from importlib import reload
from scipy.sparse import lil_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score

import tools.preprocessing as tp
import tools.analysis as ta


# Globals
DAY_ONE_ONLY = True

# Setting the directories and importing the data
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("..data/data/") + "/"
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

# Optionally limiting the features to only those from the first day
# of the actual COVID visit
if DAY_ONE_ONLY:
    features = [l[-1] for l in features]
else:
    features = [tp.flatten(l) for l in features]

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)

# Converting the features to a sparse matrix
mat = lil_matrix((len(features), len(vocab.keys()) + 1))
for row, cols in enumerate(features):
    mat[row, cols] = 1

# Converting to csr because the internet said it would be faster
X = mat.tocsr()

# Splitting the data
train, test = train_test_split(range(X.shape[0]),
                               test_size=0.25,
                               stratify=y,
                               random_state=2020)

train, val = train_test_split(train,
                              test_size=1/3,
                              stratify=y[train],
                              random_state=2020)

# Fitting a logistic regression to the whole dataset
lgr = LogisticRegression(max_iter=5000)
lgr.fit(X, y)
exp_coefs = np.exp(lgr.coef_)[0]
top_coef = np.argsort(exp_coefs)[::-1][0:30]
top_ftrs = [vocab[code] for code in top_coef]
top_codes = [all_feats[ftr] for ftr in top_ftrs]

bottom_coef = np.argsort(exp_coefs)[0:30]
bottom_ftrs = [vocab[code] for code in bottom_coef]
bottom_codes = [all_feats[ftr] for ftr in bottom_ftrs]

codes = top_codes + bottom_codes
coefs = np.concatenate([exp_coefs[top_coef],
                        exp_coefs[bottom_coef]])
coef_df = pd.DataFrame([codes, coefs]).transpose()
coef_df.to_csv(output_dir + 'lgr_coefs.csv', index=False)

# And then again to the training data to get predictive performance
lgr = LogisticRegression(max_iter=5000)
lgr.fit(X[train], y[train])
val_probs = lgr.predict_proba(X[val])[:, 1]
val_gm = ta.grid_metrics(y[val], val_probs)
f1_cut = val_gm.cutoff.values[np.argmax(val_gm.f1)]
test_probs = lgr.predict_proba(X[test])[:, 1]

lgr_roc = roc_curve(y[test], test_probs)
lgr_auc = auc(lgr_roc[0], lgr_roc[1])
lgr_pr = average_precision_score(y[test], test_probs)
lgr_stats = ta.clf_metrics(y[test],
                           ta.threshold(test_probs, f1_cut))
lgr_stats['auc'] = lgr_auc
lgr_stats['ap'] = lgr_pr

lgr_stats.to_csv(output_dir + 'lgr_stats.csv', index=False)
