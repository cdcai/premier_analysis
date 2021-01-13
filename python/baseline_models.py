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
DAY_ONE_ONLY = False
USE_DEMOG = True
OUTCOME = 'death'

# Setting the directories and importing the data
output_dir = os.path.abspath("output/") + "/"
data_dir = os.path.abspath("..data/data/") + "/"
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

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)

# Converting the features to a sparse matrix
mat = lil_matrix((n_patients, n_features + 1))
for row, cols in enumerate(features):
    mat[row, cols] = 1

# Converting to csr because the internet said it would be faster
X = mat.tocsr()

# Splitting the data
train, test = train_test_split(range(n_patients),
                               test_size=0.5,
                               stratify=y,
                               random_state=2020)

val, test = train_test_split(test,
                             test_size=0.5,
                             stratify=y[test],
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
coef_df.columns = ['feature', 'aOR']
coef_df.sort_values('aOR', ascending=False, inplace=True)
coef_df.to_csv(stats_dir + OUTCOME + '_lgr_coefs.csv', index=False)

# And then again to the training data to get predictive performance
lgr = LogisticRegression(max_iter=5000)
lgr.fit(X[train], y[train])
val_probs = lgr.predict_proba(X[val])[:, 1]
val_gm = ta.grid_metrics(y[val], val_probs)
f1_cut = val_gm.cutoff.values[np.argmax(val_gm.f1)]
test_probs = lgr.predict_proba(X[test])[:, 1]
test_preds = ta.threshold(test_probs, f1_cut)
stats = ta.clf_metrics(y[test],
                       test_probs,
                       preds_are_probs=True,
                       cutpoint=f1_cut,
                       mod_name='lgr')

if DAY_ONE_ONLY:
    stats['model'] += '_d1'

# Writing the stats to disk
stats_filename = OUTCOME + '_stats.csv'
if stats_filename in os.listdir(stats_dir):
    stats_df = pd.read_csv(stats_dir + stats_filename)
    stats_df = pd.concat([stats_df, stats], axis=0)
    stats_df.to_csv(stats_dir + stats_filename, index=False)
else:
    stats.to_csv(stats_dir + stats_filename, index=False)

# Writing the test predictions to the test predictions CSV
preds_filename = OUTCOME + '_preds.csv'
if preds_filename in os.listdir(stats_dir):
    preds_df = pd.read_csv(stats_dir + preds_filename)
else:
    preds_df = pd.read_csv(output_dir + OUTCOME + '_cohort.csv')
    preds_df = preds_df.iloc[test, :]

preds_df['lgr_prob'] = test_probs
preds_df['lgr_pred'] = test_preds
preds_df.to_csv(stats_dir + preds_filename, index=False)

