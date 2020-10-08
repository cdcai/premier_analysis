'''Runs some baseline prediction models on day-1 predictors'''
import numpy as np
import pandas as pd
import pickle
import scipy
import os
import sys

from importlib import reload
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tools.preprocessing as tp
import tools.analysis as ta

# Setting the directories
prem_dir = 'data/data/'
out_dir = 'output/'
parq_dir = out_dir + 'parquet/'
pkl_dir = out_dir + 'pkl/'

# Loading the main data file
pat = pd.read_parquet(parq_dir + 'd1.parquet')

# Loading the feature lookup dicts
ftr_dict = pickle.load(open(pkl_dir + 'feature_lookup.pkl', 'rb'))
vec_dict = pickle.load(open(pkl_dir + 'vec_vocab.pkl', 'rb'))
vec_dict = {v: k for k, v in vec_dict.items()}

# Limiting to people with LOS > 1
day_plus = np.where(pat.los > 1)[0]
pat = pat.iloc[day_plus, :]

# Loading the binary features
X = load_npz(out_dir + 'npz/features.npz')
X = X[day_plus, :]

# Setting the target
y = pat.death.values
p_y = y.sum() / len(y)

# Splitting the data
train, test = train_test_split(range(X.shape[0]))

# Running an l1 logistic regression
l1 = LogisticRegression(penalty='l1',
                        class_weight='balanced')
l1.fit(X[train], y[train])
l1_probs = l1.predict_proba(X[test])[:, 1]
l1_preds = ta.threshold(l1_probs)
l1_stats = ta.clf_metrics(y[test], l1_preds)
l1_stats['auc'] = roc_auc_score(y[test], l1_probs)
l1_stats['brier'] = ta.brier_score(y[test], l1_probs)

# Running an SVM with a linear kernel
svm = LinearSVC(class_weight='balanced',
                max_iter=5000)
svm.fit(X[train], y[train])
svm_preds = svm.predict(X[test])
svm_stats = ta.clf_metrics(y[test], svm_preds)

# Running a random forest
rf = RandomForestClassifier(n_estimators=100,
                            class_weight='balanced',
                            n_jobs=-1)
rf.fit(X[train], y[train])
rf_probs = rf.predict_proba(X[test])[:, 1]
rf_preds = ta.threshold(rf_probs, p_y)
rf_stats = ta.clf_metrics(y[test], rf_preds)
rf_stats['auc'] =  roc_auc_score(y[test], rf_probs)
rf_stats['brier'] = ta.brier_score(y[test], rf_probs)

# And running a GBM
gbm = GradientBoostingClassifier(n_estimators=500)
gbm.fit(X[train], y[train])
gbm_probs = gbm.predict_proba(X[test])[:, 1]
gbm_preds = ta.threshold(gbm_probs, p_y)
gbm_stats = ta.clf_metrics(y[test], gbm_preds)
gbm_stats['auc'] = roc_auc_score(y[test], gbm_probs)
gbm_stats['brier'] = ta.brier_score(y[test], gbm_probs)

# Concatenating the results and saving them to disk
stats = pd.concat([l1_stats, svm_stats, rf_stats, gbm_stats], axis=0)
stats = stats.drop(['true_prev', 'pred_prev',
                    'prev_diff', 'rel_prev_diff'], axis=1)
stats['model'] = ['lasso', 'svm', 'rf', 'gbm']
stats.to_csv(out_dir + 'baseline_stats.csv', index=False)
