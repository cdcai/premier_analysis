'''Runs some baseline prediction models on day-1 predictors'''
import argparse
import os
import pickle as pkl
from importlib import reload

import numpy as np
import pandas as pd
import scipy
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (auc, average_precision_score,
                             precision_recall_curve, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import tools.analysis as ta
import tools.preprocessing as tp

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--outcome",
                        type=str,
                        default="misa_pt",
                        choices=["misa_pt", "multi_class", "death"],
                        help="which outcome to use as the prediction target")
    parser.add_argument(
        '--day_one',
        help="Use only first inpatient day's worth of features (DAN only)",
        dest='day_one',
        action='store_true')
    parser.add_argument('--all_days',
                        help="Use all features in lookback period (DAN only)",
                        dest='day_one',
                        action='store_false')
    parser.set_defaults(day_one=True)
    parser.add_argument("--demog",
                        type=bool,
                        default=True,
                        help="Should the model include patient demographics?")
    parser.add_argument("--out_dir",
                        type=str,
                        default="output/",
                        help="output directory")
    parser.add_argument("--data_dir",
                        type=str,
                        default="data/data/",
                        help="path to the Premier data")
    parser.add_argument("--test_split",
                        type=float,
                        default=0.2,
                        help="Percentage of total data to use for testing")
    parser.add_argument("--validation_split",
                        type=float,
                        default=0.1,
                        help="Percentage of train data to use for validation")
    parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
    args = parser.parse_args()

    DAY_ONE_ONLY = args.day_one
    USE_DEMOG = args.demog
    OUTCOME = args.outcome
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.validation_split
    RAND = args.rand_seed

    # Setting the directories and importing the data
    output_dir = os.path.abspath(args.out_dir)
    data_dir = os.path.abspath(args.data_dir)
    pkl_dir = os.path.join(output_dir, "pkl")
    stats_dir = os.path.join(output_dir, "analysis")

    with open(os.path.join(pkl_dir, OUTCOME + "_trimmed_seqs.pkl"), "rb") as f:
        inputs = pkl.load(f)

    with open(os.path.join(pkl_dir, "all_ftrs_dict.pkl"), "rb") as f:
        vocab = pkl.load(f)

    with open(os.path.join(pkl_dir, "feature_lookup.pkl"), "rb") as f:
        all_feats = pkl.load(f)

    with open(os.path.join(pkl_dir, "demog_dict.pkl"), "rb") as f:
        demog_dict = pkl.load(f)
        demog_dict = {k: v for v, k in demog_dict.items()}

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
        demog_vocab = {k + n_features: v for k, v in demog_dict.items()}
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
    train, test = train_test_split(range(len(features)),
                                   test_size=TEST_SPLIT,
                                   stratify=y,
                                   random_state=RAND)

    train, val = train_test_split(train,
                                  test_size=VAL_SPLIT,
                                  stratify=y[train],
                                  random_state=RAND)

    # Fitting a logistic regression to the whole dataset
    lgr = LogisticRegression(max_iter=5000, random_state=RAND)
    lgr.fit(X, y)
    exp_coefs = np.exp(lgr.coef_)[0]
    top_coef = np.argsort(exp_coefs)[::-1][0:30]
    top_ftrs = [vocab[code] for code in top_coef]
    top_codes = [all_feats[ftr] for ftr in top_ftrs]

    bottom_coef = np.argsort(exp_coefs)[0:30]
    bottom_ftrs = [vocab[code] for code in bottom_coef]
    bottom_codes = [all_feats[ftr] for ftr in bottom_ftrs]

    codes = top_codes + bottom_codes
    coefs = np.concatenate([exp_coefs[top_coef], exp_coefs[bottom_coef]])
    coef_df = pd.DataFrame([codes, coefs]).transpose()
    coef_df.columns = ['feature', 'aOR']
    coef_df.sort_values('aOR', ascending=False, inplace=True)
    coef_df.to_csv(os.path.join(stats_dir, OUTCOME + '_lgr_coefs.csv'),
                   index=False)

    # And then again to the training data to get predictive performance
    lgr = LogisticRegression(max_iter=5000, random_state=RAND)
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

    # Saving the results to disk
    mod_name = stats.model.values[0]
    ta.write_stats(stats, OUTCOME)
    ta.write_preds(preds=test_preds,
                   outcome=OUTCOME,
                   mod_name=mod_name,
                   test_idx=test,
                   probs=test_probs)
