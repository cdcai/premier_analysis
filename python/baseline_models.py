'''Runs some baseline prediction models on day-1 predictors'''

import argparse
import os
import pickle as pkl
from importlib import reload

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

import tools.analysis as ta
import tools.preprocessing as tp

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome',
                        type=str,
                        default='multi_class',
                        choices=['misa_pt', 'multi_class', 'death', 'icu'],
                        help='which outcome to use as the prediction target')
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
    parser.add_argument('--use_demog',
                        type=bool,
                        default=True,
                        help='whether to iclude demographics in the features')
    parser.add_argument('--stratify',
                        type=str,
                        default='all',
                        choices=['all', 'death', 'misa_pt', 'icu'],
                        help='which label to use for the train-test split')
    parser.add_argument('--average',
                        type=str,
                        default='weighted',
                        choices=['micro', 'macro', 'weighted'],
                        help='how to average stats for multiclass predictions')
    parser.add_argument('--cohort_prefix',
                        type=str,
                        default='',
                        help='prefix for the cohort csv file, ending with _')
    parser.add_argument('--out_dir',
                        type=str,
                        help="output directory (optional)")
    parser.add_argument("--data_dir",
                        type=str,
                        help="path to the Premier data (optional)")
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

    # Setting the globals
    OUTCOME = args.outcome
    USE_DEMOG = args.use_demog
    AVERAGE = args.average
    DAY_ONE_ONLY = args.day_one
    TEST_SPLIT = args.test_split
    VAL_SPLIT = args.validation_split
    RAND = args.rand_seed
    CHRT_PRFX = args.cohort_prefix
    STRATIFY = args.stratify

    # Setting the directories and importing the data
    pwd = os.path.abspath(os.path.dirname(__file__))

    # If no args are passed to overwrite these values, use repo structure to construct
    data_dir = os.path.abspath(os.path.join(pwd, "..", "data", "data", ""))
    output_dir = os.path.abspath(os.path.join(pwd, "..", "output", ""))

    if args.data_dir is not None:
        data_dir = os.path.abspath(args.data_dir)

    if args.out_dir is not None:
        output_dir = os.path.abspath(args.out_dir)

    pkl_dir = os.path.join(output_dir, "pkl", "")
    stats_dir = os.path.join(output_dir, "analysis", "")
    probs_dir = os.path.join(stats_dir, "probs", "")

    # Create analysis dirs if it doesn't exist
    [
        os.makedirs(directory, exist_ok=True)
        for directory in [stats_dir, probs_dir, pkl_dir]
    ]

    with open(pkl_dir + CHRT_PRFX + "trimmed_seqs.pkl", "rb") as f:
        inputs = pkl.load(f)

    with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
        vocab = pkl.load(f)

    with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
        all_feats = pkl.load(f)

    with open(pkl_dir + "demog_dict.pkl", "rb") as f:
        demog_dict = pkl.load(f)
        demog_dict = {k: v for v, k in demog_dict.items()}

    # Separating the inputs and labels
    features = [t[0] for t in inputs]
    demog = [t[1] for t in inputs]
    cohort = pd.read_csv(output_dir + CHRT_PRFX + 'cohort.csv')
    labels = cohort[OUTCOME]

    # Counts to use for loops and stuff
    n_patients = len(features)
    n_features = np.max(list(vocab.keys()))
    n_classes = len(np.unique(labels))
    binary = n_classes <= 2

    # Converting the labels to an array
    y = np.array(labels, dtype=np.uint8)

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
        all_feats.update({v: v for k, v in demog_dict.items()})

    # Converting the features to a sparse matrix
    mat = lil_matrix((n_patients, n_features + 1))
    for row, cols in enumerate(features):
        mat[row, cols] = 1

    # Converting to csr because the internet said it would be faster
    X = mat.tocsr()

    # Splitting the data; 'all' will produce the same test sample
    # for every outcome (kinda nice)
    if STRATIFY == 'all':
        outcomes = ['icu', 'misa_pt', 'death']
        strat_var = cohort[outcomes].values.astype(np.uint8)
    else:
        strat_var = y

    train, test = train_test_split(range(n_patients),
                                   test_size=TEST_SPLIT,
                                   stratify=strat_var,
                                   random_state=RAND)

    # Doing a validation split for threshold-picking on binary problems
    train, val = train_test_split(train,
                                  test_size=VAL_SPLIT,
                                  stratify=strat_var[train],
                                  random_state=RAND)

    # Fitting a logistic regression to the whole dataset
    lgr = LogisticRegression(max_iter=5000, multi_class='ovr')
    lgr.fit(X, y)
    coef_list = []

    # Sorting the coefficients for
    for i in range(n_classes):
        if not binary:
            exp_coefs = np.exp(lgr.coef_)[i]
        else:
            exp_coefs = np.exp(lgr.coef_)[0]
            i = 1
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
        coef_list.append(coef_df)

    # Writing the sorted coefficients to Excel
    out_name = OUTCOME + '_lgr_'
    if DAY_ONE_ONLY:
        out_name += 'd1_'

    with pd.ExcelWriter(stats_dir + out_name + 'coefs.xlsx') as writer:
        for i, df in enumerate(coef_list):
            df.to_excel(writer, sheet_name='coef_' + str(i), index=False)

        writer.save()

    # Loading up some models to try
    mods = [
        LogisticRegression(max_iter=5000, multi_class='ovr'),
        RandomForestClassifier(n_estimators=500, n_jobs=-1),
        GradientBoostingClassifier(),
        LinearSVC(class_weight='balanced', max_iter=10000)
    ]
    mod_names = ['lgr', 'rf', 'gbc', 'svm']

    # Turning the crank like a proper data scientist
    for i, mod in enumerate(mods):
        # Fitting the model and setting the name
        mod.fit(X[train], y[train])
        mod_name = mod_names[i]
        if DAY_ONE_ONLY:
            mod_name += '_d1'

        if 'predict_proba' in dir(mod):
            if binary:
                val_probs = mod.predict_proba(X[val])[:, 1]
                val_gm = ta.grid_metrics(y[val], val_probs)
                cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
                test_probs = mod.predict_proba(X[test])[:, 1]
                test_preds = ta.threshold(test_probs, cutpoint)
                stats = ta.clf_metrics(y[test],
                                       test_probs,
                                       cutpoint=cutpoint,
                                       mod_name=mod_name,
                                       average=args.average)
                ta.write_preds(preds=test_preds,
                               outcome=OUTCOME,
                               mod_name=mod_name,
                               test_idx=test,
                               probs=test_probs)
            else:
                cutpoint = None
                test_probs = mod.predict_proba(X[test])
                test_preds = mod.predict(X[test])
                stats = ta.clf_metrics(y[test],
                                       test_probs,
                                       mod_name=mod_name,
                                       average=args.average)
                ta.write_preds(preds=test_preds,
                               probs=np.max(test_probs, axis=1),
                               outcome=OUTCOME,
                               mod_name=mod_name,
                               test_idx=test)

            # Write out multiclass probs as pkl
            probs_file = mod_name + '_' + OUTCOME + '.pkl'
            prob_out = {'cutpoint': cutpoint, 'probs': test_probs}

            with open(probs_dir + probs_file, 'wb') as f:
                pkl.dump(prob_out, f)

        else:
            test_preds = mod.predict(X[test])
            stats = ta.clf_metrics(y[test],
                                   test_preds,
                                   mod_name=mod_name,
                                   average=args.average)
            ta.write_preds(preds=test_preds,
                           outcome=OUTCOME,
                           mod_name=mod_name,
                           test_idx=test)

        # Saving the results to disk
        ta.write_stats(stats, OUTCOME)
