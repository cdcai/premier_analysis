'''Runs some baseline prediction models on day-1 predictors'''
import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import scipy
import os

from importlib import reload
from scipy.sparse import lil_matrix
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score

import tools.preprocessing as tp
import tools.analysis as ta


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outcome',
                        type=str,
                        default='multi_class',
                        choices=['misa_pt', 'multi_class', 'death', 'icu'],
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
    DAY_ONE_ONLY = True if args.history != 'all' else False
    
    # Setting the directories and importing the data
    output_dir = os.path.abspath(args.out_dir) + '/'
    data_dir = os.path.abspath(args.data_dir) + '/'
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
        demog_vocab = {k + n_features:v for k,v in demog_dict.items()}
        vocab.update(demog_vocab)
        n_features = np.max([np.max(l) for l in features])
        all_feats.update({v:v for k,v in demog_dict.items()})
    
    # Converting the features to a sparse matrix
    mat = lil_matrix((n_patients, n_features + 1))
    for row, cols in enumerate(features):
        mat[row, cols] = 1
    
    # Converting to csr because the internet said it would be faster
    X = mat.tocsr()
    
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
        coefs = np.concatenate([exp_coefs[top_coef],
                                exp_coefs[bottom_coef]])
        coef_df = pd.DataFrame([codes, coefs]).transpose()
        coef_df.columns = ['feature', 'aOR']
        coef_df.sort_values('aOR', ascending=False, inplace=True)
        coef_list.append(coef_df)
    
    # Writing the sorted coefficients to Excel
    out_name = OUTCOME + '_lgr_'
    if DAY_ONE_ONLY:
        out_name += 'd1_'
    
    writer = pd.ExcelWriter(stats_dir + out_name + 'coefs.xlsx')
    for i, df in enumerate(coef_list):
        df.to_excel(writer, 
                    sheet_name='coef_' + str(i), 
                    index=False)
    
    writer.save()
    
    # Loading up some models to try
    mods = [
        LogisticRegression(max_iter=5000, multi_class='ovr'),
        RandomForestClassifier(n_estimators=500, n_jobs=-1),
        GradientBoostingClassifier(),
        LinearSVC(class_weight='balanced', max_iter=5000)       
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
            probs_file = 'probs/' + mod_name + '_' + OUTCOME + '.pkl'
            prob_out = {'cutpoint': cutpoint, 'probs': test_probs}
            pkl.dump(prob_out, open(stats_dir + probs_file, 'wb'))
                
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
