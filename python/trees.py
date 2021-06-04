'''Gets feature importances from an RF and GB-trees'''

import numpy as np
import pandas as pd
import pickle as pkl
import argparse
import scipy
import os

from importlib import reload
from scipy.sparse import lil_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, average_precision_score

import tools.preprocessing as tp
import tools.analysis as ta



# Setting the globals
USE_DEMOG = True
AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TOP_N = 50

# Setting the directories and importing the data
output_dir = 'C:/Users/yle4/code/github/premier_analysis/python/output/'
pkl_dir = output_dir + "pkl/"
stats_dir = output_dir + 'analysis/'

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

with open(pkl_dir + "demog_dict.pkl", "rb") as f:
    demog_dict = pkl.load(f)
    demog_dict = {k:v for v, k in demog_dict.items()}

writer = pd.ExcelWriter(stats_dir + 'importances.xlsx')

for OUTCOME in ['death', 'multi_class', 'misa_pt']:
    # Load the outcome-specific data
    with open(pkl_dir + OUTCOME + "_trimmed_seqs.pkl", "rb") as f:
        inputs = pkl.load(f)
    
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
    
    # Setting things up for the loop
    gb = GradientBoostingClassifier(n_estimators=500)
    rf = RandomForestClassifier(n_estimators=500, 
                                oob_score=True,
                                n_jobs=-1)
    mods = [gb, rf]
    mod_names = ['gb', 'rf']
    imp_df = pd.DataFrame(np.zeros(shape=(TOP_N, len(mods))),
                        columns=mod_names)
    
    for i, mod in enumerate(mods):
        mod.fit(X, y)
        imps = mod.feature_importances_
        f_args = np.argsort(imps)[::-1]
        top_f = [all_feats[f] for f in [vocab[i] for i in f_args[:TOP_N]]]
        imp_df[mod_names[i]] = top_f
    
    imp_df.to_excel(writer,
                    sheet_name=OUTCOME,
                    index=False)

writer.save()
