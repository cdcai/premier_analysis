# Databricks notebook source
!pip install mlflow --quiet
!pip install mljar-supervised --quiet

# COMMAND ----------

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
import mlflow

# COMMAND ----------

import mlflow

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='388290745206631',
  label='Experiment ID'
)

dbutils.widgets.dropdown("outcome","icu",["misa_pt", "multi_class", "death", "icu"])
OUTCOME = dbutils.widgets.get("outcome")

dbutils.widgets.dropdown("demographics", "True", ["True", "False"])
USE_DEMOG = dbutils.widgets.get("demographics")
if USE_DEMOG == "True": DEMOG = True
else: USE_DEMOG = False

dbutils.widgets.dropdown("stratify", "all", ['all', 'death', 'misa_pt', 'icu'])
STRATIFY = dbutils.widgets.get("stratify")

dbutils.widgets.dropdown("average", "weighted", ['micro', 'macro', 'weighted'])
AVERAGE = dbutils.widgets.get("average")

dbutils.widgets.dropdown("experimenting", "False",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id

# COMMAND ----------

# Setting the globals
#OUTCOME = 'misa_pt'
#USE_DEMOG = True
#STRATIFY ='all'

AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2022
CHRT_PRFX = ''

# Setting the directories and importing the data
# If no args are passed to overwrite these values, use repo structure to construct
# Setting the directories
output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'

if data_dir is not None:
    data_dir = os.path.abspath(data_dir)

if output_dir is not None:
    output_dir = os.path.abspath(output_dir)

pkl_dir = os.path.join(output_dir, "pkl", "")
stats_dir = os.path.join(output_dir, "analysis", "")
probs_dir = os.path.join(stats_dir, "probs", "")

# COMMAND ----------

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

# COMMAND ----------

# Separating the inputs and labels
features = [t[0] for t in inputs]
demog = [t[1] for t in inputs]
cohort = pd.read_csv(os.path.join(output_dir, CHRT_PRFX, 'cohort.csv'))
labels = cohort[OUTCOME]

# Counts to use for loops and stuff
n_patients = len(features)
n_features = np.max(list(vocab.keys()))
n_classes = len(np.unique(labels))
binary = n_classes <= 2

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)

# COMMAND ----------

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

# COMMAND ----------

train_set, test = train_test_split(range(n_patients),
                               test_size=TEST_SPLIT,
                               stratify=strat_var,
                               random_state=RAND)

# Doing a validation split for threshold-picking on binary problems
train, val = train_test_split(train_set,
                              test_size=VAL_SPLIT,
                              stratify=strat_var[train_set],
                              random_state=RAND)

# COMMAND ----------

suffix = "_outcome_"+OUTCOME+"_stratify_"+STRATIFY
if EXPERIMENTING == True:
    X_train = X[train][:100]
    X_val = X[val][:100]
    X_test = X[test][:100]

    y_train = y[train][:100]
    y_val = y[val][:100]
    y_test = y[test][:100]
else:
    X_train = X[train]
    X_val = X[val]
    X_test = X[test]

    y_train = y[train]
    y_val = y[val]
    y_test = y[test]

# COMMAND ----------

X_train.shape

# COMMAND ----------

from datetime import datetime
date_time = str(datetime.now()).replace('-','_').replace(':','_').replace('.','_')
mljar_folder = '/tmp/mljar_'+date_time
mljar_folder

# COMMAND ----------

#
# requested by AutoML
#

X_train_df = pd.DataFrame(X_train.toarray())
X_val_df  = pd.DataFrame(X_val.toarray())
X_test_df  = pd.DataFrame(X_test.toarray())

y_train_df = pd.DataFrame(y_train)
y_val_df  = pd.DataFrame(y_val)
y_test_df  = pd.DataFrame(y_test)

X_train_df = pd.concat([X_train_df, X_val_df])
y_train_df = pd.concat([y_train_df, y_val_df])


# COMMAND ----------

X_train_df.shape

# COMMAND ----------

from supervised.automl import AutoML
import mlflow
mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

vs = {"validation_type" : "split", "train_ratio":.8, "shuffle":False, "stratify": False}




automl = AutoML(results_path=mljar_folder, validation_strategy=vs)
automl.fit(X_train_df, y_train_df)

# COMMAND ----------

y_pred_proba = automl.predict_proba(X_test_df)
y_pred_proba

# COMMAND ----------

import tools.analysis as ta
out = ta.clf_metrics(y_test_df.to_numpy(),y_pred_proba[:,1])
for i in out.columns:
    mlflow.log_metric("Testing "+i, out[i].iloc[0])
out

# COMMAND ----------

mlflow.log_param("average", AVERAGE)
mlflow.log_param("demographics", USE_DEMOG)
mlflow.log_param("outcome", OUTCOME)
mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

mlflow.log_artifacts(mljar_folder)

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


