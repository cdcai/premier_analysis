# Databricks notebook source
!pip install mlflow --quiet

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

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='1910247067387441',
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


dbutils.widgets.dropdown("experimenting", "True",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id

print("OUTCOME ", OUTCOME)
print("STRATIFY ", STRATIFY)

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

#df_train_set = pd.DataFrame(X[train_set][:20000].toarray())
#df_train_set = df_train_set.rename(columns=lambda x: "c"+str(x))

# COMMAND ----------

def calculatePCA(X_train_set,n_components, batch_size):
    from sklearn.decomposition import IncrementalPCA
    transformer = IncrementalPCA(n_components=n_components, batch_size=batch_size)
    x_transformed = transformer.fit_transform(X_train_set)
    return x_transformed

# COMMAND ----------

def prepareDfToAutoML (X,y,target):
    df = pd.DataFrame(X)
    df = df.rename(columns=lambda x: "c"+str(x))
    df[target] = y
    df[target] = df[target].astype('bool')
    return df

# COMMAND ----------

def calculate_estimated_n_pca_componetns(X_train_set):
    from sklearn.decomposition import PCA
    
    df_train_set = pd.DataFrame(X_train_set.toarray())
    pca = PCA (n_components=.95)
    components = pca.fit_transform(df_train_set)
    return pca.n_components_


# COMMAND ----------

#estimated_n_pca = calculate_estimated_n_pca_componetns(X[train][:10000])
#
# result was 1574
#

# COMMAND ----------

#estimated_n_pca

# COMMAND ----------

#pcas = calculatePCA(X,n_components=estimated_n_pca, batch_size=estimated_n_pca)
#pcas = calculatePCA(X,1574, batch_size=1574)
#pcas = calculatePCA(X[:100],100, batch_size=100)

# COMMAND ----------

suffix = "_outcome_"+OUTCOME+"_stratify_"+STRATIFY
if EXPERIMENTING == True:
    pcas_train = calculatePCA(X[train][:100],100, batch_size=100)
    pcas_val   = calculatePCA(X[val][:100],100, batch_size=100)
    pcas_test  = calculatePCA(X[test][:100],100, batch_size=100)

    df_train = prepareDfToAutoML(pcas_train, y[train][:100], 'target')
    df_val   = prepareDfToAutoML(pcas_val, y[val][:100], 'target')
    df_test  = prepareDfToAutoML(pcas_test, y[test][:100], 'target')
else:
    pcas_train = calculatePCA(X[train],1574, batch_size=1574)
    pcas_val   = calculatePCA(X[val],1574, batch_size=1574)
    pcas_test  = calculatePCA(X[test],1574, batch_size=1574)

    df_train = prepareDfToAutoML(pcas_train, y[train], 'target')
    df_val   = prepareDfToAutoML(pcas_val, y[val], 'target')
    df_test  = prepareDfToAutoML(pcas_test, y[test], 'target')



# COMMAND ----------

suffix = "_outcome_"+OUTCOME+"_stratify_"+STRATIFY
if EXPERIMENTING == True:
    df_train.to_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas_only_100'+suffix+'.csv')
    df_val.to_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas_only_100'+suffix+'.csv')
    df_test.to_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas_only_100'+suffix+'.csv')
else:
    df_train.to_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas'+suffix+'.csv')
    df_val.to_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas'+suffix+'.csv')
    df_test.to_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas'+suffix+'.csv')


# COMMAND ----------

EXPERIMENTING

# COMMAND ----------

pcas_train.shape

# COMMAND ----------


