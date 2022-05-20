# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

# MAGIC %md
# MAGIC ```
# MAGIC dbutils.widgets.removeAll()
# MAGIC dbutils.widgets.text(
# MAGIC   name='experiment_id',
# MAGIC   defaultValue='1910247067387441',
# MAGIC   label='Experiment ID'
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC ```
# MAGIC import mlflow
# MAGIC experiment = dbutils.widgets.get("experiment_id")
# MAGIC assert experiment is not None
# MAGIC current_experiment = mlflow.get_experiment(experiment)
# MAGIC assert current_experiment is not None
# MAGIC experiment_id= current_experiment.experiment_id
# MAGIC ```

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

from databricks import feature_store 
from delta.tables import * 

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- create database
# MAGIC CREATE DATABASE IF NOT EXISTS tnk6_demo

# COMMAND ----------

# MAGIC  %md 
# MAGIC  ```
# MAGIC     parser = argparse.ArgumentParser()
# MAGIC     parser.add_argument('--outcome',
# MAGIC                         type=str,
# MAGIC                         default='multi_class',
# MAGIC                         choices=['misa_pt', 'multi_class', 'death', 'icu'],
# MAGIC                         help='which outcome to use as the prediction target')
# MAGIC     parser.add_argument(
# MAGIC         '--day_one',
# MAGIC         help="Use only first inpatient day's worth of features (DAN only)",
# MAGIC         dest='day_one',
# MAGIC         action='store_true')
# MAGIC     parser.add_argument('--all_days',
# MAGIC                         help="Use all features in lookback period (DAN only)",
# MAGIC                         dest='day_one',
# MAGIC                         action='store_false')
# MAGIC     parser.set_defaults(day_one=True)
# MAGIC     parser.add_argument('--use_demog',
# MAGIC                         type=bool,
# MAGIC                         default=True,
# MAGIC                         help='whether to iclude demographics in the features')
# MAGIC     parser.add_argument('--stratify',
# MAGIC                         type=str,
# MAGIC                         default='all',
# MAGIC                         choices=['all', 'death', 'misa_pt', 'icu'],
# MAGIC                         help='which label to use for the train-test split')
# MAGIC     parser.add_argument('--average',
# MAGIC                         type=str,
# MAGIC                         default='weighted',
# MAGIC                         choices=['micro', 'macro', 'weighted'],
# MAGIC                         help='how to average stats for multiclass predictions')
# MAGIC     parser.add_argument('--cohort_prefix',
# MAGIC                         type=str,
# MAGIC                         default='',
# MAGIC                         help='prefix for the cohort csv file, ending with _')
# MAGIC     parser.add_argument('--out_dir',
# MAGIC                         type=str,
# MAGIC                         help="output directory (optional)")
# MAGIC     parser.add_argument("--data_dir",
# MAGIC                         type=str,
# MAGIC                         help="path to the Premier data (optional)")
# MAGIC     parser.add_argument("--test_split",
# MAGIC                         type=float,
# MAGIC                         default=0.2,
# MAGIC                         help="Percentage of total data to use for testing")
# MAGIC     parser.add_argument("--validation_split",
# MAGIC                         type=float,
# MAGIC                         default=0.1,
# MAGIC                         help="Percentage of train data to use for validation")
# MAGIC     parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
# MAGIC     args = parser.parse_args()
# MAGIC 
# MAGIC     # Setting the globals
# MAGIC     OUTCOME = args.outcome
# MAGIC     USE_DEMOG = args.use_demog
# MAGIC     AVERAGE = args.average
# MAGIC     DAY_ONE_ONLY = args.day_one
# MAGIC     TEST_SPLIT = args.test_split
# MAGIC     VAL_SPLIT = args.validation_split
# MAGIC     RAND = args.rand_seed
# MAGIC     CHRT_PRFX = args.cohort_prefix
# MAGIC     STRATIFY = args.stratify
# MAGIC 
# MAGIC     # Setting the directories and importing the data
# MAGIC     pwd = os.path.abspath(os.path.dirname(__file__))
# MAGIC 
# MAGIC     # If no args are passed to overwrite these values, use repo structure to construct
# MAGIC     data_dir = os.path.abspath(os.path.join(pwd, "..", "data", "data", ""))
# MAGIC     output_dir = os.path.abspath(os.path.join(pwd, "..", "output", ""))
# MAGIC 
# MAGIC     if args.data_dir is not None:
# MAGIC         data_dir = os.path.abspath(args.data_dir)
# MAGIC 
# MAGIC     if args.out_dir is not None:
# MAGIC         output_dir = os.path.abspath(args.out_dir)
# MAGIC 
# MAGIC     pkl_dir = os.path.join(output_dir, "pkl", "")
# MAGIC     stats_dir = os.path.join(output_dir, "analysis", "")
# MAGIC     probs_dir = os.path.join(stats_dir, "probs", "")
# MAGIC ```

# COMMAND ----------

# Setting the globals
OUTCOME = 'icu'
USE_DEMOG = True
AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2022
CHRT_PRFX = ''
STRATIFY ='icu'

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
print(f'Stats Dir: {stats_dir}')

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

tmp_lookup = pd.DataFrame([all_feats],index=['desc']).transpose()
print(tmp_lookup.shape)
tmp_lookup2 = pd.DataFrame([demog_dict],index=['desc']).transpose()
print(tmp_lookup2.shape)
tmp_lookup2
#all_feats
pd.DataFrame([vocab],index=['code']).transpose()

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

train, test = train_test_split(range(n_patients),
                               test_size=TEST_SPLIT,
                               stratify=strat_var,
                               random_state=RAND)

# Doing a validation split for threshold-picking on binary problems
train, val = train_test_split(train,
                              test_size=VAL_SPLIT,
                              stratify=strat_var[train],
                              random_state=RAND)

for var in [train,test,val]:
    print(len(var))

# COMMAND ----------

tmp_df = pd.DataFrame(X[test][0:25].toarray())
tmp_df['target'] = y[test][0:25].astype(int)
tmp_df

# COMMAND ----------

# Taking only 25 as whole test set takes 10 minutes
tmp_df = spark.createDataFrame(tmp_df)

# COMMAND ----------

# MAGIC %md 
# MAGIC # FEATURE STORE

# COMMAND ----------

# create feature store table

fs = feature_store.FeatureStoreClient()
fs.create_table(name = "tnk6_demo.FeatureStore_premier_test",
               primary_keys=list(tmp_df.columns),
               df = tmp_df,
               description='Training set used for Premier Analysis')

fs.write_table(name="tnk6_demo.FeatureStore_premier_test",
              df=tmp_df,
              mode='overwrite')   # or merge to upsert rows

# COMMAND ----------

# MAGIC %md 
# MAGIC # DELTA TABLE

# COMMAND ----------

# Also write delta tables
tmp_df.write.mode("overwrite").format("delta").save("dbfs:/home/tnk6/premier_delta/test")

# COMMAND ----------

# MAGIC %md 
# MAGIC -check from_pandas(pd_df)
# MAGIC -

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- DROP TABLE IF EXISTS tnk6_demo.delta_premier_test;
# MAGIC CREATE TABLE tnk6_demo.delta_premier_test USING DELTA LOCATION "dbfs:/home/tnk6/premier_delta/test";

# COMMAND ----------

spark.conf.get("spark.databricks.delta.lastCommitVersionInSession")

# COMMAND ----------

# MAGIC %md 
# MAGIC ```
# MAGIC test_delta_table = spark.read.format("delta").load('/home/tnk6/premier_delta/test')
# MAGIC display(test_delta_table)
# MAGIC # 16.31 mins - very slow
# MAGIC ```

# COMMAND ----------

# MAGIC %md 
# MAGIC # Using delta table
# MAGIC - Note: TODO: compare delta table with current version and only merge if there are changes (?)

# COMMAND ----------

test_table_name = "tnk6_demo.delta_premier_test"
test_delta_table = DeltaTable.forName(spark, test_table_name)
test_delta_table

# COMMAND ----------

##deltTable.restoreToVersions(0) # oldest version
display(test_delta_table.history())

# COMMAND ----------

tmp_val = test_delta_table.history().select("version").collect()
print(tmp_val[-1].__getitem__('version'))

# COMMAND ----------

# Fitting a logistic regression to the whole dataset
lgr = LogisticRegression(max_iter=5000, multi_class='ovr')
mlflow.sklearn.autolog(log_models=True)

experiment_name = "/Shared/premier_experiment_test"
mlflow.set_experiment(experiment_name) # this creates a workspace experiment if it does not exist 

#with mlflow.start_run(experiment_id=experiment_id) as run:

with mlflow.start_run() as run:
    lgr.fit(X, y)
    mlflow.log_param("Test Dataset",test_table_name)
    mlflow.log_param("Test Dataset Version",tmp_val[-1].__getitem__('version'))
    mlflow.sklearn.log_model(lgr, "lgr")
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

#with pd.ExcelWriter(stats_dir + out_name + 'coefs.xlsx') as writer:
#    for i, df in enumerate(coef_list):
#        df.to_excel(writer, sheet_name='coef_' + str(i), index=False)

#    writer.save()

# COMMAND ----------

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
    with mlflow.start_run() as run:
#    with mlflow.start_run(experiment_id=experiment_id) as run:
        mod.fit(X[train], y[train])
        mlflow.sklearn.log_model(mod, mod_names[i])
        mlflow.log_param("Test Dataset",test_table_name)
        mlflow.log_param("Test Dataset Version",tmp_val[-1].__getitem__('version'))
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
                                   average=AVERAGE)
            ta.write_preds(output_dir=output_dir + "/",
                           preds=test_preds,
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
                                   average=AVERAGE)
            ta.write_preds(output_dir=output_dir + "/",
                           preds=test_preds,
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
                               average=AVERAGE)
        ta.write_preds(output_dir=output_dir + "/",
                       preds=test_preds,
                       outcome=OUTCOME,
                       mod_name=mod_name,
                       test_idx=test)

    # Saving the results to disk
    ta.write_stats(stats, OUTCOME, stats_dir=stats_dir)

# COMMAND ----------

stats_dir

# COMMAND ----------


