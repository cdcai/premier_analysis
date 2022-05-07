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
OUTCOME = 'misa_pt'
USE_DEMOG = True
AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2022
CHRT_PRFX = ''
STRATIFY ='all'

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

stats_dir

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

X[train_set][:100].shape

# COMMAND ----------

X[val].shape

# COMMAND ----------

df_X = pd.DataFrame(X[train_set][:10].toarray())
#df_X = df_X.rename(columns=lambda x: "c"+str(X))

# COMMAND ----------

df_X['target']=y[train_set][:10]
df_X['target'] = df_X['target'].astype(int)

# COMMAND ----------

spark_x_train= spark.createDataFrame(df_X)

# COMMAND ----------

from databricks import automl
summary = automl.classify(spark_x_train, target_col="target", timeout_minutes=120)


# COMMAND ----------

help(summary)

# COMMAND ----------

model_uri = summary.best_trial.model_path

# COMMAND ----------

# MAGIC %md
# MAGIC Inference
# MAGIC 
# MAGIC You can use the model trained by AutoML to make predictions on new data. 
# MAGIC 
# MAGIC The examples below demonstrate how to make predictions on data in pandas DataFrames, or register the model as a Spark UDF for prediction on Spark DataFrames.

# COMMAND ----------

import mlflow
 
# Prepare test dataset
test_pdf = pd.DataFrame(X[test][:10].toarray())
test_pdf['target']=y[test][:10]
test_pdf['target'] = test_pdf['target'].astype(int)

y_test = test_pdf["target"]
X_test = test_pdf.drop("target", axis=1)
 
# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
predictions = model.predict(X_test)
test_pdf["target_predicted"] = predictions
display(test_pdf)

# COMMAND ----------

#
# Spark
#
predict_udf = mlflow.pyfunc.spark_udf(spark, model_uri=model_uri, result_type="integer")
display(test_pdf.withColumn("target_predicted_spark", predict_udf()))

# COMMAND ----------

import sklearn.metrics
 
model = mlflow.sklearn.load_model(model_uri)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------


