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

def automl_data_prep_pipeline(X_train, y_train,  target):
    estimated_n_pca = calculate_estimated_n_pca_componetns(X_train[:1000])
    pcas = calculatePCA(X_train,n_components=estimated_n_pca, batch_size=estimated_n_pca)
    df = prepareDfToAutoML(X_train, y_train, target)

# COMMAND ----------

#estimated_n_pca = calculate_estimated_n_pca_componetns(X[train][:10000])
#
# result was 1574
#

# COMMAND ----------

#estimated_n_pca

# COMMAND ----------

#pcas = calculatePCA(X,n_components=estimated_n_pca, batch_size=estimated_n_pca)
pcas = calculatePCA(X,1574, batch_size=1574)

# COMMAND ----------

automl_df = prepareDfToAutoML(pcas, y, 'target')


# COMMAND ----------

import time, datetime
 
time.sleep(1)
automl_df['time_column']=None
automl_df['time_column'][train]=datetime.datetime.now()
time.sleep(1)
automl_df['time_column'][val]=datetime.datetime.now()
time.sleep(1)
automl_df['time_column'][test]=datetime.datetime.now()


# COMMAND ----------

automl_df[(automl_df["time_column"] == None)]

# COMMAND ----------

from databricks import automl
summary = automl.classify(automl_df, target_col='target',  time_col='time_column',timeout_minutes=600)

# COMMAND ----------

#
#
#spark_df_train_set = spark.createDataFrame(df_train_set)
#from pyspark.ml.feature import VectorAssembler
#vc = VectorAssembler(inputCols=df_train_set.columns.tolist(), outputCol="features")
#stream_df = vc.transform(spark_df_train_set)
##
# spark PCA could not handle the size of the input vector
#
#from pyspark.ml.feature import PCA
#from pyspark.ml.linalg import Vectors
#for i in range(2,1000):
#    print (i)
#    pca = PCA(k=i, inputCol="features")
#    model = pca.fit(stream_df)
#    transformed = model.transform(stream_df)
#    explainedVariance = model.explainedVariance
#    if explainedVariance.toArray().sum() > .75:
#        print (i)

# COMMAND ----------

import mlflow

model_uri = summary.best_trial.model_path


# COMMAND ----------

# Prepare test dataset
test_pdf = prepareDfToAutoML(pcas[test],y[test], 'target')
 
y_test = test_pdf.pop('target')
time_column = test_pdf.pop('time_column')
X_test = test_pdf

# COMMAND ----------

X_test.shape

# COMMAND ----------

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
model

# COMMAND ----------

predictions = model.predict(X_test)
predictions

# COMMAND ----------

test_pdf["target_predicted"] = predictions
display(test_pdf)

# COMMAND ----------

import sklearn.metrics

model = mlflow.sklearn.load_model(model_uri)
sklearn.metrics.plot_confusion_matrix(model, X_test, y_test)

# COMMAND ----------

stats = ta.clf_metrics(y_test,
                       predictions,
                       mod_name=summary.best_trial.model_description.split("(")[0],
                       average=AVERAGE)
stats


# COMMAND ----------

summary.best_trial.mlflow_run_id

# COMMAND ----------


