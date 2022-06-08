# Databricks notebook source
!pip install mlflow --quiet

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

dbutils.widgets.dropdown("experimenting", "False",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

# COMMAND ----------

import mlflow
experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id


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

# Setting the globals
#OUTCOME = 'misa_pt'
#USE_DEMOG = True
AVERAGE = 'weighted'
DAY_ONE_ONLY = True
TEST_SPLIT = 0.2
VAL_SPLIT = 0.1
RAND = 2022
CHRT_PRFX = ''
#STRATIFY ='all'

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

train, test = train_test_split(range(n_patients),
                               test_size=TEST_SPLIT,
                               stratify=strat_var,
                               random_state=RAND)

# Doing a validation split for threshold-picking on binary problems
train, val = train_test_split(train,
                              test_size=VAL_SPLIT,
                              stratify=strat_var[train],
                              random_state=RAND)

# COMMAND ----------

if EXPERIMENTING == True: SAMPLE = 100
else: SAMPLE = X.shape[0]

# COMMAND ----------

X.shape

# COMMAND ----------

def convert_pandas_to_spark_with_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0

    inc=min(10000, a_dataframe.shape[0])
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):

        
        if (i*inc) < a_dataframe.shape[0]:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
            a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names)  )

            #a_df = spark.createDataFrame(a_rdd, c_names)

            vecAssembler = VectorAssembler(outputCol="features")
            vecAssembler.setInputCols(c_names)
            a_spark_vector = vecAssembler.transform(a_df)

            if bool == True:
                spark_df = a_spark_vector
                bool = False
            else:
                spark_df = spark_df.union(a_spark_vector)
    
    old_col_name = "_"+str(a_dataframe.shape[1]-1) # vector assembler would change the name of collumn y
    print(old_col_name)
    spark_df = spark_df.withColumnRenamed (old_col_name,'y')
    time_col_name = "_"+str(a_dataframe.shape[1]) # vector assembler would change the name of collumn y
    print(time_col_name)
    spark_df = spark_df.withColumnRenamed (old_col_name,'time-col')
    
    return spark_df


def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

from datetime import datetime
c_names = change_columns_names(X)

#
# add y and time stamp to pandas 
#

X_train_pandas = pd.DataFrame(X[train][:SAMPLE].toarray())
X_train_pandas['y'] = y[train][:SAMPLE]
X_train_pandas['time-col'] = datetime.today().timestamp()
print( datetime.today().timestamp())

X_val_pandas = pd.DataFrame(X[val][:SAMPLE].toarray())
X_val_pandas['y'] = y[val][:SAMPLE]
X_val_pandas['time-col'] =  datetime.today().timestamp()
print( datetime.today().timestamp())


X_test_pandas = pd.DataFrame(X[test][:SAMPLE].toarray())
X_test_pandas['y'] = y[test][:SAMPLE]
X_test_pandas['time-col'] =  datetime.today().timestamp()
print( datetime.today().timestamp())

#
# create concated pandas with train, val, and test
#

X_pandas = pd.concat([X_train_pandas,X_val_pandas,X_test_pandas])

#
# create incrementally spark data frames
#

#X_train_spark = convert_pandas_to_spark_with_vectors(X_train_pandas, c_names)
#X_val_spark =   convert_pandas_to_spark_with_vectors(X_val_pandas, c_names)
#X_test_spark =  convert_pandas_to_spark_with_vectors(X_test_pandas, c_names)
X_spark =  convert_pandas_to_spark_with_vectors(X_pandas, c_names)



# COMMAND ----------

display(X_spark.select(['features', 'y','time-col']))

# COMMAND ----------

X_spark = X_spark.withColumn("y",col("y").cast(IntegerType))

# COMMAND ----------

from pyspark.ml.functions import vector_to_array
to_automl = X_spark.select(vector_to_array("features").alias("features"),"y")


# COMMAND ----------

from databricks import automl
summary = automl.classify(to_automl, time_col="time-col",primary_metric="roc_auc",target_col="y", timeout_minutes=1200)

# COMMAND ----------

summary = automl.classify(to_automl, time_col="time-col",primary_metric="log_loss",target_col="y", timeout_minutes=1200)

# COMMAND ----------

summary = automl.classify(to_automl, time_col="time-col",primary_metric="f1",target_col="y", timeout_minutes=1200)

# COMMAND ----------


