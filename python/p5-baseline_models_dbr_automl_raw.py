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

train_set_pd = pd.DataFrame(X[train_set].toarray())
#train_set_pd['y'] = y[train_set].astype(int)
test_set_pd = pd.DataFrame(X[test].toarray())
#test_set_pd['y'] =  y[test].astype(int)

# COMMAND ----------

c_names = list()
for i in range(0, X.shape[1]):
    c_names = c_names + ['c'+str(i)] 
type(c_names)

# COMMAND ----------

def convert_pandas_to_spark(a_dataframe, c_names):
    from pyspark.sql import SparkSession

    inc=10000
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):
        #X_rdd = spark.sparkContext.parallelize(X[i*inc:(1+i)*inc].toarray())
        rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
        df_tmp = (rdd.map(lambda x: x.tolist()).toDF(c_names)  )
        if bool == True:
            spark_df = df_tmp
            bool = False
        else:
            spark_df = spark_df.union(df_tmp)
    return spark_df

# COMMAND ----------

def create_array_of_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(c_names)
    spark_vector = vecAssembler.transform(a_dataframe)

    return spark_vector

# COMMAND ----------

def convert_pandas_to_spark_with_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler


    inc=20000
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):
        #X_rdd = spark.sparkContext.parallelize(X[i*inc:(1+i)*inc].toarray())
        a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
        a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names)  )
        
        vecAssembler = VectorAssembler(outputCol="features")
        vecAssembler.setInputCols(c_names)
        a_spark_vector = vecAssembler.transform(a_df)
        
        if bool == True:
            spark_df = a_spark_vector
            bool = False
        else:
            spark_df = spark_df.union(a_spark_vector)
    return spark_df

# COMMAND ----------

#X_test_spark = convert_pandas_to_spark(test_set_pd, cnames)


# COMMAND ----------


#X_train_spark = convert_pandas_to_spark(train_set_pd, c_names)
#X_train_spark_vector = create_array_of_vectors(X_train_spark, c_names)

# COMMAND ----------

X_train_spark_vector = convert_pandas_to_spark_with_vectors(train_set_pd, c_names)

# COMMAND ----------

y_train_pd = pd.DataFrame(y[train_set],columns=['y']).astype(int)
y_train_spark = spark.createDataFrame(y_train_pd)
#y_test_pd = pd.DataFrame(y[test],columns=['y']).astype(int)
#y_test_spark = spark.createDataFrame(y_test_pd)


# COMMAND ----------



# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

X_train_spark_vector = X_train_spark_vector.withColumn("id",monotonically_increasing_id() )
y_train_spark = y_train_spark.withColumn( "id", monotonically_increasing_id() )
X_y_train_spark = X_train_spark_vector.join(y_train_spark,X_train_spark_vector.id == y_train_spark.id, how='inner')


# COMMAND ----------

#train_df, test_df = X_y_df.randomSplit([0.80, 0.20], seed=RAND)
from pyspark.ml.functions import vector_to_array
to_automl = X_y_train_spark.select(vector_to_array("features").alias("features"),"y")
#to_automl = X_y_train_spark.select(*c_names,"y")
#to_automl = X_train_spark

# COMMAND ----------

display(to_automl)

# COMMAND ----------

to_automl.printSchema()

# COMMAND ----------



to_automl = to_automl.withColumn("y", to_automl["y"].cast('int'))
to_automl.printSchema()


# COMMAND ----------

from databricks import automl

summary = automl.classify(to_automl, target_col="y", timeout_minutes=600)

# COMMAND ----------

help(summary)

# COMMAND ----------

model_uri = summary.best_trial.model_path

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

y_test_pd = pd.DataFrame(y[test],columns=['y']).astype(int)
y_test_spark = spark.createDataFrame(y_test_pd)
y_test_spark = y_test_spark.withColumn( "id", monotonically_increasing_id() )


test_set_pd = pd.DataFrame(X[test].toarray())
X_test_spark_vector = convert_pandas_to_spark_with_vectors(test_set_pd, c_names)
X_test_spark_vector = X_test_spark_vector.withColumn("id",monotonically_increasing_id() )

X_y_test_spark = X_test_spark_vector.join(y_test_spark,X_test_spark_vector.id == y_test_spark.id, how='inner')
to_automl_test = X_y_test_spark.select(vector_to_array("features").alias("features"),"y")


# COMMAND ----------

test_pdf =  to_automl_test.toPandas()
y_test = test_pdf["y"]
X_test = test_pdf.drop("y", axis=1)
predictions = model.predict(X_test)


# COMMAND ----------

from sklearn.metrics import f1_score
f1 = f1_score(y_test, predictions)
print("f1: "+str(f1))

# COMMAND ----------

display(test_pdf)

# COMMAND ----------

import sklearn.metrics

model = mlflow.sklearn.load_model(model_uri)
sklearn.metrics.plot_confusion_matrtix(model,X_test, y_test)

# COMMAND ----------


