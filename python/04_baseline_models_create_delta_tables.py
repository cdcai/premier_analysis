# Databricks notebook source
dbutils.widgets.removeAll()

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

dbutils.widgets.text(
  name='database',
  defaultValue='too9_premier_analysis_demo',
  label='Database'
)
DATABASE=dbutils.widgets.get("database")

dbutils.widgets.text(
  name='train_dt',
  defaultValue='train_data_set',
  label='Trainning Table'
)
TRAINNING_DT=dbutils.widgets.get("train_dt")


dbutils.widgets.text(
  name='val_dt',
  defaultValue='val_data_set',
  label='Validation Table'
)
VALIDATION_DT=dbutils.widgets.get("val_dt")

dbutils.widgets.text(
  name='test_dt',
  defaultValue='test_data_set',
  label='Testing Table'
)
TESTING_DT=dbutils.widgets.get("test_dt")


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
LABEL_COLUMN='label'
FEATURES_COLUMN='features'

# COMMAND ----------


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

#
# used for limiting the sample size for testing
#
if EXPERIMENTING == True: 
    ROWS = 1000
    COLS = 100
else:
    ROWS = X.shape[0]
    COLS = X.shape[1]

# COMMAND ----------

#
# when converting sparce/dense matrices to Spark Data Frames, column names are required.#
#
def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

#
# If there is enough memory available in the master node,
# this functions proved to be faster
#
def convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0
    
    number_of_partitions = int(spark.sparkContext.defaultParallelism)*2

    a_rdd = spark.sparkContext.parallelize(a_dataframe.to_numpy(), number_of_partitions)
    
    a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names+[LABEL_COLUMN]) )

    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(c_names)
    spark_df = vecAssembler.transform(a_df)

    return spark_df

# COMMAND ----------

#
# If there is NOT enough memory available in the master node,
# this functions proved to be useful
#
def incrementaly_convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
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
            a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names+LABEL_COLUMN) )

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

#
# working around to transform pandas DF to spark DF
#
def pandas_to_spark_via_parquet_files(pDF, c_names, results, index): 
    from pyspark.ml.feature import VectorAssembler
    import time
    
    seconds = time.time()
    
    fileName = "/FileStore/tmp/file"+str(seconds)+".parquet"

    pDF.to_parquet("/dbfs/"+fileName, compression="gzip")  
    sDF=spark.read.parquet(fileName)
    results[index] = VectorAssembler(outputCol=FEATURES_COLUMN)\
                    .setInputCols(c_names)\
                    .transform(sDF).select(LABEL_COLUMN, FEATURES_COLUMN).cache()
    
def convert_pDF_to_sDF_via_parquet_files(list_of_pandas, c_names):
    from threading import Thread

    results = [None] * len(list_of_pandas)
    threads = [None] * len(list_of_pandas)

    for index in range(0,len(threads)):
            threads [index] = Thread(target=pandas_to_spark_via_parquet_files, 
                                     args=(list_of_pandas[index], 
                                           c_names, 
                                           results, 
                                           index))
            threads[index].start()

    for i in range(len(threads)):
        threads[i].join()

    return results


# COMMAND ----------

#
# Since the converstion from Pandas to Spark have been extremelly slow,
# this function allows the converstion to be done in parallel
#
def pandas_to_spark(pDF, c_names, results, index): 
    results[index] = convert_pandas_to_spark_with_vectors(pDF, c_names).select([LABEL_COLUMN,FEATURES_COLUMN]).cache()
        
def convert_pDF_to_sDF(list_of_pandas, c_names):
    from threading import Thread

    results = [None] * len(list_of_pandas)
    threads = [None] * len(list_of_pandas)

    for index in range(0,len(threads)):
            threads [index] = Thread(target=pandas_to_spark, 
                                     args=(list_of_pandas[index], 
                                           c_names, 
                                           results, 
                                           index))
            threads[index].start()

    for i in range(len(threads)):
        threads[i].join()

    return results

# COMMAND ----------

#
# create pandas frames from X
#
c_names = change_columns_names(X)[:COLS]

X_train_pandas = pd.DataFrame(X[train][:ROWS,:COLS].toarray(),columns=c_names)
X_train_pandas[LABEL_COLUMN] = y[train][:ROWS].astype("int")

X_val_pandas = pd.DataFrame(X[val][:ROWS,:COLS].toarray(),columns=c_names)
X_val_pandas[LABEL_COLUMN] = y[val][:ROWS].astype("int")

X_test_pandas = pd.DataFrame(X[test][:ROWS,:COLS].toarray(),columns=c_names)
X_test_pandas[LABEL_COLUMN] = y[test][:ROWS].astype("int")

# COMMAND ----------

results = None
list_of_pandas = [X_train_pandas,X_val_pandas,X_test_pandas]
results = convert_pDF_to_sDF(list_of_pandas,c_names)

X_train_spark = results[0]
X_val_spark   = results[1]
X_test_spark  = results[2]

# COMMAND ----------

#
# .count() forces the cache
#
for i in range(len(results)): 
    print(results[i].count())
    print(results[i].rdd.getNumPartitions())

# COMMAND ----------

sql = [f'CREATE DATABASE IF NOT EXISTS {DATABASE};']

# COMMAND ----------

for sql_st in sql: spark.sql(sql_st)

# COMMAND ----------

#
# save Spark Data Frames to Delta Tables
# it seems that Spark Frames created from delta tables
# work faster
#
X_train_spark.write.mode("overwrite").format("delta").saveAsTable(f'{DATABASE}.{TRAINNING_DT}')
X_train_spark.write.mode("overwrite").format("delta").saveAsTable(f'{DATABASE}.{VALIDATION_DT}')
X_train_spark.write.mode("overwrite").format("delta").saveAsTable(f'{DATABASE}.{TESTING_DT}')

