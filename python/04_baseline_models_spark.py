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

if EXPERIMENTING == True: SAMPLE = 10
else: SAMPLE = X.shape[1]

# COMMAND ----------

def convert_pandas_to_spark_with_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler


    inc=20000
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):
        if isinstance (a_dataframe,  pd.DataFrame):
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
        else:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].toarray())

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


def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

c_names = change_columns_names(X)

X_train_spark = convert_pandas_to_spark_with_vectors(X[train][:SAMPLE], c_names).select(['features']).withColumn("id",monotonically_increasing_id() )
X_val_spark =   convert_pandas_to_spark_with_vectors(X[val][:SAMPLE], c_names).select(['features']).withColumn("id",monotonically_increasing_id() )
X_test_spark =  convert_pandas_to_spark_with_vectors(X[test][:SAMPLE], c_names).select(['features']).withColumn("id",monotonically_increasing_id() )


y_train_spark = spark.createDataFrame(pd.DataFrame(y[train][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )
y_val_spark =   spark.createDataFrame(pd.DataFrame(y[val][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )
y_test_spark =  spark.createDataFrame(pd.DataFrame(y[test][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )


# COMMAND ----------

X_y_train_spark = X_train_spark.join(y_train_spark,X_train_spark.id == y_train_spark.id, how='inner')
X_y_val_spark = X_val_spark.join(y_val_spark,X_val_spark.id == y_val_spark.id, how='inner')
X_y_test_spark = X_test_spark.join(y_test_spark,X_test_spark.id == y_test_spark.id, how='inner')

X_y_spark = X_y_train_spark.union(X_y_val_spark).union(X_y_test_spark)
X_y_train_val_spark = X_y_train_spark.union(X_y_val_spark)

# COMMAND ----------

def get_array_of_probs (spark_df):
    from pyspark.ml.functions import vector_to_array

    probsp = spark_df.select(vector_to_array("probability", "float32").alias("probability")).toPandas()
    probss= probsp['probability']
    probsn = probss.to_numpy()
    prob_list = list()
    for prob in probsn: 
        prob_list = prob_list + [prob[1]]
    prob_array = np.array(prob_list)
    return prob_array

# COMMAND ----------

def get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name, average=AVERAGE):
    val_gm = ta.grid_metrics(y_val, val_probs)
    cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
    test_preds = ta.threshold(test_probs, cutpoint)
    stats = ta.clf_metrics(y_test,
                           test_probs,
                           cutpoint=cutpoint,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

def get_statistics_from_predict(test_predict, y_test, mod_name, average=AVERAGE):
    stats = ta.clf_metrics(y_test,
                           test_predict,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

def log_stats_in_mlflow(stats):
    for i in stats:
        if not isinstance(stats[i].iloc[0], str):
            mlflow.log_metric("Testing "+i, stats[i].iloc[0])

# COMMAND ----------

def log_param_in_mlflow():
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

y_val = y_val_spark.select('y').toPandas()['y'].to_numpy()
y_test = y_test_spark.select('y').toPandas()['y'].to_numpy()

# COMMAND ----------

def executeLogisticRegression():
    #import the logistic regression 
    from pyspark.ml.classification import LogisticRegression
    #
    # start a new MLFlow Experiment
    #
    import mlflow
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.autolog()
    #
    #Apply the logistic regression model
    #
    model=LogisticRegression(labelCol='y')
    model_fit = model.fit(X_y_train_spark.select(['features','y']))
    predictions_val = model_fit.transform(X_y_val_spark.select(['features','y']))
    predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']))
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name='LogisticRegression', average=AVERAGE)
    #
    # log statistics and parameters into ml flow and end experiment
    #
    log_stats_in_mlflow(stats)
    log_param_in_mlflow()

    mlflow.end_run()
    return stats

# COMMAND ----------

def executeRandomForestClassifier():
    #import the Random Forrest  
    from pyspark.ml.classification import RandomForestClassifier
    #
    # start a new MLFlow Experiment
    #
    import mlflow
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.autolog()

    model=RandomForestClassifier(featuresCol='features',labelCol='y')
    model_fit = model.fit(X_y_train_spark.select(['features','y']))
    predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']))

    stats = get_statistics_from_predict(predictions_test.select('prediction').toPandas()['prediction'].to_numpy(), 
                                        y_test, 
                                        "RandomForestClassifier", 
                                        average=AVERAGE)
    #
    # log statistics and parameters into ml flow and end experiment
    #
    log_stats_in_mlflow(stats)
    log_param_in_mlflow()

    mlflow.end_run()
    return stats

# COMMAND ----------

def executeGBTClassifier():
    from pyspark.ml.classification import GBTClassifier
    #
    # start a new MLFlow Experiment
    #
    import mlflow
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.spark.autolog()
    #
    #
    model=GBTClassifier(featuresCol='features',labelCol='y')
    model_fit = model.fit(X_y_train_spark.select(['features','y']))
    predictions_val = model_fit.transform(X_y_val_spark.select(['features','y']))
    predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']))
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name='GBTClassifier', average=AVERAGE)
    #
    # log statistics and parameters into ml flow and end experiment
    #
    log_stats_in_mlflow(stats)
    log_param_in_mlflow()

    mlflow.end_run()
    return stats

# COMMAND ----------

def executeDecisionTreeClassifier():
    #import the Random Forrest  
    from pyspark.ml.classification import DecisionTreeClassifier
    #
    # start a new MLFlow Experiment
    #
    import mlflow
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.spark.autolog()

    model=DecisionTreeClassifier(featuresCol='features',labelCol='y')
    model_fit = model.fit(X_y_train_spark.select(['features','y']))
    predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']))

    stats = get_statistics_from_predict(predictions_test.select('prediction').toPandas()['prediction'].to_numpy(), 
                                        y_test, 
                                        "DecisionTreeClassifier", 
                                        average=AVERAGE)
    #
    # log statistics and parameters into ml flow and end experiment
    #
    log_stats_in_mlflow(stats)
    log_param_in_mlflow()

    mlflow.end_run()
    return stats

# COMMAND ----------

def executeLinearSVC():
    from pyspark.ml.classification import LinearSVC
    #
    # start a new MLFlow Experiment
    #
    import mlflow
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.spark.autolog()
    #
    #
    model=LinearSVC(featuresCol='features',labelCol='y')
    model_fit = model.fit(X_y_train_spark.select(['features','y']))
    predictions_val = model_fit.transform(X_y_val_spark.select(['features','y']))
    predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']))
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name='LinearSVC', average=AVERAGE)
    #
    # log statistics and parameters into ml flow and end experiment
    #
    log_stats_in_mlflow(stats)
    log_param_in_mlflow()

    mlflow.end_run()
    return stats

# COMMAND ----------

stats = executeLogisticRegression()
stats

# COMMAND ----------

stats = executeRandomForestClassifier()
stats

# COMMAND ----------

stats = executeGBTClassifier()
stats

# COMMAND ----------

stats = executeDecisionTreeClassifier()
stats

# COMMAND ----------

stas = executeLinearSVC()
stats

# COMMAND ----------

row_list =X_y_train_spark.take(100)



# COMMAND ----------

model=LogisticRegression(labelCol='y')
model_fit = model.fit(X_y_train_spark.select(['features','y']).take(100)
predictions_val = model_fit.transform(X_y_val_spark.select(['features','y']).take(100))
predictions_test = model_fit.transform(X_y_test_spark.select(['features','y']).take(100))
val_probs  = get_array_of_probs (predictions_val)
test_probs = get_array_of_probs (predictions_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name='LogisticRegression', average=AVERAGE)
