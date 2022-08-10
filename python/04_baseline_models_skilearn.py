# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='32908833595366',
  label='Experiment ID'
)


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
LABEL_COLUMN='label'
FEATURES_COLUMN='features'

# COMMAND ----------

#
# working around to import data frames from delta tables
# ML algorithms work significally faster
#
from delta.tables import DeltaTable

X_train_dt = spark.table(f"{DATABASE}.{TRAINNING_DT}")
X_val_dt = spark.table(f"{DATABASE}.{VALIDATION_DT}")
X_test_dt = spark.table(f"{DATABASE}.{TESTING_DT}")


# COMMAND ----------

#
# Spark ML return predictions as vector of probabilities
# this function return the positive probabilities
#
def get_array_of_probs (predictions_sDF):
    from pyspark.ml.functions import vector_to_array
    import numpy as np

    p = predictions_sDF.select(vector_to_array("probability", "float32").alias("probability")).toPandas()['probability'].to_numpy()
    
    return np.array(list(map(lambda x: x[1], p)))

# COMMAND ----------

#
# this function calculates the statistics from validation and testing predictions
#
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

#
# some Spark ML algorithms do not return probabilities
# this function calculates statistics from predictions without probabilities
#
def get_statistics_from_predict(test_predict, y_test, mod_name, average=AVERAGE):
    stats = ta.clf_metrics(y_test,
                           test_predict,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

#
# this function logs statistics on MLFLow
#
def log_stats_in_mlflow(stats):
    for i in stats:
        if not isinstance(stats[i].iloc[0], str):
            mlflow.log_metric("testing_"+i, stats[i].iloc[0])

# COMMAND ----------

#
# this function logs a few MLFlow parameters about the type of prediction
#
def log_param_in_mlflow():
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

#
# H2O returns predicit probabilities in a different way of Spark ML
# this functions returns the probabilities
#
def get_array_of_probabilities_from_sparkling_water_prediction(predict_sDF):
    p = predict_sDF.select('detailed_prediction').collect()
    probs = list()
    for row in range(len(p)):
        prob = p[row].asDict()['detailed_prediction']['probabilities'][1]
        probs = probs + [prob]
    
    return np.asarray(probs)

# COMMAND ----------

### to be used only if the input are spark dataframes
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array as v2a

y_train = X_train_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()
X_train = X_train_dt.select(v2a(FEATURES_COLUMN).alias(FEATURES_COLUMN)).toPandas()[FEATURES_COLUMN].to_numpy()

X_val = X_val_dt.select(v2a(FEATURES_COLUMN).alias(FEATURES_COLUMN)).toPandas()[FEATURES_COLUMN].to_numpy()
y_val = X_val_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()

X_test = X_test_dt.select(v2a(FEATURES_COLUMN).alias(FEATURES_COLUMN)).toPandas()[FEATURES_COLUMN].to_numpy()
y_test = X_test_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()


# COMMAND ----------

X_train[0]

# COMMAND ----------

#
# this part user ski-learn 
#

from scipy.sparse import lil_matrix
from sklearn.ensemble import GradientBoostingClassifier as sk_gbc
from sklearn.ensemble import RandomForestClassifier as sk_rfc
from sklearn.linear_model import LogisticRegression as sk_lr
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC as sk_svc

# Loading up some models to try
stats = None
mods = [
    sk_lr(max_iter=5000, multi_class='ovr'),
    sk_rfc(n_estimators=500, n_jobs=-1),
    sk_gbc(),
    sk_svc(class_weight='balanced', max_iter=10000)
]
mod_names = ['lgr', 'rf', 'gbc', 'svm']

# Turning the crank like a proper data scientist
for i, mod in enumerate(mods):
    #
    # add execution parameters to MLFLOW
    #
    mlflow.end_run()
    modelName = mod_names[i]
    run_name=f"sci-learn_{modelName}"

    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    mlflow.log_param("training delta table", f"{DATABASE}.{TRAINNING_DT}")
    mlflow.log_param("validation delta table", f"{DATABASE}.{VALIDATION_DT}")
    mlflow.log_param("testing delta table", f"{DATABASE}.{TESTING_DT}")
    #
    #
    #
    model_fit = mod.fit(x_train, y_train)
    
    mlflow.sklearn.log_model(model_fit, "model")
    # to make sure model can be found progrmatically, 
    # use "model" as the name of the model
    
    
    mod_name = mod_names[i]
    if DAY_ONE_ONLY:
        mod_name += '_d1'

    if 'predict_proba' in dir(mod):
        if binary:
            val_probs = model_fit.predict_proba(X_val)[:, 1]
            val_gm = ta.grid_metrics(y_val, val_probs)
            cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
            test_probs = model_fit.predict_proba(X_test)[:, 1]
            test_preds = ta.threshold(test_probs, cutpoint)
            stats = ta.clf_metrics(y_test,
                                   test_probs,
                                   cutpoint=cutpoint,
                                   mod_name=mod_name,
                                   average=AVERAGE)
        else:
            cutpoint = None
            test_probs = model_fit.predict_proba(X_test)
            test_preds = model_fit.predict(X_test)
            stats = ta.clf_metrics(y_test,
                                   test_probs,
                                   mod_name=mod_name,
                                   average=AVERAGE)
    else:
        test_preds = mod.predict(X_test)
        stats = ta.clf_metrics(y_test,
                               test_preds,
                               mod_name=mod_name,
                               average=AVERAGE)
    #
    #
    # add metrics to MLFLow
    #
    log_stats_in_mlflow(stats)


# COMMAND ----------


