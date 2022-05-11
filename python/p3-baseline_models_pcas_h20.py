# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

!pip install requests --quiet
!pip install tabulate --quiet
!pip install future --quiet
!pip install -f http://h2o-release.s3.amazon.com/h2o/latest_stable_Py.html h2o --quiet

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

dbutils.widgets.dropdown("experimenting", "True",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id


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

#
# Initialize H2O
#
import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)
h2o.init()

# COMMAND ----------

import pandas as pd
import pandas as pd

if EXPERIMENTING == True:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas_only_100.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas_only_100.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas_only_100.csv')
else:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas.csv')

# COMMAND ----------

X_train =  pd.concat([train_pd, val_pd], axis=0)

# COMMAND ----------

#
# Prepare sets for H2O
#
target = "target"

X_train_h2o_df = h2o.H2OFrame(X_train)
x = X_train_h2o_df.columns
x.remove(target)
X_train_h2o_df[target] = X_train_h2o_df[target].asfactor()

# COMMAND ----------

# Run AutoML for 20 base models
import mlflow

mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

aml = H2OAutoML(max_models=5, seed=1)
aml.train(x=x, y=target, training_frame=X_train_h2o_df)
mlflow.log_metric("rmse", aml.leader.rmse())
mlflow.log_metric("mse", aml.leader.mse())
mlflow.log_metric("log_loss", aml.leader.logloss())
#mlflow.log_metric("mean_per_class_error", aml.leader.mean_per_class_error())
mlflow.log_metric("auc", aml.leader.auc())
mlflow.h2o.log_model(aml.leader, "model")
lb = aml.leaderboard
lb = h2o.automl.get_leaderboard(aml, extra_columns='ALL')
print(lb.head(rows=lb.nrows))

# COMMAND ----------

lb = h2o.automl.get_leaderboard(aml, extra_columns="ALL")
lb

# COMMAND ----------

best_model = aml.get_best_model()
best_model

# COMMAND ----------


X_test_h2o_df = h2o.H2OFrame(test_pd)
X_test_h2o_df[target] = X_test_h2o_df[target].asfactor()
y_test = test_pd['target'].astype('int')

predictions = best_model.predict(X_test_h2o_df)

# COMMAND ----------

y_predicted = predictions.as_data_frame()
y_predicted

# COMMAND ----------

from sklearn.metrics import f1_score
f1_score_weighted = f1_score(y_test, y_predicted['predict'], average='weighted')
mlflow.log_metric("testing f1 score weighted", f1_score_weighted)
f1_score_weighted

# COMMAND ----------

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_test, y_predicted['True'])
mlflow.log_metric("testing auc", f1_score_weighted)
auc

# COMMAND ----------


mlflow.end_run()

# COMMAND ----------


