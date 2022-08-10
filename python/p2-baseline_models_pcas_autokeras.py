# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

!pip install git+https://github.com/keras-team/keras-tuner.git --quiet
!pip install autokeras --quiet


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

train_pd.shape

# COMMAND ----------

x_train = train_pd
y_train = x_train.pop('target')
y_train = pd.DataFrame(y_train)

x_test = test_pd
y_test = x_test.pop('target')
y_test = pd.DataFrame(y_test)

x_val = val_pd
y_val = x_val.pop('target')
y_val = pd.DataFrame(y_val)

# COMMAND ----------

#
# import autokeras required libraris
#
import numpy as np
import pandas as pd
import tensorflow as tf
import autokeras as ak


# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /tmp/autokeras

# COMMAND ----------

# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10,directory="/tmp/autokeras")
# Feed the structured data classifier with training data.

import mlflow
mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

# COMMAND ----------

# Predict with the best model.
y_predicted = clf.predict(x_test)
y_predicted.shape

# COMMAND ----------

y_test['target'] = y_test['target'].astype('int')
y_test['target']

# COMMAND ----------

from sklearn.metrics import f1_score
f1_score_weighted = f1_score(np.array(y_test), y_predicted, average='weighted')
mlflow.log_metric("testing f1 score weighted", f1_score_weighted)
f1_score_weighted

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------


