# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

!pip install git+https://github.com/keras-team/keras-tuner.git --quiet
!pip install autokeras --quiet


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

dbutils.widgets.dropdown("average", "weighted", ['micro', 'macro', 'weighted'])
AVERAGE = dbutils.widgets.get("average")

# COMMAND ----------

import mlflow
experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id


# COMMAND ----------

import pandas as pd
train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas.csv')
val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas.csv')
test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas.csv')

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

# It tries 10 different models.
clf = ak.StructuredDataClassifier(overwrite=True, max_trials=10,directory="/tmp")
# Feed the structured data classifier with training data.

import mlflow
mlflow.start_run(experiment_id=experiment_id)
clf.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10)

# COMMAND ----------

# Predict with the best model.
predicted_y = clf.predict(x_test)

# COMMAND ----------

y_test['target'] = y_test['target'].astype('int')
y_test['target']

# COMMAND ----------

import tools.analysis as ta
import tools.preprocessing as tp


stats = ta.clf_metrics(y_test.to_numpy(), predicted_y, mod_name="h2o",average=AVERAGE)

# COMMAND ----------

stats

# COMMAND ----------

stats.to_csv('/dbfs/home/tnk6/premier_output/analysis/results_autokeras.csv')

# COMMAND ----------

mlflow.log_metric("f1", stats['f1'][0])

mlflow.end_run()
