# Databricks notebook source
!pip install mlflow --quiet
!pip install mljar-supervised --quiet

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

X_train =  pd.concat([train_pd, val_pd], axis=0)
y_train = X_train.pop('target')
X_test = test_pd
y_test = X_test.pop('target')


# COMMAND ----------

# MAGIC %sh
# MAGIC rm -r /tmp/mljar

# COMMAND ----------

from supervised.automl import AutoML
import mlflow

mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

automl = AutoML(results_path='/tmp/mljar',mode='Compete')
automl.fit(X_train, y_train, )

# COMMAND ----------

y_predicted = automl.predict(X_test)

# COMMAND ----------

result = pd.DataFrame({"Predicted": y_predicted, "Target": np.array(y_test)})

# COMMAND ----------

filtro = result.Predicted == result.Target

# COMMAND ----------

print(filtro.value_counts(normalize=True))

# COMMAND ----------



# COMMAND ----------

from sklearn.metrics import f1_score
f1_score_weighted = f1_score(np.array(y_test), y_predicted, average='weighted')
mlflow.log_metric("testing f1 score weighted", f1_score_weighted)
f1_score_weighted

# COMMAND ----------

from sklearn.metrics import roc_auc_score
y_pred_proba = automl.predict_proba(X_test)[::,1]
auc = roc_auc_score(y_test, y_pred_proba)
mlflow.log_metric("testing auc", f1_score_weighted)
auc

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show

# COMMAND ----------

# confusion matrix
df = pd.DataFrame(result)
confusion_matrix = pd.crosstab(df['Target'], df['Predicted'], rownames = ['Target'], colnames=['Predicted'], margins=True)
confusion_matrix

# COMMAND ----------

#plot with seaborn
import seaborn as sn
import matplotlib.pyplot as plt

sn.heatmap(confusion_matrix, annot=True)
plt.show()

# COMMAND ----------



# COMMAND ----------

# MAGIC %cat /tmp/mljar/README.md

# COMMAND ----------

from IPython.display import display, Markdown
with open('/tmp/mljar/README.md', 'r') as fh:
    content = fh.read()
#display(Markdown(content))
pic = plt.imread('/tmp/mljar/ldb_performance_boxplot.png')
plt.imshow(pic, aspect='auto')

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /tmp/mljar/*

# COMMAND ----------


