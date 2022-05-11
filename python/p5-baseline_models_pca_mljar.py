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

print(OUTCOME)
print(STRATIFY)

# COMMAND ----------

import pandas as pd

suffix = "_outcome_"+OUTCOME+"_stratify_"+STRATIFY

if EXPERIMENTING == True:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas_only_100'+suffix+'.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas_only_100'+suffix+'.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas_only_100'+suffix+'.csv')
else:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas'+suffix+'.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas'+suffix+'.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas'+suffix+'.csv')

# COMMAND ----------

train_pd.shape

# COMMAND ----------

X_train =  pd.concat([train_pd, val_pd], axis=0)
y_train = X_train.pop('target')
X_test = test_pd
y_test = X_test.pop('target')


# COMMAND ----------

from datetime import datetime
date_time = str(datetime.now()).replace('-','_').replace(':','_').replace('.','_')
mljar_folder = '/tmp/mljar_'+date_time
mljar_folder

# COMMAND ----------

from supervised.automl import AutoML
import mlflow

mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

automl = AutoML(results_path=mljar_folder)
automl.fit(X_train, y_train, )

# COMMAND ----------

y_predicted = automl.predict(X_test)
y_predicted

# COMMAND ----------

result = pd.DataFrame({"Predicted": y_predicted, "Target": np.array(y_test)})

# COMMAND ----------

filtro = result.Predicted == result.Target

# COMMAND ----------

print(filtro.value_counts(normalize=True))

# COMMAND ----------

from sklearn.metrics import f1_score
f1_score_weighted = f1_score(np.array(y_test), y_predicted, average='weighted')
mlflow.log_metric("testing f1 score weighted", f1_score_weighted)
f1_score_weighted

# COMMAND ----------

from sklearn.metrics import roc_auc_score
y_pred_proba = automl.predict_proba(X_test)
y_pred_proba

# COMMAND ----------

auc = roc_auc_score(y_test, y_pred_proba[:,1])
mlflow.log_metric("testing auc", auc)
auc

# COMMAND ----------

import tools.analysis as ta

mcnemar_x_square_test = ta.mcnemar_test(y_test, y_predicted, cc=True)
stat = mcnemar_x_square_test.loc[0,'stat']
pval = mcnemar_x_square_test.loc[0,'pval']
mlflow.log_metric("McNemar X' squre test - stat ", stat)
mlflow.log_metric("McNemar X' squre test - pval ", pval)
mcnemar_x_square_test

# COMMAND ----------

from sklearn.metrics import brier_score_loss
brier_score_loss = brier_score_loss (y_test, y_pred_proba[:,1])
mlflow.log_metric("brien score loss", brier_score_loss)
brier_score_loss

# COMMAND ----------

from sklearn.metrics import recall_score
recall_score = recall_score(y_test, y_predicted, average=AVERAGE)
mlflow.log_metric("recall score", recall_score)
recall_score

# COMMAND ----------

mlflow.log_param("average", AVERAGE)
mlflow.log_param("demographics", USE_DEMOG)
mlflow.log_param("outcome", OUTCOME)
mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

mlflow.log_artifacts('/tmp/mljar')

# COMMAND ----------

mlflow.end_run()

# COMMAND ----------

from sklearn import metrics
import matplotlib.pyplot as plt

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1])
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

# MAGIC %sh
# MAGIC mkdir /FileStore/too9

# COMMAND ----------

ls /tmp/ml

# COMMAND ----------

cp -R /tmp/mljar/* /dbfs/FileStore/too0/mljar/

# COMMAND ----------

# MAGIC %python
# MAGIC displayHTML('''<img src = "/dbfs/FileStore/too0/mljar/correlation_heatmap.png" stype="width:600px;height:600px;">''')

# COMMAND ----------

# MAGIC %sh
# MAGIC ls /dbfs/FileStore/too0/mljar/

# COMMAND ----------


