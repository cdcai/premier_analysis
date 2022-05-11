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

dbutils.widgets.dropdown("experimenting", "True",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")

# COMMAND ----------

import pandas as pd

if EXPERIMENTING:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas_only_100.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas_only_100.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas_only_100.csv')
else:
    train_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/train_pcas.csv')
    val_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/val_pcas.csv')
    test_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/test_pcas.csv')
    
train_pd['time_collumn']=1
val_pd['time_collumn']=2
test_pd['time_collumn']=3

X_pd = pd.concat([train_pd, val_pd, test_pd], axis=0)

# COMMAND ----------

from databricks import automl


summary = automl.classify(X_pd, target_col='target',time_col='time_collumn', timeout_minutes=600)

# COMMAND ----------

import mlflow

model_uri = summary.best_trial.model_path


# COMMAND ----------

# Run inference using the best model
model = mlflow.pyfunc.load_model(model_uri)
model

# COMMAND ----------


