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
  defaultValue='train_data_set_t',
  label='Trainning Table'
)
TRAINNING_DT=dbutils.widgets.get("train_dt")


dbutils.widgets.text(
  name='val_dt',
  defaultValue='val_data_set_t',
  label='Validation Table'
)
VALIDATION_DT=dbutils.widgets.get("val_dt")

dbutils.widgets.text(
  name='test_dt',
  defaultValue='test_data_set_t',
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

import numpy as np
import pandas as pd
import tools.analysis as ta
import tools.preprocessing as tp

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

from delta.tables import DeltaTable

X_train_dt = spark.table(f"{DATABASE}.{TRAINNING_DT}")
X_val_dt = spark.table(f"{DATABASE}.{VALIDATION_DT}")
X_test_dt = spark.table(f"{DATABASE}.{TESTING_DT}")


# COMMAND ----------


### to be used only if the input are spark dataframes
y_val = X_val_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()
y_test = X_test_dt.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()

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

#
# This part uses H2O Sparking Water
# Fi1st of all, you must install required libaries
#

!pip install requests
!pip install tabulate
!pip install future
!pip install h2o_pysparkling_3.2

# COMMAND ----------

#
# Import required libraries and start H2O
#
from pysparkling import *

hc = H2OContext.getOrCreate()


# COMMAND ----------

#
# This part use H2O AutoML
#
from pysparkling.ml import H2OAutoMLClassifier

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = "SparkingWater_AutoMl")

model = H2OAutoMLClassifier(labelCol = LABEL_COLUMN, 
                            maxModels=10, 
                            stoppingMetric="logloss",
                           seed=2022)

model_fit = model.fit(X_train_dt)

mlflow.log_param("training delta table", f"{DATABASE}.{TRAINNING_DT}")
mlflow.log_param("validation delta table", f"{DATABASE}.{VALIDATION_DT}")
mlflow.log_param("testing delta table", f"{DATABASE}.{TESTING_DT}")

bestmodel = model.getAllModels()[0]
mlflow.spark.log_model(bestmodel,"model")

prediction_val = bestmodel.transform(X_val_dt)
prediction_test = bestmodel.transform(X_test_dt)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name="H2O_sparking_water_AutoMl", average=AVERAGE)

log_stats_in_mlflow(stats)
mlflow.end_run()

# COMMAND ----------


