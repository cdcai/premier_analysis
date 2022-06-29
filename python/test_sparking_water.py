# Databricks notebook source
!pip install requests
!pip install tabulate
!pip install future
!pip install h2o_pysparkling_3.2

# COMMAND ----------

!pip install mlflow

# COMMAND ----------

from pysparkling import *
hc = H2OContext.getOrCreate()

# COMMAND ----------

import mlflow
import mlflow.h2o



# COMMAND ----------

import h2o
frame = h2o.import_file("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/prostate/prostate.csv")
sparkDF = hc.asSparkFrame(frame)
sparkDF = sparkDF.withColumn("CAPSULE", sparkDF.CAPSULE.cast("string"))
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])

# COMMAND ----------

display(sparkDF)

# COMMAND ----------

from pysparkling.ml import H2OAutoML
import mlflow

mlflow.end_run()
mlflow.start_run()
automl = H2OAutoML(labelCol="CAPSULE", ignoredCols=["ID"]).setMaxModels(2).setSortMetric("logloss")

# COMMAND ----------

model = automl.fit(trainingDF)

# COMMAND ----------

bestmodel = automl.getAllModels()[0]
mlflow.spark.log_model(bestmodel,"test")

# COMMAND ----------

import mlflow.h2o

# Set metrics to log
mlflow.log_metric("log_loss", bestmodel.getTrainingMetrics().get('Logloss'))
mlflow.log_metric("AUC", bestmodel.getTrainingMetrics().get('AUC'))

# Log and save best model (mlflow.h2o provides API for logging & loading H2O models)
mlflow.log_model(automl,"automl")

model_uri = mlflow.get_artifact_uri("model")
print(f'AutoML best model saved in {model_uri}')

# Get IDs of current experiment run
exp_id = experiment.experiment_id
run_id = mlflow.active_run().info.run_id

# Save leaderboard as CSV
lb = get_leaderboard(automl, extra_columns='ALL')
lb_path = f'mlruns/{exp_id}/{run_id}/artifacts/model/leaderboard.csv'
lb.as_data_frame().to_csv(lb_path, index=False) 

# COMMAND ----------


