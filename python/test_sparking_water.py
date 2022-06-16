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

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='2196115870945772',
  label='Experiment ID'
)

experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id

# COMMAND ----------

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()

# COMMAND ----------

import h2o
frame = h2o.import_file("https://raw.githubusercontent.com/h2oai/sparkling-water/master/examples/smalldata/prostate/prostate.csv")
sparkDF = hc.asSparkFrame(frame)
sparkDF = sparkDF.withColumn("CAPSULE", sparkDF.CAPSULE.cast("string"))
[trainingDF, testingDF] = sparkDF.randomSplit([0.8, 0.2])

# COMMAND ----------

#from pysparkling.ml import H2OAutoML
from h2o.automl import H2OAutoML

automl = H2OAutoML(labelCol="CAPSULE", ignoredCols=["ID"])

# COMMAND ----------

automl.setMaxModels(10)

# COMMAND ----------

automl.setSortMetric("logloss")
model = automl.fit(trainingDF)

# COMMAND ----------

leaderboard = automl.getLeaderboard()
leaderboard.show(truncate = False)

# COMMAND ----------

bestmodel = automl.getAllModels()[0]
type(bestmodel)
bestmodel.getModelDetails()

# COMMAND ----------

bestmodel.getHGLM()

# COMMAND ----------

bestmodel.getTrainingMetrics().get('Logloss')

# COMMAND ----------



# after downloading mojo file, we treat it as an artifact
# and log it using mlflow.log_artifact()
mlflow.log_artifact(temp_folder+"h2o-model-mojo.zip", artifact_path="h2o-model-mojo")

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


