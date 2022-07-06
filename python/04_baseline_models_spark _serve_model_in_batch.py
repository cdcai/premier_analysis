# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

import mlflow

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
  name='input_table',
  defaultValue='test_data_set_t',
  label='Input Table'
)
INPUT_TABLE=dbutils.widgets.get("input_table")

dbutils.widgets.text(
  name='output_table',
  defaultValue='prediction_results',
  label='Output Table'
)
OUTPUT_TABLE=dbutils.widgets.get("output_table")

dbutils.widgets.dropdown("Metric","auc",["auc","f1", "npv", "ppv", "sens","spec"])
METRIC = dbutils.widgets.get("Metric")

# COMMAND ----------

experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id
assert DATABASE is not None
assert INPUT_TABLE is not None
assert OUTPUT_TABLE is not None
assert METRIC is not None
METRIC="metrics.testing_"+METRIC
INPUT_TABLE=DATABASE+"."+INPUT_TABLE
OUTPUT_TABLE=DATABASE+"."+OUTPUT_TABLE

# COMMAND ----------

def get_best_model(experiment_id, metric = 'metrics.testing_auc'):
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)  # get child experiments under the parent experiment id
    max_run = df_runs[df_runs[metric] == df_runs[metric].max()] # get the run that yield the max metric
    run_id = 'runs:/'+str(max_run['run_id'].values[0])+'/model'        # prepare run id string
    return run_id

# COMMAND ----------

def predict(sDF, experiment_id):
    import mlflow
    import pandas as pd
    
    # get best model
    logged_model = get_best_model(experiment_id)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    p = loaded_model.predict(sDF.toPandas())
    return p

# COMMAND ----------

dbutils.fs.rm("hive_metastore.too9_premier_analysis_demo.prediction_results")


# COMMAND ----------

run_id = get_best_model(experiment_id=experiment_id, metric=METRIC)
model = mlflow.spark.load_model(run_id)
batch_spark = spark.table(INPUT_TABLE)
prediction = model.transform(batch_spark)
spark.sql("drop table if exists "+OUTPUT_TABLE+";")
prediction.write.mode("overwrite").format("delta").saveAsTable(OUTPUT_TABLE)

# COMMAND ----------

›
