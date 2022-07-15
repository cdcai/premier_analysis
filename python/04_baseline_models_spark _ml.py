# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='1767884267538855',
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
  defaultValue='train_data_set_demo_july',
  label='Trainning Table'
)
TRAINNING_DT=dbutils.widgets.get("train_dt")


dbutils.widgets.text(
  name='val_dt',
  defaultValue='val_data_set_demo_july',
  label='Validation Table'
)
VALIDATION_DT=dbutils.widgets.get("val_dt")

dbutils.widgets.text(
  name='test_dt',
  defaultValue='test_data_set_demo_july',
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
import mlflow

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
# This part use Spark ML binary classification algoritms#
#
from pyspark.ml.classification import LinearSVC as svc
from pyspark.ml.classification import DecisionTreeClassifier as dtc
from pyspark.ml.classification import GBTClassifier as gbt
from pyspark.ml.classification import RandomForestClassifier as rfc
from pyspark.ml.classification import LogisticRegression as lr

model_class = [lr(maxIter=5000,featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN),
              gbt(seed=2022,featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN), 
              dtc(seed=2022,featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN), 
              rfc (numTrees=500,seed=2022,featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN), 
              svc(maxIter=100,featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN)]
model_names = ["LogisticRegression",
              "GBTClassifier", 
              "DecisionTreeClassifier", 
              "RandomForestClassifier", 
              "LinearSVC"]
KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES = ["LogisticRegression", 
                                             "GBTClassifier",
                                             "RandomForestClassifier"]

mlflow.end_run()

for  i in range(len(model_class)):
    modelName = model_names[i]
    run_name="spark_with_delta_tables_"+modelName
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        model = model_class[i]
        model_fit = model.fit(X_train_dt)

        mlflow.spark.log_model(model_fit, "model")
        # to make sure model can be found progrmatically, 
        # use "model" as the name of the model
        mlflow.log_param("training delta table", f"{DATABASE}.{TRAINNING_DT}")
        mlflow.log_param("validation delta table", f"{DATABASE}.{VALIDATION_DT}")
        mlflow.log_param("testing delta table", f"{DATABASE}.{TESTING_DT}")

        predictions_test = model_fit.transform(X_test_dt)

        if modelName in KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES:
            predictions_val = model_fit.transform(X_val_dt)
            val_probs  = get_array_of_probs (predictions_val)
            test_probs = get_array_of_probs (predictions_test)
            stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, 
                                                      mod_name=modelName, average=AVERAGE)
        else:
            y_predict = predictions_test.select('prediction').toPandas()['prediction'].to_numpy()

            stats = get_statistics_from_predict(y_predict, 
                                        y_test, 
                                        str(modelName), 
                                        average=AVERAGE)
        log_stats_in_mlflow(stats)

# COMMAND ----------

#
# This part provides an examples about how to use hyper parameters with Spark ML
#
from pyspark.ml.classification import LogisticRegression as spark_lr
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

mlflow.end_run()
run_name="spark_with_delta_tables_tunned_lr"
with mlflow.start_run(
    run_name=run_name,
    experiment_id=experiment_id,
):
    lr = spark_lr(featuresCol=FEATURES_COLUMN,labelCol=LABEL_COLUMN)

    paramGrid = (ParamGridBuilder()
         .addGrid(lr.regParam, [0.01, 0.5, 2.0])
         .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
         .addGrid(lr.maxIter, [100, 500, 1000])
         .build())

    evaluator = BinaryClassificationEvaluator()

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, parallelism=100)

    cvModel = cv.fit(X_train_dt)

    mlflow.spark.log_model(cvModel.bestModel,  "model")
    # to make sure model can be found progrmatically, 
    # use "model" as the name of the model
    
    mlflow.log_param("elasticNetParam", cvModel.bestModel.getElasticNetParam())
    mlflow.log_param("maxIter", cvModel.bestModel.getMaxIter())
    mlflow.log_param("regParam", cvModel.bestModel.getRegParam())
    mlflow.log_param("training delta table", f"{DATABASE}.{TRAINNING_DT}")
    mlflow.log_param("validation delta table", f"{DATABASE}.{VALIDATION_DT}")
    mlflow.log_param("testing delta table", f"{DATABASE}.{TESTING_DT}")

    predictions_test = cvModel.bestModel.transform(X_test_dt)
    predictions_val  = cvModel.bestModel.transform(X_val_dt)
    val_probs  = get_array_of_probs (predictions_val)
    test_probs = get_array_of_probs (predictions_test)
    stats = get_statistics_from_probabilities(val_probs, 
                                              test_probs, 
                                              y_val, 
                                              y_test, 
                                              mod_name="lr", 
                                              average=AVERAGE)

    log_stats_in_mlflow(stats)


# COMMAND ----------


