# Databricks notebook source
!pip install mlflow

# COMMAND ----------

import tools.analysis as ta
import tools.preprocessing as tp
import mlflow
import pandas as pd

# COMMAND ----------

AVERAGE = 'weighted'

# COMMAND ----------

def get_array_of_probs (spark_df):
    from pyspark.ml.functions import vector_to_array

    probsp = spark_df.select(vector_to_array("probability", "float32").alias("probability")).toPandas()
    probss= probsp['probability']
    probsn = probss.to_numpy()
    prob_list = list()
    for prob in probsn: 
        prob_list = prob_list + [prob[1]]
    prob_array = np.array(prob_list)
    
    return prob_array

# COMMAND ----------

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

def get_statistics_from_predict(test_predict, y_test, mod_name, average=AVERAGE):
    stats = ta.clf_metrics(y_test,
                           test_predict,
                           mod_name=mod_name,
                           average=average)
    return stats

# COMMAND ----------

def log_stats_in_mlflow(stats):
    for i in stats:
        if not isinstance(stats[i].iloc[0], str):
            mlflow.log_metric("Testing "+i, stats[i].iloc[0])

# COMMAND ----------

def log_param_in_mlflow():
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

import numpy as np

# COMMAND ----------

NROWS = 1000
NCOLS = 10

# COMMAND ----------

a = np.random.randint(0,2,(NROWS,NCOLS))

# COMMAND ----------

a.shape

# COMMAND ----------

c = list()
for i in range(0,NCOLS-1): c = c + ["c"+str(i)]
c = ["y"] + c

# COMMAND ----------

a_rdd = spark.sparkContext.parallelize(a)
a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c) )

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
vecAssembler = VectorAssembler(outputCol="features")
vecAssembler.setInputCols(c[1:])
a_spark_vector = vecAssembler.transform(a_df)

# COMMAND ----------

X_train_spark, X_val_spark, X_test_spark = a_spark_vector.select(['y', 'features']).randomSplit([0.6, 0.2, 0.2], seed=2022)

# COMMAND ----------

y_val_list = X_val_spark.select('y').rdd.map(lambda r: r[0]).collect()  # python list
y_val = np.array(y_val_list)  # numpy array


# COMMAND ----------

y_test_list = X_test_spark.select('y').rdd.map(lambda r: r[0]).collect()  # python list
y_test = np.array(y_test_list)  # numpy array

# COMMAND ----------

from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import LogisticRegression

bool = True

KNOWN_REGRESSORS = {
    r.__name__: r
    for r in [LogisticRegression, GBTClassifier, DecisionTreeClassifier, RandomForestClassifier, LinearSVC]
    #for r in [LogisticRegression]
}
KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES = ["LogisticRegression", "GBTClassifier","RandomForestClassifier"]

mlflow.end_run()
with mlflow.start_run(
):
    mlflow.pyspark.ml.autolog()
    for model_name, model_class in KNOWN_REGRESSORS.items():
        with mlflow.start_run(
            run_name=f"premier_analysis_{model_name}",
            nested=True,
        ):
            model = model_class(featuresCol='features',labelCol='y')
            model_fit = model.fit(X_train_spark)
            #mlflow.log_model(model, model_name)
            
            predictions_test = model_fit.transform(X_test_spark)

            if model_name in KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES:
                predictions_val = model_fit.transform(X_val_spark)
                val_probs  = get_array_of_probs (predictions_val)
                test_probs = get_array_of_probs (predictions_test)
                stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name=model_name, average=AVERAGE)
            else:
                y_predict = predictions_test.select('prediction').toPandas()['prediction'].to_numpy()
                y_predict_count = y_predict.shape[0]
                y_test_count = y_test.shape[0]

                
                print("y_test_count: "+str(y_test_count)+" y_predict_count: "+str(y_predict_count))
                assert y_predict_count == y_test_count

                stats = get_statistics_from_predict(y_predict, 
                                            y_test, 
                                            str(model_name), 
                                            average=AVERAGE)            
            log_stats_in_mlflow(stats)
            #log_param_in_mlflow()
            
            if bool == True:
                bool = False
                all_stats = stats
            else:
                all_stats = all_stats.append(stats)

    display(all_stats)
    all_stats.to_csv("/tmp/stats.csv")
    mlflow.log_artifact("/tmp/stats.csv")
    


# COMMAND ----------

import mlflow
logged_model = 'runs:/36b9fd6e637744a691a2a582d31325fe/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
#import pandas as pd
#loaded_model.predict(pd.DataFrame(data))

# COMMAND ----------

p = X_test_spark.toPandas()

# COMMAND ----------

def get_best_model(experiment_id = "6f19b824d95e4a85a8a7b235718cb9b4", metric = 'metrics.Testing auc'):
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)  # get child experiments under the parent experiment id
    max_run = df_runs[df_runs[metric] == df_runs[metric].max()] # get the run that yield the max metric
    run_id = 'runs:/'+str(max_run['run_id'][1])+'/model'        # prepare run id string
    return run_id

# COMMAND ----------

import mlflow
logged_model = get_best_model()

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(run_id)

# Predict on a Pandas DataFrame.
import pandas as pd
p = loaded_model.predict(X_test_spark.toPandas())

# COMMAND ----------

p

# COMMAND ----------


