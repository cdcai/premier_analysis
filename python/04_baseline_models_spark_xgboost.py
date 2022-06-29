# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='2331064140756563',
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

dbutils.widgets.dropdown("experimenting", "False",  ["True", "False"])
EXPERIMENTING = dbutils.widgets.get("experimenting")
if EXPERIMENTING == "True": EXPERIMENTING = True
else: EXPERIMENTING = False

# COMMAND ----------

import mlflow
experiment = dbutils.widgets.get("experiment_id")
assert experiment is not None
current_experiment = mlflow.get_experiment(experiment)
assert current_experiment is not None
experiment_id= current_experiment.experiment_id


# COMMAND ----------

import argparse
import os
import pickle as pkl
from importlib import reload

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix
from sklearn.model_selection import train_test_split

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
#STRATIFY ='all'

# Setting the directories and importing the data
# If no args are passed to overwrite these values, use repo structure to construct
# Setting the directories
output_dir = '/dbfs/home/tnk6/premier_output/'
data_dir = '/dbfs/home/tnk6/premier/'

if data_dir is not None:
    data_dir = os.path.abspath(data_dir)

if output_dir is not None:
    output_dir = os.path.abspath(output_dir)

pkl_dir = os.path.join(output_dir, "pkl", "")
stats_dir = os.path.join(output_dir, "analysis", "")
probs_dir = os.path.join(stats_dir, "probs", "")

# COMMAND ----------

stats_dir

# COMMAND ----------

# Create analysis dirs if it doesn't exist
[
    os.makedirs(directory, exist_ok=True)
    for directory in [stats_dir, probs_dir, pkl_dir]
]

with open(pkl_dir + CHRT_PRFX + "trimmed_seqs.pkl", "rb") as f:
    inputs = pkl.load(f)

with open(pkl_dir + "all_ftrs_dict.pkl", "rb") as f:
    vocab = pkl.load(f)

with open(pkl_dir + "feature_lookup.pkl", "rb") as f:
    all_feats = pkl.load(f)

with open(pkl_dir + "demog_dict.pkl", "rb") as f:
    demog_dict = pkl.load(f)
    demog_dict = {k: v for v, k in demog_dict.items()}

# COMMAND ----------

# Separating the inputs and labels
features = [t[0] for t in inputs]
demog = [t[1] for t in inputs]
cohort = pd.read_csv(os.path.join(output_dir, CHRT_PRFX, 'cohort.csv'))
labels = cohort[OUTCOME]

# Counts to use for loops and stuff
n_patients = len(features)
n_features = np.max(list(vocab.keys()))
n_classes = len(np.unique(labels))
binary = n_classes <= 2

# Converting the labels to an array
y = np.array(labels, dtype=np.uint8)

# COMMAND ----------

# Optionally limiting the features to only those from the first day
# of the actual COVID visit
if DAY_ONE_ONLY:
    features = [l[-1] for l in features]
else:
    features = [tp.flatten(l) for l in features]

# Optionally mixing in the demographic features
if USE_DEMOG:
    new_demog = [[i + n_features for i in l] for l in demog]
    features = [features[i] + new_demog[i] for i in range(n_patients)]
    demog_vocab = {k + n_features: v for k, v in demog_dict.items()}
    vocab.update(demog_vocab)
    n_features = np.max([np.max(l) for l in features])
    all_feats.update({v: v for k, v in demog_dict.items()})

# Converting the features to a sparse matrix
mat = lil_matrix((n_patients, n_features + 1))
for row, cols in enumerate(features):
    mat[row, cols] = 1

# Converting to csr because the internet said it would be faster
X = mat.tocsr()

# Splitting the data; 'all' will produce the same test sample
# for every outcome (kinda nice)
if STRATIFY == 'all':
    outcomes = ['icu', 'misa_pt', 'death']
    strat_var = cohort[outcomes].values.astype(np.uint8)
else:
    strat_var = y

# COMMAND ----------

train, test = train_test_split(range(n_patients),
                               test_size=TEST_SPLIT,
                               stratify=strat_var,
                               random_state=RAND)

# Doing a validation split for threshold-picking on binary problems
train, val = train_test_split(train,
                              test_size=VAL_SPLIT,
                              stratify=strat_var[train],
                              random_state=RAND)

# COMMAND ----------

if EXPERIMENTING == True: SAMPLE = 1000
else: SAMPLE = X.shape[0]

# COMMAND ----------

X.shape

# COMMAND ----------

def convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0

    a_rdd = spark.sparkContext.parallelize(a_dataframe.to_numpy(),100)
    a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names) )

    vecAssembler = VectorAssembler(outputCol="features")
    vecAssembler.setInputCols(c_names)
    spark_df = vecAssembler.transform(a_df)
    old_col_name = "_"+str(a_dataframe.shape[1]) # vector assembler would change the name of collumn y
    spark_df = spark_df.withColumnRenamed (old_col_name,'label')
    
    return spark_df

# COMMAND ----------

def incrementaly_convert_pandas_to_spark_with_vectors(a_dataframe,c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler

    assert isinstance (a_dataframe,  pd.DataFrame)
    assert c_names is not None
    assert len(c_names)>0

    inc=min(10000, a_dataframe.shape[0])
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):

        
        if (i*inc) < a_dataframe.shape[0]:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
            a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names) )

            #a_df = spark.createDataFrame(a_rdd, c_names)

            vecAssembler = VectorAssembler(outputCol="features")
            vecAssembler.setInputCols(c_names)
            a_spark_vector = vecAssembler.transform(a_df)

            if bool == True:
                spark_df = a_spark_vector
                bool = False
            else:
                spark_df = spark_df.union(a_spark_vector)
    
    old_col_name = "_"+str(a_dataframe.shape[1]) # vector assembler would change the name of collumn y
    spark_df = spark_df.withColumnRenamed (old_col_name,'label')
    return spark_df


def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

#
# create pandas frames from X
#
c_names = change_columns_names(X)

X_train_pandas = pd.DataFrame(X[train][:SAMPLE].toarray())
X_train_pandas['label'] = y[train][:SAMPLE]

X_val_pandas = pd.DataFrame(X[val][:SAMPLE].toarray())
X_val_pandas['label'] = y[val][:SAMPLE]

X_test_pandas = pd.DataFrame(X[test][:SAMPLE].toarray())
X_test_pandas['label'] = y[test][:SAMPLE]

# COMMAND ----------

from threading import Thread

def pandas_to_spark(pDF, c_names, results, index): 
    results[index] = convert_pandas_to_spark_with_vectors(pDF, c_names).select(['label','features'])
    
list_of_pandas = [X_train_pandas, X_val_pandas, X_test_pandas]

results = [None] * len(list_of_pandas)
threads = [None] * len(list_of_pandas)

for index in range(0,len(threads)):
    threads [index] = Thread(target=pandas_to_spark, args=(list_of_pandas[index], c_names, results, index))
    threads[index].start()

for i in range(len(threads)):
    threads[i].join()
    
X_train_spark = results[0].cache()
X_val_spark   = results[1].cache()
X_test_spark  = results[2].cache()

# COMMAND ----------

#X_train_spark = convert_pandas_to_spark_with_vectors(X_train_pandas, c_names).select(['label','features'])
#X_val_spark =   convert_pandas_to_spark_with_vectors(X_val_pandas, c_names).select(['label','features'])
#X_test_spark =  convert_pandas_to_spark_with_vectors(X_test_pandas, c_names).select(['label','features'])

# COMMAND ----------

X_train_pandas = None
X_test_pandas  = None
X_val_pandas   = None
X = None
y = None

# COMMAND ----------

def add_index_to_sDF(a_df,columns):
    a_rdd =  a_df.rdd.zipWithIndex()
    a_df =   a_rdd.toDF()
    for column_name in collumns:
        a_df =   a_df.withColumn(column_name,a_df['_1'].getItem(column_name))
    a_df =   a_df.withColumnRenamed('_2','id')
    a_df =   a_df.select(columns)
    return a_df

# COMMAND ----------

def get_array_of_probs (predictions_sDF):
    from pyspark.ml.functions import vector_to_array
    import numpy as np

    p = predictions_sDF.select(vector_to_array("probability", "float32").alias("probability")).toPandas()['probability'].to_numpy()
    
    return np.array(list(map(lambda x: x[1], p)))

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
            mlflow.log_metric("testing_"+i, stats[i].iloc[0])

# COMMAND ----------

def log_param_in_mlflow():
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)

# COMMAND ----------

def get_array_of_probabilities_from_sparkling_water_prediction(predict_sDF):
    p = predict_sDF.select('detailed_prediction').collect()
    probs = list()
    for row in range(len(p)):
        prob = p[row].asDict()['detailed_prediction']['probabilities'][1]
        probs = probs + [prob]
    
    return np.asarray(probs)

# COMMAND ----------

# MAGIC %md
# MAGIC y_val = X_val_spark.select('label').toPandas()['label'].to_numpy()
# MAGIC y_test = X_test_spark.select('label').toPandas()['label'].to_numpy()

# COMMAND ----------

# MAGIC %md
# MAGIC y_val  = np.array(X_val_spark.select('label').rdd.map(lambda r: r[0]).collect())  # numpy array
# MAGIC y_test = np.array(X_test_spark.select('label').rdd.map(lambda r: r[0]).collect() )  # numpy array

# COMMAND ----------

from threading import Thread

def spark_to_numpy(sDF,  results, index): 
    results[index] =  np.array(sDF.select('label').rdd.map(lambda r: r[0]).collect())
    
list_of_sparks = [X_val_spark, X_test_spark]
results = [None] * 2
threads = [None] * 2

for index in range(0,len(threads)):
    threads [index] = Thread(target=spark_to_numpy, args=(list_of_sparks[index], results, index))
    threads [index].start()

for i in range(len(threads)):
    threads[i].join()
    
y_val  = results[0]
y_test = results[1]


# COMMAND ----------

!pip install requests
!pip install tabulate
!pip install future
!pip install h2o_pysparkling_3.2

# COMMAND ----------

from pysparkling import *

hc = H2OContext.getOrCreate()

# COMMAND ----------

from pysparkling.ml import H2OXGBoostClassifier 

run_name = "Premier_Analysis_H2O_SparkingWater_XGBoost"
mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = run_name)

model = H2OXGBoostClassifier(labelCol = 'label', 
                            stoppingMetric="logloss")

model_fit = model.fit(X_train_spark)
mlflow.spark.log_model(model_fit,run_name)

prediction_val = model_fit.transform(X_val_spark)
prediction_test = model_fit.transform(X_test_spark)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, 
                                          test_probs, 
                                          y_val, y_test, 
                                          mod_name=run_name, 
                                          average=AVERAGE)

log_stats_in_mlflow(stats)
log_param_in_mlflow()
mlflow.end_run()

# COMMAND ----------

from pysparkling.ml import H2OXGBoostClassifier 
from pysparkling.ml import H2OGridSearch

run_name = "Premier_Analysis_H2O_SparkingWater_tunned_XGBoost"

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = run_name)

algo = H2OXGBoostClassifier (labelCol = 'label', 
                             stoppingMetric="logloss",
                             booster="gbtree",
                             treeMethod="hist",
                             growPolicy="lossguide",
                            backend="gpu")

hyperSpace = {"eta":       [0.001, 0.01, 0.1],
             "maxDepth":   [5, 7, 11, 13],
             "ntrees":     [53, 61, 71, 83],
             "regAlpha":   [0, 0.1, 0.3, 0.5, 1.1, 2.3, 3.12],
             "regLambda": [0, 0.1, 0.3, 0.5, 1.1, 2.3, 3.12],
             "gamma":      [0, 0.1, 0.3, 0.5, 1.1, 2.3, 3.12]}

grid = H2OGridSearch(hyperParameters=hyperSpace, 
                     parallelism=0,
                     algo=algo, 
                     strategy="RandomDiscrete",
                     maxModels=100,
                     seed=2022,)

model = grid.fit(X_train_spark)

mlflow.spark.log_model(model, run_name)

prediction_val =  model.transform(X_val_spark)
prediction_test = model.transform(X_test_spark)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name=run_name, average=AVERAGE)

log_stats_in_mlflow(stats)
log_param_in_mlflow()
#
# log XGBoost tunning parameters of best model
#
mlflow.log_param("eta", model.getEta())
mlflow.log_param("maxDepth", model.getMaxDepth())
mlflow.log_param("ntrees", model.getNtrees())
mlflow.log_param("regAlpha", model.getRegAlpha())
mlflow.log_param("regLambda", model.getRegLambda())
mlflow.log_param("gamma", model.getGamma())
mlflow.log_param("booster", model.getBooster())
mlflow.log_param("treeMethod", model.getTreeMethod())
mlflow.log_param("growPolicy", model.getGrowPolicy())

mlflow.end_run()

# COMMAND ----------

mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id, 
                 run_name = "Premier_Analysis_H2O_SparkingWater_AutoMl")

model = H2OAutoMLClassifier(labelCol = 'label', 
                            maxModels=100, 
                            stoppingMetric="logloss")

model_fit = model.fit(X_train_spark)

bestmodel = automl.getAllModels()[0]
mlflow.spark.log_model(bestmodel,"h2o_sparking_water")

prediction_val = bestmodel.transform(X_val_spark)
prediction_test = bestmodel.transform(X_test_spark)
val_probs  = get_array_of_probabilities_from_sparkling_water_prediction (prediction_val)
test_probs = get_array_of_probabilities_from_sparkling_water_prediction (prediction_test)
stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name="H2O_sparking_water_AutoMl", average=AVERAGE)

log_stats_in_mlflow(stats)
log_param_in_mlflow()
mlflow.end_run()

# COMMAND ----------

display(stats)

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

run_id = get_best_model(experiment_id=experiment_id)

# COMMAND ----------

run_id

# COMMAND ----------

model = mlflow.spark.load_model(run_id)

# COMMAND ----------

prediction = model.transform(X_val_spark)


# COMMAND ----------

display(prediction)

# COMMAND ----------

predict(X_val_spark, experiment_id)

# COMMAND ----------


