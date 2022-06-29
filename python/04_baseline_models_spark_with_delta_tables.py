# Databricks notebook source
!pip install mlflow --quiet

# COMMAND ----------

dbutils.widgets.removeAll()
dbutils.widgets.text(
  name='experiment_id',
  defaultValue='3289544690048618',
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

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
#STRATIFY ='all'
LABEL_COLUMN='label'

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

if EXPERIMENTING == True: 
    ROWS = 1000
    COLS = 100
else:
    ROWS = X.shape[0]
    COLS = X.shape[1]

# COMMAND ----------

def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

#
# create pandas frames from X
#
c_names = change_columns_names(X)[:COLS]

X_train_pandas = pd.DataFrame(X[train][:ROWS,:COLS].toarray(),columns=c_names)
X_train_pandas[LABEL_COLUMN] = y[train][:ROWS].astype("int")

X_val_pandas = pd.DataFrame(X[val][:ROWS,:COLS].toarray(),columns=c_names)
X_val_pandas[LABEL_COLUMN] = y[val][:ROWS].astype("int")

X_test_pandas = pd.DataFrame(X[test][:ROWS,:COLS].toarray(),columns=c_names)
X_test_pandas[LABEL_COLUMN] = y[test][:ROWS].astype("int")

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE DATABASE IF NOT EXISTS too9_premier_analysis_demo;

# COMMAND ----------

X_train_sdf = spark.createDataFrame(X_train_pandas)

# COMMAND ----------

X_train_sdf.write.mode("overwrite").format("delta").saveAsTable("too9_premier_analysis_demo.train_data_set")

# COMMAND ----------

#
# convert delta table to spark data frame
#
def get_sDF_from_delta_table(delta_table_name):
    from delta.tables import DeltaTable
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.feature import VectorAssembler

    
    sDF = spark.table(delta_table_name)
    
    sDF = sDF.withColumnRenamed ("target",LABEL_COLUMN)
    #sDF = sDF.withColumn("features",array_to_vector("features").alias("features"))
    
    return sDF

# COMMAND ----------

#
# create spark data frames from delta tables
#
X_train_spark = get_sDF_from_delta_table("tnk6_premier_demo.output_train_dataset")
X_val_spark = get_sDF_from_delta_table("tnk6_premier_demo.output_val_dataset")
X_test_spark = get_sDF_from_delta_table("tnk6_premier_demo.output_test_dataset")

# COMMAND ----------

print(X_train_spark.rdd.getNumPartitions())
print(X_train_spark.count())
print(X_val_spark.rdd.getNumPartitions())
print(X_val_spark.count())
print(X_test_spark.rdd.getNumPartitions())
print(X_test_spark.count())

# COMMAND ----------

from pyspark.ml.functions import vector_to_array

#train_pDF=X_train_spark.withColumn("features",vector_to_array("features").alias("features")).toPandas()
train_pDF=X_train_spark.toPandas()
train_pDF.shape

# COMMAND ----------

D_train=train_pDF['features'][:1].apply(pd.Series).to_numpy().astype("int")
D_train.shape
print(type(D_train))
print(D_train.shape)

# COMMAND ----------

X_train=np.array(X[train][:1].todense()).astype("int")
X_train.shape
print(type(X_train))
print(X_train.shape)

# COMMAND ----------

D_train.sum()

# COMMAND ----------

X_train.sum()

# COMMAND ----------

(D_train==X_train).all()
for i in range(D_train.shape[1]):
    if D_train[0, i] != X_train[0,i]:
        print( "D "+str(D_train[0, i]) )
        print( "X "+str(X_train[0, i] ))

        

# COMMAND ----------

X_train

# COMMAND ----------

#
### to be used only if the input are spark dataframes
#
y_val = X_val_spark.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()
y_test = X_test_spark.select(LABEL_COLUMN).toPandas()[LABEL_COLUMN].to_numpy()

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

def get_array_of_probabilities_from_sparkling_water_prediction(predict_sDF):
    p = predict_sDF.select('detailed_prediction').collect()
    probs = list()
    for row in range(len(p)):
        prob = p[row].asDict()['detailed_prediction']['probabilities'][1]
        probs = probs + [prob]
    
    return np.asarray(probs)

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

from pyspark.ml.classification import LinearSVC as svc
from pyspark.ml.classification import DecisionTreeClassifier as dtc
from pyspark.ml.classification import GBTClassifier as gbt
from pyspark.ml.classification import RandomForestClassifier as rfc
from pyspark.ml.classification import LogisticRegression as lr

bool = True

model_class = [lr(maxIter=100,featuresCol='features',labelCol=LABEL_COLUMN),
              gbt(seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              dtc(seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              rfc (numTrees=500,seed=2022,featuresCol='features',labelCol=LABEL_COLUMN), 
              svc(maxIter=100,featuresCol='features',labelCol=LABEL_COLUMN)]
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
    run_name="spark_ml_delta_table_"+modelName
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        model = model_class[i]
        model_fit = model.fit(X_train_spark)

        mlflow.spark.log_model(model_fit, run_name)

        predictions_test = model_fit.transform(X_test_spark)

        if modelName in KNOWN_REGRESSORS_THAT_YIELD_PROBABILITIES:
            predictions_val = model_fit.transform(X_val_spark)
            val_probs  = get_array_of_probs (predictions_val)
            test_probs = get_array_of_probs (predictions_test)
            stats = get_statistics_from_probabilities(val_probs, test_probs, y_val, y_test, mod_name=modelName, average=AVERAGE)
        else:
            y_predict = predictions_test.select('prediction').toPandas()['prediction'].to_numpy()

            stats = get_statistics_from_predict(y_predict, 
                                        y_test, 
                                        str(modelName), 
                                        average=AVERAGE)
        log_stats_in_mlflow(stats)

        if bool == True:
            bool = False
            all_stats = stats
        else:
            all_stats = all_stats.append(stats)
display(all_stats)

# COMMAND ----------

display(all_stats)

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, CrossValidatorModel

run_name="premier_analysis_spark_ml_tunned_LR"

mlflow.end_run()
with mlflow.start_run(
    run_name=run_name,
    experiment_id=experiment_id,
):
    lr = LogisticRegression(featuresCol='features',labelCol='label')

    paramGrid = (ParamGridBuilder()
         .addGrid(lr.regParam, [0.01, 0.5, 2.0])
         .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
         .addGrid(lr.maxIter, [1, 5, 10])
         .build())

    evaluator = BinaryClassificationEvaluator()

    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, parallelism=100)

    cvModel = cv.fit(X_train_spark)

    mlflow.spark.log_model(cvModel.bestModel, run_name)
    mlflow.log_param("elasticNetParam", cvModel.bestModel.getElasticNetParam())
    mlflow.log_param("maxIter", cvModel.bestModel.getMaxIter())
    mlflow.log_param("regParam", cvModel.bestModel.getRegParam())

    
    predictions_test = cvModel.bestModel.transform(X_test_spark)
    predictions_val  = cvModel.bestModel.transform(X_val_spark)
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

X_train_spark.count()

# COMMAND ----------

X_val_spark.count()

# COMMAND ----------

X_test_spark.count()

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
                             growPolicy="lossguide")

hyperSpace = {"eta":       [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3],
             "maxDepth":   [5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
             "ntrees":     [50, 65, 80, 100, 115, 130, 150],
             "regAlpha":   [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.12, 25.6, 51.2, 102.4, 200],
             "regLambda":  [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.12, 25.6, 51.2, 102.4, 200],
             "gamma":      [0, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 4.12, 25.6, 51.2, 102.4, 200],}

grid = H2OGridSearch(hyperParameters=hyperSpace, 
                     parallelism=0,
                     algo=algo, 
                     strategy="Cartesian",
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

def get_best_model(experiment_id, metric = 'metrics.testing_auc'):
    df_runs = mlflow.search_runs(experiment_ids=experiment_id)  # get child experiments under the parent experiment id
    max_run = df_runs[df_runs[metric] == df_runs[metric].max()] # get the run that yield the max metric
    run_id = 'runs:/'+str(max_run['run_id'].values[0])+'/model'        # prepare run id string
    return run_id

# COMMAND ----------

def get_prediction(sDF, experiment_id):
    
    #
    # get the run id of best model under the experiment id
    #
    run_id = get_best_model(experiment_id=experiment_id)
    #
    # load the best model based using the run id 
    #
    model = mlflow.spark.load_model(run_id)
    #
    # serve the model by appying.
    # In this case, no need to transform to a Pandas DF
    # You can provide the predictions directly from a given Spark DF
    #
    predictions = model.transform(sDF)
    
    return predictions

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


