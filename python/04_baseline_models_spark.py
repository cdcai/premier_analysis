# Databricks notebook source
!pip install mlflow --quiet

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

# MAGIC  %md 
# MAGIC  ```
# MAGIC     parser = argparse.ArgumentParser()
# MAGIC     parser.add_argument('--outcome',
# MAGIC                         type=str,
# MAGIC                         default='multi_class',
# MAGIC                         choices=['misa_pt', 'multi_class', 'death', 'icu'],
# MAGIC                         help='which outcome to use as the prediction target')
# MAGIC     parser.add_argument(
# MAGIC         '--day_one',
# MAGIC         help="Use only first inpatient day's worth of features (DAN only)",
# MAGIC         dest='day_one',
# MAGIC         action='store_true')
# MAGIC     parser.add_argument('--all_days',
# MAGIC                         help="Use all features in lookback period (DAN only)",
# MAGIC                         dest='day_one',
# MAGIC                         action='store_false')
# MAGIC     parser.set_defaults(day_one=True)
# MAGIC     parser.add_argument('--use_demog',
# MAGIC                         type=bool,
# MAGIC                         default=True,
# MAGIC                         help='whether to iclude demographics in the features')
# MAGIC     parser.add_argument('--stratify',
# MAGIC                         type=str,
# MAGIC                         default='all',
# MAGIC                         choices=['all', 'death', 'misa_pt', 'icu'],
# MAGIC                         help='which label to use for the train-test split')
# MAGIC     parser.add_argument('--average',
# MAGIC                         type=str,
# MAGIC                         default='weighted',
# MAGIC                         choices=['micro', 'macro', 'weighted'],
# MAGIC                         help='how to average stats for multiclass predictions')
# MAGIC     parser.add_argument('--cohort_prefix',
# MAGIC                         type=str,
# MAGIC                         default='',
# MAGIC                         help='prefix for the cohort csv file, ending with _')
# MAGIC     parser.add_argument('--out_dir',
# MAGIC                         type=str,
# MAGIC                         help="output directory (optional)")
# MAGIC     parser.add_argument("--data_dir",
# MAGIC                         type=str,
# MAGIC                         help="path to the Premier data (optional)")
# MAGIC     parser.add_argument("--test_split",
# MAGIC                         type=float,
# MAGIC                         default=0.2,
# MAGIC                         help="Percentage of total data to use for testing")
# MAGIC     parser.add_argument("--validation_split",
# MAGIC                         type=float,
# MAGIC                         default=0.1,
# MAGIC                         help="Percentage of train data to use for validation")
# MAGIC     parser.add_argument("--rand_seed", type=int, default=2021, help="RNG seed")
# MAGIC     args = parser.parse_args()
# MAGIC 
# MAGIC     # Setting the globals
# MAGIC     OUTCOME = args.outcome
# MAGIC     USE_DEMOG = args.use_demog
# MAGIC     AVERAGE = args.average
# MAGIC     DAY_ONE_ONLY = args.day_one
# MAGIC     TEST_SPLIT = args.test_split
# MAGIC     VAL_SPLIT = args.validation_split
# MAGIC     RAND = args.rand_seed
# MAGIC     CHRT_PRFX = args.cohort_prefix
# MAGIC     STRATIFY = args.stratify
# MAGIC 
# MAGIC     # Setting the directories and importing the data
# MAGIC     pwd = os.path.abspath(os.path.dirname(__file__))
# MAGIC 
# MAGIC     # If no args are passed to overwrite these values, use repo structure to construct
# MAGIC     data_dir = os.path.abspath(os.path.join(pwd, "..", "data", "data", ""))
# MAGIC     output_dir = os.path.abspath(os.path.join(pwd, "..", "output", ""))
# MAGIC 
# MAGIC     if args.data_dir is not None:
# MAGIC         data_dir = os.path.abspath(args.data_dir)
# MAGIC 
# MAGIC     if args.out_dir is not None:
# MAGIC         output_dir = os.path.abspath(args.out_dir)
# MAGIC 
# MAGIC     pkl_dir = os.path.join(output_dir, "pkl", "")
# MAGIC     stats_dir = os.path.join(output_dir, "analysis", "")
# MAGIC     probs_dir = os.path.join(stats_dir, "probs", "")
# MAGIC ```

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

if EXPERIMENTING == True: SAMPLE = 10
else: SAMPLE = X.shape[1]

# COMMAND ----------

def convert_pandas_to_spark_with_vectors(a_dataframe, c_names):
    from pyspark.sql import SparkSession
    from pyspark.ml.feature import VectorAssembler


    inc=20000
    bool = True
    for i in range((a_dataframe.shape[0]//inc)+1):
        if isinstance (a_dataframe,  pd.DataFrame):
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].to_numpy())
        else:
            a_rdd = spark.sparkContext.parallelize(a_dataframe[i*inc:(1+i)*inc].toarray())

        a_df = (a_rdd.map(lambda x: x.tolist()).toDF(c_names)  )
        
        vecAssembler = VectorAssembler(outputCol="features")
        vecAssembler.setInputCols(c_names)
        a_spark_vector = vecAssembler.transform(a_df)
        
        if bool == True:
            spark_df = a_spark_vector
            bool = False
        else:
            spark_df = spark_df.union(a_spark_vector)
    return spark_df


def change_columns_names (X):
    c_names = list()
    for i in range(0, X.shape[1]):
        c_names = c_names + ['c'+str(i)] 
    return c_names

# COMMAND ----------

from pyspark.sql.functions import monotonically_increasing_id

c_names = change_columns_names(X)

X_train_spark = convert_pandas_to_spark_with_vectors(X[train][:SAMPLE], c_names).withColumn("id",monotonically_increasing_id() )
X_val_spark =   convert_pandas_to_spark_with_vectors(X[val][:SAMPLE], c_names).withColumn("id",monotonically_increasing_id() )
X_test_spark =  convert_pandas_to_spark_with_vectors(X[test][:SAMPLE], c_names).withColumn("id",monotonically_increasing_id() )


y_train_spark = spark.createDataFrame(pd.DataFrame(y[train][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )
y_val_spark =   spark.createDataFrame(pd.DataFrame(y[val][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )
y_test_spark =  spark.createDataFrame(pd.DataFrame(y[test][:SAMPLE], columns=['y']).astype(int)).withColumn("id",monotonically_increasing_id() )


# COMMAND ----------

X_y_train_spark = X_train_spark.join(y_train_spark,X_train_spark.id == y_train_spark.id, how='inner')
X_y_val_spark = X_val_spark.join(y_val_spark,X_val_spark.id == y_val_spark.id, how='inner')
X_y_test_spark = X_test_spark.join(y_test_spark,X_test_spark.id == y_test_spark.id, how='inner')

X_y_spark = X_y_train_spark.union(X_y_val_spark).union(X_y_test_spark)
X_y_train_val_spark = X_y_train_spark.union(X_y_val_spark)

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

def get_statistics(val_probs, test_probs, y_val, y_test, mod_name, average=AVERAGE):
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

y_val = y_val_spark.select('y').toPandas()['y'].to_numpy()
y_test = y_test_spark.select('y').toPandas()['y'].to_numpy()

# COMMAND ----------

#import the logistic regression 
from pyspark.ml.classification import LogisticRegression
#
# start a new MLFlow Experiment
#
import mlflow
mlflow.end_run()
mlflow.start_run(experiment_id=experiment_id)
mlflow.autolog()
#
#Apply the logistic regression model
#
lr=LogisticRegression(labelCol='y')
lrModel = lr.fit(X_y_train_spark.select(['features','y']))
lrPredictions_val = lrModel.transform(X_y_val_spark.select(['features','y']))
lrPredictions_test = lrModel.transform(X_y_test_spark.select(['features','y']))
val_probs  = get_array_of_probs (lrPredictions_val)
test_probs = get_array_of_probs (lrPredictions_test)
stats = get_statistics(val_probs, test_probs, y_val, y_test, mod_name='lr', average=AVERAGE)

#
# add parameters and metrics to MLFlow experiment and end experiment
#
for i in stats:
    if not isinstance(stats[i].iloc[0], str):
        mlflow.log_metric("Testing "+i, stats[i].iloc[0])
    
mlflow.log_param("average", AVERAGE)
mlflow.log_param("demographics", USE_DEMOG)
mlflow.log_param("outcome", OUTCOME)
mlflow.log_param("stratify", STRATIFY)

mlflow.end_run()

# COMMAND ----------

#
stats = ta.clf_metrics(y_true, probabilities[1],  mod_name="spark lr",
                               average=AVERAGE)
stats

# COMMAND ----------

stats = ta.clf_metrics(y[test][:SAMPLE],
                               results.select(vector_to_array("probability", "float32").alias("probability")).toPandas().to_numpy(),
                               mod_name="spark lr",
                               average=AVERAGE)

# COMMAND ----------

stats

# COMMAND ----------

lrn_summary = log_reg.summary

# COMMAND ----------

log_reg.predictProbability(X_y_test_spark.features).show()

# COMMAND ----------

# Fitting a logistic regression to the whole dataset
lgr = LogisticRegression(max_iter=5000, multi_class='ovr')
mlflow.sklearn.autolog(log_models=True)
with mlflow.start_run(experiment_id=experiment_id) as run:
    lgr.fit(X, y)
    mlflow.sklearn.log_model(lgr, "lgr")
coef_list = []

# Sorting the coefficients for
for i in range(n_classes):
    if not binary:
        exp_coefs = np.exp(lgr.coef_)[i]
    else:
        exp_coefs = np.exp(lgr.coef_)[0]
        i = 1
    top_coef = np.argsort(exp_coefs)[::-1][0:30]
    top_ftrs = [vocab[code] for code in top_coef]
    top_codes = [all_feats[ftr] for ftr in top_ftrs]

    bottom_coef = np.argsort(exp_coefs)[0:30]
    bottom_ftrs = [vocab[code] for code in bottom_coef]
    bottom_codes = [all_feats[ftr] for ftr in bottom_ftrs]

    codes = top_codes + bottom_codes
    coefs = np.concatenate([exp_coefs[top_coef], exp_coefs[bottom_coef]])
    coef_df = pd.DataFrame([codes, coefs]).transpose()
    coef_df.columns = ['feature', 'aOR']
    coef_df.sort_values('aOR', ascending=False, inplace=True)
    coef_list.append(coef_df)

# Writing the sorted coefficients to Excel
out_name = OUTCOME + '_lgr_'
if DAY_ONE_ONLY:
    out_name += 'd1_'

#with pd.ExcelWriter(stats_dir + out_name + 'coefs.xlsx') as writer:
#    for i, df in enumerate(coef_list):
#        df.to_excel(writer, sheet_name='coef_' + str(i), index=False)

#    writer.save()

# COMMAND ----------

# Loading up some models to try
stats = None
mods = [
    LogisticRegression(max_iter=5000, multi_class='ovr'),
    RandomForestClassifier(n_estimators=500, n_jobs=-1),
    GradientBoostingClassifier(),
    LinearSVC(class_weight='balanced', max_iter=10000)
]
mod_names = ['lgr', 'rf', 'gbc', 'svm']

# Turning the crank like a proper data scientist
for i, mod in enumerate(mods):
    # Fitting the model and setting the name
    
    #
    # add execution parameters to MLFLOW
    #
    
    mlflow.end_run()
    mlflow.start_run(experiment_id=experiment_id)
    mlflow.autolog()
    mlflow.log_param("average", AVERAGE)
    mlflow.log_param("demographics", USE_DEMOG)
    mlflow.log_param("outcome", OUTCOME)
    mlflow.log_param("stratify", STRATIFY)
    #
    #
    #
    
    mod.fit(X[train], y[train])
    mlflow.sklearn.log_model(mod, mod_names[i])
    mod_name = mod_names[i]
    if DAY_ONE_ONLY:
        mod_name += '_d1'

    if 'predict_proba' in dir(mod):
        if binary:
            val_probs = mod.predict_proba(X[val])[:, 1]
            val_gm = ta.grid_metrics(y[val], val_probs)
            cutpoint = val_gm.cutoff.values[np.argmax(val_gm.f1)]
            test_probs = mod.predict_proba(X[test])[:, 1]
            test_preds = ta.threshold(test_probs, cutpoint)
            stats = ta.clf_metrics(y[test],
                                   test_probs,
                                   cutpoint=cutpoint,
                                   mod_name=mod_name,
                                   average=AVERAGE)

            ta.write_preds(output_dir=output_dir + "/",
                           preds=test_preds,
                           outcome=OUTCOME,
                           mod_name=mod_name,
                           test_idx=test,
                           probs=test_probs)
        else:
            cutpoint = None
            test_probs = mod.predict_proba(X[test])
            test_preds = mod.predict(X[test])
            stats = ta.clf_metrics(y[test],
                                   test_probs,
                                   mod_name=mod_name,
                                   average=AVERAGE)
            ta.write_preds(output_dir=output_dir + "/",
                           preds=test_preds,
                           probs=np.max(test_probs, axis=1),
                           outcome=OUTCOME,
                           mod_name=mod_name,
                           test_idx=test)

        # Write out multiclass probs as pkl
        probs_file = mod_name + '_' + OUTCOME + '.pkl'
        prob_out = {'cutpoint': cutpoint, 'probs': test_probs}

        with open(probs_dir + probs_file, 'wb') as f:
            pkl.dump(prob_out, f)

    else:
        test_preds = mod.predict(X[test])
        stats = ta.clf_metrics(y[test],
                               test_preds,
                               mod_name=mod_name,
                               average=AVERAGE)
        ta.write_preds(output_dir=output_dir + "/",
                       preds=test_preds,
                       outcome=OUTCOME,
                       mod_name=mod_name,
                       test_idx=test)

    # Saving the results to disk
    ta.write_stats(stats, OUTCOME, stats_dir=stats_dir)
    #
    #
    # add metrics to MLFLow
    #
    print(stats)
    for i in stats.columns:
        if not isinstance(stats[i].iloc[0], str):
            mlflow.log_metric("Testing "+i, stats[i].iloc[0])
    #
    #
    #
    

# COMMAND ----------

stats_dir

# COMMAND ----------

stats_pd = pd.read_csv('/dbfs/home/tnk6/premier_output/analysis/icu_stats.csv')

# COMMAND ----------

stats_pd

# COMMAND ----------


