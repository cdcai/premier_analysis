# Databricks notebook source
!pip install requests
!pip install tabulate
!pip install future
!pip install h2o_pysparkling_3.2

# COMMAND ----------

from pysparkling import *
hc = H2OContext.getOrCreate()

# COMMAND ----------

import h2o
from h2o.automl import H2OAutoML

# Start the H2O cluster (locally)

# Import a sample binary outcome train/test set into H2O
train = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_train_10k.csv")
test = h2o.import_file("https://s3.amazonaws.com/erin-data/higgs/higgs_test_5k.csv")

# Identify predictors and response
x = train.columns
y = "response"
x.remove(y)

# For binary classification, response should be a factor
train[y] = train[y].asfactor()
test[y] = test[y].asfactor()



# COMMAND ----------

train_sDF = hc.asSparkFrame(train)


# COMMAND ----------

# Run AutoML for 20 base models
aml = H2OAutoML(max_models=2, seed=1)
aml.train(x=x, y=y, training_frame=train)

# View the AutoML Leaderboard
lb = aml.leaderboard
lb.head(rows=lb.nrows)  # Print all rows instead of default (10 rows)

aml.leader

# COMMAND ----------

preds = aml.leader.predict(test)


# COMMAND ----------

preds

# COMMAND ----------


