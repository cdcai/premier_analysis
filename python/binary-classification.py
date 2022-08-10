# Databricks notebook source
# MAGIC %md
# MAGIC ## Decision Trees
# MAGIC 
# MAGIC You can read more about [Decision Trees](http://spark.apache.org/docs/latest/mllib-decision-tree.html) in the Spark MLLib Programming Guide.
# MAGIC The Decision Trees algorithm is popular because it handles categorical
# MAGIC data and works out of the box with multiclass classification tasks.

# COMMAND ----------

from pyspark.ml.classification import DecisionTreeClassifier

# Create initial Decision Tree Model
dt = DecisionTreeClassifier(labelCol="label", featuresCol="features", maxDepth=3)

# Train model with Training Data
dtModel = dt.fit(trainingData)

# COMMAND ----------

# MAGIC %md
# MAGIC You can extract the number of nodes in the decision tree as well as the
# MAGIC tree depth of the model.

# COMMAND ----------

print("numNodes = ", dtModel.numNodes)
print("depth = ", dtModel.depth)

# COMMAND ----------

display(dtModel)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = dtModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the Decision Tree model with
# MAGIC `BinaryClassificationEvaluator`.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Entropy and the Gini coefficient are the supported measures of impurity for Decision Trees. This is ``Gini`` by default. Changing this value is simple, ``model.setImpurity("Entropy")``.

# COMMAND ----------

dt.getImpurity()

# COMMAND ----------

# MAGIC %md
# MAGIC Now tune the model with using `ParamGridBuilder` and `CrossValidator`.
# MAGIC 
# MAGIC With three values for maxDepth and three values for maxBin, the grid has 4 x 3 = 12 parameter settings for `CrossValidator`. 

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 6, 10])
             .addGrid(dt.maxBins, [20, 40, 80])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=dt, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# Run cross validations
cvModel = cv.fit(trainingData)
# Takes ~5 minutes

# COMMAND ----------

print("numNodes = ", cvModel.bestModel.numNodes)
print("depth = ", cvModel.bestModel.depth)

# COMMAND ----------

# Use test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Random Forest
# MAGIC 
# MAGIC Random Forests uses an ensemble of trees to improve model accuracy.
# MAGIC You can read more about [Random Forest] from the [classification and regression] section of MLlib Programming Guide.
# MAGIC 
# MAGIC [classification and regression]: https://spark.apache.org/docs/latest/ml-classification-regression.html
# MAGIC [Random Forest]: https://spark.apache.org/docs/latest/ml-classification-regression.html#random-forests

# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier

# Create an initial RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="features")

# Train model with Training Data
rfModel = rf.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the Transformer.transform() method.
predictions = rfModel.transform(testData)

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# View model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC Evaluate the Random Forest model with `BinaryClassificationEvaluator`.

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator

# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC Now tune the model with `ParamGridBuilder` and `CrossValidator`.
# MAGIC 
# MAGIC With three values for maxDepth, two values for maxBin, and two values for numTrees,
# MAGIC the grid has 3 x 2 x 2 = 12 parameter settings for `CrossValidator`.

# COMMAND ----------

# Create ParamGrid for Cross Validation
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

paramGrid = (ParamGridBuilder()
             .addGrid(rf.maxDepth, [2, 4, 6])
             .addGrid(rf.maxBins, [20, 60])
             .addGrid(rf.numTrees, [5, 20])
             .build())

# COMMAND ----------

!pip install mlflow

# COMMAND ----------

import mlflow
mlflow.end_run()
mlflow.pyspark.ml.autolog()

with mlflow.start_run(
    experiment_id="1484437891988512",
):
  # Create 5-fold CrossValidator
  cv = CrossValidator(estimator=rf, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

  # Run cross validations.  This can take about 6 minutes since it is training over 20 trees!
  cvModel = cv.fit(trainingData)

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

# cvModel uses the best model found from the Cross Validation
# Evaluate best model
evaluator.evaluate(predictions)

# COMMAND ----------

# View Best model's predictions and probabilities of each prediction class
selected = predictions.select("label", "prediction", "probability", "age", "occupation")
display(selected)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Make Predictions
# MAGIC As Random Forest gives the best areaUnderROC value, use the bestModel obtained from Random Forest for deployment,
# MAGIC and use it to generate predictions on new data.
# MAGIC Instead of new data, this example generates predictions on the entire dataset.

# COMMAND ----------

bestModel = cvModel.bestModel

# COMMAND ----------

# Generate predictions for entire dataset
finalPredictions = bestModel.transform(dataset)

# COMMAND ----------

# Evaluate best model
evaluator.evaluate(finalPredictions)

# COMMAND ----------

# MAGIC %md
# MAGIC This example shows predictions grouped by age and occupation.

# COMMAND ----------

finalPredictions.createOrReplaceTempView("finalPredictions")

# COMMAND ----------

# MAGIC %md
# MAGIC In an operational environment, analysts may use a similar machine learning pipeline to obtain predictions on new data, organize it into a table and use it for analysis or lead targeting.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT occupation, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY occupation, prediction
# MAGIC ORDER BY occupation

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT age, prediction, count(*) AS count
# MAGIC FROM finalPredictions
# MAGIC GROUP BY age, prediction
# MAGIC ORDER BY age
