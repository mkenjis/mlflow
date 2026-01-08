from pyspark.sql import SparkSession

# Build or retrieve an existing SparkSession
spark = SparkSession.builder \
    .master("local[*]") \
    .appName("MyAppName") \
    .getOrCreate()

df = spark.read.option("inferSchema",True).option("header",True).csv("data/Advertising.csv")

df.printSchema()

train, test = df.randomSplit([0.8,0.2],42)

from pyspark.ml.feature import RFormula
rf = RFormula().setFormula("sales ~ .")

from pyspark.ml.regression import LinearRegression

lr = LinearRegression().setFeaturesCol("features").setRegParam(0.5)

from pyspark.ml import Pipeline
pipeline = Pipeline().setStages([rf,lr])
pipelinemodel = pipeline.fit(train)

from pyspark.ml.evaluation import RegressionEvaluator
reval = RegressionEvaluator()

pred_train = pipelinemodel.transform(train)
reval.evaluate(pred_train)

pred_test = pipelinemodel.transform(test)
reval.evaluate(pred_test)

import mlflow
import mlflow.spark

mlflow.set_experiment("Advertising_via_github")
mlflow.set_tracking_uri(uri="http://192.168.0.23:5000/")

with mlflow.start_run(run_name="linear-regression"):
    # Log model
    pipelinemodel = pipeline.fit(train)
    mlflow.spark.log_model(pipelinemodel, "model")
    # Log params: num_trees and max_depth
    mlflow.log_param("iterations", pipelinemodel.stages[-1].getMaxIter())
    mlflow.log_param("regparam", pipelinemodel.stages[-1].getRegParam())
    predDF = pipelinemodel.transform(test)
    reval = RegressionEvaluator().setLabelCol("label").setPredictionCol("prediction")
    # Log metrics: RMSE and R2
    rmse = reval.setMetricName("rmse").evaluate(predDF)
    r2 = reval.setMetricName("r2").evaluate(predDF)
    #r2 = regressionEvaluator.setMetricName("r2").evaluate(predDF)
    mlflow.log_metrics({"rmse": rmse, "r2": r2 })