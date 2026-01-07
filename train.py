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

pred_train = pipelinemodel.transform(train)
reval.evaluate(pred_train)

pred_test = pipelinemodel.transform(test)
reval.evaluate(pred_test)