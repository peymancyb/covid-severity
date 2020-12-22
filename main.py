from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import when, row_number, monotonically_increasing_id
import pyspark.sql.functions as F
from pyspark.sql import SparkSession
import pandas as pd

spark = SparkSession.builder.appName('covid-19').getOrCreate()
df = spark.read.csv('./Cleaned-Data.csv', header = True, inferSchema = True)
df.printSchema()


res = df.withColumn("idx", monotonically_increasing_id() +
                    1).select("idx", *df.columns)
severity = when(res.Severity_Mild != 0, 2).when(res.Severity_Moderate != 0, 3).when(
    res.Severity_None != 0, 1).when(res.Severity_Severe != 0, 4)
df2 = res.withColumn("severity", severity)
pd.DataFrame(df2.take(5), columns=df2.columns)


columns_to_drop = ['Severity_Mild', 'Severity_Moderate',
                   'Severity_None', 'Severity_Severe']
df3 = df2.drop(*columns_to_drop)

indexer = StringIndexer(inputCol='Country', outputCol='country_cat')
indexed = indexer.fit(df3).transform(df3)


cols = indexed.columns
cols.remove("idx")
cols.remove("severity")
cols.remove("Country")
assembler = VectorAssembler(inputCols=cols, outputCol="features")
df4 = assembler.transform(indexed)
df4 = df4.select(["features", 'severity'])
pd.DataFrame(df4.take(5), columns=df4.columns)

train, test = df4.randomSplit([0.7, 0.3], seed=2018)

lr = LinearRegression(featuresCol='features', labelCol='severity')
model = lr.fit(train)


training_summary = model.summary
print("RMSE training: %f" % training_summary.rootMeanSquaredError)
print("r2 training: %f" % training_summary.r2)


predictions = model.transform(test)
evaluator = RegressionEvaluator(
    predictionCol='prediction', labelCol='severity', metricName='r2')
print("r2 testing = %g" % evaluator.evaluate(predictions))
test_result = model.evaluate(test)
print("RMSE testing = %g" % test_result.rootMeanSquaredError)
