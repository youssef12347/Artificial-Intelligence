from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()
    
# create a dataframe out of it
df = spark.read.parquet(r'C:/Users/User/Downloads/hmp.parquet')

# register a corresponding query table
df.createOrReplaceTempView('df')

#Given below is the feature engineering pipeline from the lecture.
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

#now lets create a pipeline for Kmeans:
    
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

kmeans = KMeans(featuresCol="features").setK(14).setSeed(1) #we now that we have 14 clusters already.
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df)
predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette)) #The closer Silhouette gets to 1, the better.


# #now same thing but using normalized features, change pipeline accordingly:
# kmeans = KMeans(featuresCol="features_norm").setK(14).setSeed(1)
# pipeline = Pipeline(stages=[vectorAssembler, normalizer, kmeans])
# model = pipeline.fit(df)

# predictions = model.transform(df)

# evaluator = ClusteringEvaluator()

# silhouette = evaluator.evaluate(predictions)
# print("Silhouette with squared euclidean distance = " + str(silhouette))

#Sometimes, inflating the dataset helps, here we multiply x by 10, let’s see if the performance inceases:
from pyspark.sql.functions import col
df_denormalized = df.select([col('*'),(col('x')*10)]).drop('x').withColumnRenamed('(x * 10)','x')

df_denormalized.show()

#check accuracy:
kmeans = KMeans(featuresCol="features").setK(14).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler, kmeans])
model = pipeline.fit(df_denormalized)
predictions = model.transform(df_denormalized)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))

#Now Use GaussianMixture instead of Kmeans
from pyspark.ml.clustering import GaussianMixture

gmm = GaussianMixture().setK(2).setSeed(1)
pipeline = Pipeline(stages=[vectorAssembler,gmm])

model = pipeline.fit(df)

predictions = model.transform(df)

evaluator = ClusteringEvaluator()

silhouette = evaluator.evaluate(predictions)
print("Silhouette with squared euclidean distance = " + str(silhouette))


