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
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, Normalizer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="class", outputCol="classIndex")
encoder = OneHotEncoder(inputCol="classIndex", outputCol="categoryVec")
vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")
normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

minmaxscaler = MinMaxScaler(inputCol="features_norm", outputCol="features_minmax")

pipeline = Pipeline(stages=[indexer, encoder, vectorAssembler, normalizer,minmaxscaler])
model = pipeline.fit(df)
prediction = model.transform(df)
prediction.show()

"""The difference between a transformer and an estimator is state. A transformer is stateless whereas an estimator 
keeps state. Therefore “VectorAsselmbler” is a transformer since it only need to read row by row. Normalizer, on the 
other hand need to compute statistics on the dataset before, therefore it is an estimator. An estimator has an additional
“fit” function."""
