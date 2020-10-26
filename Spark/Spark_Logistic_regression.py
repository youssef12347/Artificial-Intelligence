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

#split train test 20-80
splits = df.randomSplit([0.8, 0.2])
df_train = splits[0]
df_test = splits[1]

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import Normalizer


indexer = StringIndexer(inputCol="class", outputCol="label")

vectorAssembler = VectorAssembler(inputCols=["x","y","z"],
                                  outputCol="features")

normalizer = Normalizer(inputCol="features", outputCol="features_norm", p=1.0)

#logistic regression
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline

lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)
pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer,lr])
model = pipeline.fit(df_train)
prediction = model.transform(df_test) #predict on test


#If we look at the schema of the prediction dataframe we see that there is an additional column called prediction 
#which contains the best guess for the class our model predicts.
prediction.printSchema()

#evaluate
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
MulticlassClassificationEvaluator().setMetricName("accuracy").evaluate(prediction) 

############# Now with RandomTreeClassifier instead of logistic regression
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline

rr = RandomForestClassifier().setLabelCol("label").setFeaturesCol("features").setNumTrees(5)
pipeline = Pipeline(stages=[indexer, vectorAssembler, normalizer, rr])
model = pipeline.fit(df_train)
prediction = model.transform(df_test)
