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

df.show()
df.printSchema()

#checking if the classes are balanced, i.e, if there are more or less the same number of example in each class.
#Let’s find out by a simple aggregation using SQL.
spark.sql('select class,count(*) from df group by class').show()

#or using DataFrame instead of SQL: df.groupBy('class').count().show()

# #Let’s create a bar plot from this data. We’re using the pixidust library, because of its simplicity.
# import pixiedust
# from pyspark.sql.functions import col
# counts = df.groupBy('class').count().orderBy('count')
# display(counts)


from pyspark.sql.functions import col, min, max, mean, stddev

df \
    .groupBy('class') \
    .count() \
    .select([ 
        min(col("count")).alias('min'), 
        max(col("count")).alias('max'), 
        mean(col("count")).alias('mean'), 
        stddev(col("count")).alias('stddev') 
    ]) \
    .select([
        col('*'),
        (col("max") / col("min")).alias('minmaxratio')
    ]) \
    .show()

#Imbalanced classes can cause pain in machine learning. Therefore let’s rebalance. In the flowing we limit the number 
#of elements per class to the amount of the least represented class. This is called undersampling. 
from pyspark.sql.functions import min

# create a lot of distinct classes from the dataset
classes = [row[0] for row in df.select('class').distinct().collect()]

# compute the number of elements of the smallest class in order to limit the number of samples per calss
min = df.groupBy('class').count().select(min('count')).first()[0]

# define the result dataframe variable
df_balanced = None

# iterate over distinct classes
for cls in classes:
    
    # only select examples for the specific class within this iteration
    # shuffle the order of the elements (by setting fraction to 1.0 sample works like shuffle)
    # return only the first n samples
    df_temp = df \
        .filter("class = '"+cls+"'") \
        .sample(False, 1.0) \
        .limit(min)
    
    # on first iteration, assing df_temp to empty df_balanced
    if df_balanced == None:    
        df_balanced = df_temp
    # afterwards, append vertically
    else:
        df_balanced=df_balanced.union(df_temp)

df_balanced.groupBy('class').count().orderBy('count').show()
