from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()
    
rdd = sc.parallelize(range(100))

print(rdd.count())

print(rdd.sum())

# #fct if greater than 50
# def gt50(i):
#     return i > 50

#Now let’s use the lambda notation to define the function.
gt50 = lambda i: i > 50
print(gt50(4))
print(gt50(51))

#let's shuffle our list to make it a bit more interesting
from random import shuffle
l = list(range(100))
shuffle(l)
rdd = sc.parallelize(l)

#Let’s filter values from our list which are equals or less than 50 by applying our “gt50” function to the list 
#using the “filter” function. Note that by calling the “collect” function, all elements are returned to the Apache Spark 
#Driver. This is not a good idea for BigData, please use “.sample(10,0.1).collect()” or “take(n)” instead.
rdd.filter(gt50).collect()
#or this: rdd.filter(lambda i: i > 50).collect() 

#Now we want to compute the sum for elements in that list which are greater than 50 but less than 75.
print(rdd.filter(lambda x: x > 50).filter(lambda x: x < 75).sum())