from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

df = spark.read.parquet(r'C:/users/user/Downloads/washing.parquet')
df.createOrReplaceTempView('washing')
df.show()

""" This is the feature transformation part of this exercise. Since our table is mixing schemas from different sensor 
data sources we are creating new features. In other word we use existing columns to calculate new ones. We only use min 
and max for now, but using more advanced aggregations as we've learned in week three may improve the results. 
We are calculating those aggregations over a sliding window "w". This window is defined in the SQL statement and 
basically reads the table by a one by one stride in direction of increasing timestamp. Whenever a row leaves the window 
a new one is included. Therefore this window is called sliding window"""

result = spark.sql("""
SELECT * from (
    SELECT
    min(temperature) over w as min_temperature,
    max(temperature) over w as max_temperature, 
    min(voltage) over w as min_voltage,
    max(voltage) over w as max_voltage,
    min(flowrate) over w as min_flowrate,
    max(flowrate) over w as max_flowrate,
    min(frequency) over w as min_frequency,
    max(frequency) over w as max_frequency,
    min(hardness) over w as min_hardness,
    max(hardness) over w as max_hardness,
    min(speed) over w as min_speed,
    max(speed) over w as max_speed
    FROM washing 
    WINDOW w AS (ORDER BY ts ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING) 
)
WHERE min_temperature is not null 
AND max_temperature is not null
AND min_voltage is not null
AND max_voltage is not null
AND min_flowrate is not null
AND max_flowrate is not null
AND min_frequency is not null
AND max_frequency is not null
AND min_hardness is not null
AND min_speed is not null
AND max_speed is not null   
""")

#Since this table contains null values also our window might contain them. 
#In case for a certain feature all values in that window are null we obtain also null. 
#As we can see here (in my dataset) this is the case for 7 rows.
print(df.count()-result.count())

from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler


#Lets define a vector transformation helper class which takes all our input features (result.columns) and created one 
#additional column called "features" which contains all our input features as one single column wrapped in "DenseVector"
#objects.
assembler = VectorAssembler(inputCols=result.columns, outputCol="features")

#Now we actually transform the data, note that this is highly optimized code and runs really fast in contrast 
#if we had implemented it.
features = assembler.transform(result)

#Let's have a look at how this new additional column "features" looks like:
features.rdd.map(lambda r : r.features).take(10)

#Since the source data set has been prepared as a list of DenseVectors we can now apply PCA. 
#Note that the first line again only prepares the algorithm by finding the transformation matrices (fit method)

pca = PCA(k=3, inputCol="features", outputCol="pcaFeatures")
model = pca.fit(features)

#Now we can actually transform the data. Let's have a look at the first 20 rows
result_pca = model.transform(features).select("pcaFeatures")
result_pca.show(truncate=False)

#So we obtained three completely new columns which we can plot now. Run a final check if the number of rows is the same.
print(result_pca.count())

#Plot:
rdd = result_pca.rdd.sample(False,0.8)
x = rdd.map(lambda a : a.pcaFeatures).map(lambda a : a[0]).collect() #1st dimension
y = rdd.map(lambda a : a.pcaFeatures).map(lambda a : a[1]).collect() #2nd dimension
z = rdd.map(lambda a : a.pcaFeatures).map(lambda a : a[2]).collect() #3rd dimension

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z, c='r', marker='o')

ax.set_xlabel('dimension1')
ax.set_ylabel('dimension2')
ax.set_zlabel('dimension3')

plt.show()