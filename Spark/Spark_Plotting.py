from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

sc = SparkContext.getOrCreate(SparkConf().setMaster("local[*]"))

spark = SparkSession \
    .builder \
    .getOrCreate()

# create a dataframe out of it
df = spark.read.parquet(r'C:/Users/User/Downloads/washing.parquet')
print(df.count())

#Now we register the data frame in the ApacheSparkSQL catalog so that we can query it using SQL.
df.createOrReplaceTempView("washing")
spark.sql("SELECT * FROM washing").show()

#visualize voltage using a box plot
result = spark.sql("select voltage from washing where voltage is not null")
#unpack it from row and sample it to 10%
result_array = result.rdd.map(lambda row : row.voltage).sample(False,0.1).collect() 

#just print the 1st 15 elements
print(result_array[:15])

#Boxplot
import matplotlib.pyplot as plt
plt.boxplot(result_array)
plt.show()

#Since we are dealing with time series data we want to make use of the time dimension as well.
result = spark.sql("select voltage,ts from washing where voltage is not null order by ts asc")
result_rdd = result.rdd.sample(False,0.1).map(lambda row : (row.ts,row.voltage))
result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()
print(result_array_ts[:15])
print(result_array_voltage[:15])

#scatter plot
plt.plot(result_array_ts,result_array_voltage)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()


#Now lets repeat the previous steps but only show data for hour. 

#Timestamp are the number of millisecons passed since the 1st of Jan. 1970. You can also use an online tool like 
#http://www.epochconverter.com/ to convert these. But for now just an interval of 60 minutes (10006060)=3600000 within 
#the range above (note that we have removed the sample function because the data set is already reduced).

#first find min max times:
spark.sql("select min(ts),max(ts) from washing").show()

#next 
result = spark.sql(
"""
select voltage,ts from washing 
    where voltage is not null and 
    ts > 1547808720911 and
    ts <= 1547810064867+3600000
    order by ts asc
""")
result_rdd = result.rdd.map(lambda row : (row.ts,row.voltage))
result_array_ts = result_rdd.map(lambda ts_voltage: ts_voltage[0]).collect()
result_array_voltage = result_rdd.map(lambda ts_voltage: ts_voltage[1]).collect()
plt.plot(result_array_ts,result_array_voltage)
plt.xlabel("time")
plt.ylabel("voltage")
plt.show()

#As you can see we are not only able to spot the outliers but also see a time pattern of these outliers occuring.

#let's go for three in a so-called 3D scatter plot:
result_df = spark.sql("""
select hardness,temperature,flowrate from washing
    where hardness is not null and 
    temperature is not null and 
    flowrate is not null
""")
result_rdd = result_df.rdd.sample(False,0.1).map(lambda row : (row.hardness,row.temperature,row.flowrate))
result_array_hardness = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[0]).collect()
result_array_temperature = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[1]).collect()
result_array_flowrate = result_rdd.map(lambda hardness_temperature_flowrate: hardness_temperature_flowrate[2]).collect()

#now plot using already made 3D model:
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(result_array_hardness,result_array_temperature,result_array_flowrate, c='r', marker='o')

ax.set_xlabel('hardness')
ax.set_ylabel('temperature')
ax.set_zlabel('flowrate')

plt.show()


#now Hist for hardness:
plt.hist(result_array_hardness)
plt.show()