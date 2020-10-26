import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.spatial import distance_matrix 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets.samples_generator import make_blobs 

#vehicle example

pdf = pd.read_csv(r'C:\Users\User\Downloads\cars_clus.csv')
pdf.head()

#clean data to only what we want by dropping the rows that have null value.
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
       'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
       'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
pdf.head(5)



#our feature set:
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

#normalization
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
feature_mtx [0:5]

import scipy
leng = feature_mtx.shape[0]
D = scipy.zeros([leng,leng])
for i in range(leng):
    for j in range(leng):
        D[i,j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])
        
import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(D, 'complete')

#Hierarchical clustering does not require a pre-specified number of clusters. 
#However, in some applications we want a partition of disjoint clusters just as in flat clustering
from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
clusters

#det nb of clusters directly
from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
clusters

#dendrogram
fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
#dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')




########################################
#redo again but using sklearn

dist_matrix = distance_matrix(feature_mtx,feature_mtx) 
print(dist_matrix)

agglom = AgglomerativeClustering(n_clusters = 6, linkage = 'complete')
agglom.fit(feature_mtx)
agglom.labels_

#add a new field to our dataframe to show the cluster of each row
pdf['cluster_'] = agglom.labels_
print(pdf.head())

#plot
import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, c=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
"""
As you can see, we are seeing the distribution of each cluster using the scatter plot, 
but it is not very clear where is the centroid of each cluster. Moreover, there are 2 types of vehicles in our dataset, 
"truck" (value of 1 in the type column) and "car" (value of 1 in the type column). 
So, we use them to distinguish the classes, and summarize the cluster
"""
#we count the number of cases in each group:
pdf.groupby(['cluster_','type'])['cluster_'].count()

#we can look at the characteristics of each cluster:
agg_cars = pdf.groupby(['cluster_','type'])['horsepow','engine_s','mpg','price'].mean()
print(agg_cars)

""" It is obvious that we have 3 main clusters with the majority of vehicles in those.

Cars: (type 0)

Cluster 1: with almost high mpg, and low in horsepower.
Cluster 2: with good mpg and horsepower, but higher price than average.
Cluster 3: with low mpg, high horsepower, highest price.

Trucks: (type 1)

Cluster 1: with almost highest mpg among trucks, and lowest in horsepower and price.
Cluster 2: with almost low mpg and medium horsepower, but higher price than average.
Cluster 3: with good mpg and horsepower, low price.

"""
#Please notice that we did not use type , and price of cars in the clustering process, 
#but Hierarchical clustering could forge the clusters and discriminate them with quite high accuracy.
plt.figure(figsize=(16,10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in subset.index:
        plt.text(subset.loc[i][0]+5, subset.loc[i][2], 'type='+str(int(i)) + ', price='+str(int(subset.loc[i][3]))+'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price*20, c=color, label='cluster'+str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')



















