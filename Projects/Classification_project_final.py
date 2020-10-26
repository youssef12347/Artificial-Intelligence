import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import pandas as pd
import numpy as np
import matplotlib.ticker as ticker
from sklearn import preprocessing


df = pd.read_csv(r'C:\users\user\downloads\cbb.csv')
df.head()

#Next we'll add a column that will contain "true" if the wins above bubble are over 7 and "false" if not. 
#We'll call this column Win Index or "windex" for short.
df['windex'] = np.where(df.WAB > 7, 'True', 'False')

#Next we'll filter the data set to the teams that made the Sweet Sixteen, the Elite Eight, 
#and the Final Four in the post season. 
#We'll also create a new dataframe that will hold the values with the new column.
df1 = df.loc[df['POSTSEASON'].str.contains('F4|S16|E8', na=False)]
df1.head()

df1['POSTSEASON'].value_counts()


#plot columns

import seaborn as sns

#based on Power Rating
bins = np.linspace(df1.BARTHAG.min(), df1.BARTHAG.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=6)
g.map(plt.hist, 'BARTHAG', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()

#based on Adjusted Offensive Efficiency
bins = np.linspace(df1.ADJOE.min(), df1.ADJOE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJOE', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


#based on Adjusted Defense Efficiency 
bins = np.linspace(df1.ADJDE.min(), df1.ADJDE.max(), 10)
g = sns.FacetGrid(df1, col="windex", hue="POSTSEASON", palette="Set1", col_wrap=2)
g.map(plt.hist, 'ADJDE', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()

#Lets look at the postseason:
df1.groupby(['windex'])['POSTSEASON'].value_counts(normalize=True)

"""13% of teams with 6 or less wins above bubble make it into F4 while 17% of teams with 7 or more do."""


#Lets convert wins above bubble (winindex) under 7 to 0 and over 7 to 1:
df1['windex'].replace(to_replace=['False','True'], value=[0,1],inplace=True)
df1.head()


#Lets defind feature sets, X:
X = df1[['G', 'W', 'ADJOE', 'ADJDE', 'BARTHAG', 'EFG_O', 'EFG_D',
       'TOR', 'TORD', 'ORB', 'DRB', 'FTR', 'FTRD', '2P_O', '2P_D', '3P_O',
       '3P_D', 'ADJ_T', 'WAB', 'SEED', 'windex']]
X[0:5]

#our labels
y = df1['POSTSEASON'].values
print(y[0:5])


#normalize data
X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

#train_test_split
# We split the X into train and test to find the best k
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Validation set:', X_val.shape,  y_val.shape)


###################### Classification time (Youssef)

#KNN
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

k = 5 # start with 5 for now
#Train Model and Predict  
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
yhat = neigh.predict(X_val)
print("Train set Accuracy: ", accuracy_score(y_train, neigh.predict(X_train)))
print("Test set Accuracy: ", accuracy_score(y_val, yhat))

from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

for k in range(1,16):
    neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
    yhat = neigh.predict(X_val)
    print("Test set Accuracy: ", accuracy_score(y_val, yhat))


#Decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

for k in range(1, 16):
    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = k)
    drugTree.fit(X_train,y_train)
    predTree = drugTree.predict(X_val)
    print("DecisionTrees's Accuracy for k = ", k, ":" , metrics.accuracy_score(y_val, predTree))


#SVM
from sklearn import svm
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

clf = svm.SVC(kernel='poly') # or radial basis function : rbf
clf.fit(X_train, y_train)
#prediction
yhat = clf.predict(X_test)
print(yhat [0:5])


#with jaccard accuracy
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))

from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

#Logistic regression
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print('Train set:', X_train.shape,  y_train.shape)
print('Test set:', X_test.shape,  y_test.shape)

from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR

#predict our testset
yhat = LR.predict(X_test)
yhat

#accuracy
from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))




























