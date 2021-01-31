import pandas as pd

df = pd.read_csv(r'C:/Users/User/Desktop/Crops.csv')

df = df[df.label.isin(['chickpea', 'maize', 'kidneybeans', 'muskmelon', 'coconut', 'cotton', 'papaya', 'lentil', 'pomegranate', 'banana', 'apple', 'orange', 'mango', 'grapes', 'watermelon' ])]

labelss= ['chickpea', 'maize', 'kidneybeans', 'muskmelon', 'coconut', 'cotton', 'papaya', 'lentil', 'pomegranate', 'banana', 'apple', 'orange', 'mango', 'grapes', 'watermelon' ]

df = df.reset_index().drop('index', 1)

print(df.head())

#Check for duplicates and drop them if any:
duplicated_rows = df[df.duplicated()]
print(duplicated_rows.shape)
df = df.drop_duplicates()

#check for missing values:
print(df.isnull().sum().sum())



#######################################################

# m = df.label.nunique()
# print(m)

X = df.drop('label', 1)
y = df.label

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size = 0.4, random_state=7)


from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(C=0.01, solver = 'liblinear').fit(X_train,y_train)
LR



from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# fit model no training data
model = XGBClassifier()
model.fit(X_train, y_train)




# # save the model to disk
# import joblib
# filename = 'Crop_final.sav'
# joblib.dump(model, open(filename, 'wb'))



# make predictions for test data
y_pred = model.predict(X_test)
yhat_prob = model.predict_proba(X_test)


# evaluate predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))