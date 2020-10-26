import pandas as pd
import numpy as np

concrete_data = pd.read_csv('https://ibm.box.com/shared/static/svl8tu7cmod6tizo6rk0ke4sbuhtpdfx.csv')
print(concrete_data.head())

print(concrete_data.shape)

#check for null values:
concrete_data.describe()
concrete_data.isnull().sum()
#looks very clean

#Slpit into predictors and target:
concrete_data_columns = concrete_data.columns

predictors = concrete_data[concrete_data_columns[concrete_data_columns != 'Strength']] # all columns except Strength
target = concrete_data['Strength'] # Strength column

#normalize data
predictors_norm = (predictors - predictors.mean()) / predictors.std()

n_cols = predictors_norm.shape[1] # number of predictors = nb of columns

#Regression with keras

import keras
from keras.models import Sequential
from keras.layers import Dense

# define regression model
def regression_model():
    # create model
    model = Sequential()
    #create hidden layers
    model.add(Dense(50, activation='relu', input_shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    #create output layer
    model.add(Dense(1))
    
    # compile model
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# build the model
model = regression_model()

# fit the model = train and test model
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)

#youssef to test
predictions = model.predict(predictors_norm[:101])
print(predictions)




