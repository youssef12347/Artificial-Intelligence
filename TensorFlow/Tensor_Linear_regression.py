import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
import tensorflow as tf
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)

X = np.arange(0.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
a = 1
b = 0

Y= a * X + b 

plt.plot(X, Y) 
plt.ylabel('Dependent Variable')
plt.xlabel('Indepdendent Variable')
plt.show()

df = pd.read_csv(r'C:/Users/User/Downloads/FuelConsumptionCo2.csv')

# take a look at the dataset
df.head()

#Lets say we want to use linear regression to predict Co2Emission of cars based on their engine size
train_x = np.asanyarray(df[['ENGINESIZE']])
train_y = np.asanyarray(df[['CO2EMISSIONS']])

#First, we initialize the variables a and b, with any random guess, and then we define the linear function:
a = tf.Variable(20.0)
b = tf.Variable(30.2)

def h(x):
   y = a*x + b
   return y

#we want to minimize the square of the predicted values minus the target value. 
#So we define the equation to be minimized as loss.
#To find value of our loss, we use tf.reduce_mean(). 
#This function finds the mean of a multidimensional tensor, and the result can have a different dimension.
#basically we want the mean squared error.
def loss_object(y,train_y) :
    return tf.reduce_mean(tf.square(y - train_y))
    # Below is a predefined method offered by TensorFlow to calculate loss function
    #loss_object = tf.keras.losses.MeanSquaredLogarithmicError()
    
#Now we are ready to start training and run the graph. We use GradientTape to calculate gradients:
learning_rate = 0.01
train_data = []
loss_values =[]
# steps of looping through all your data to update the parameters
training_epochs = 200

# train model
for epoch in range(training_epochs):
    with tf.GradientTape() as tape:
        y_predicted = h(train_x)
        loss_value = loss_object(train_y,y_predicted)
        loss_values.append(loss_value)

        # get gradients
        gradients = tape.gradient(loss_value, [b,a])
        
        # compute and adjust weights
        b.assign_sub(gradients[0]*learning_rate)
        a.assign_sub(gradients[1]*learning_rate)
        if epoch % 5 == 0:
            train_data.append([a, b])

plt.plot(loss_values, 'ro')

#Lets visualize how the coefficient and intercept of line has changed to fit the data:

cr, cg, cb = (1.0, 1.0, 0.0)
for f in train_data:
    cb += 1.0 / len(train_data)
    cg -= 1.0 / len(train_data)
    if cb > 1.0: cb = 1.0
    if cg < 0.0: cg = 0.0
    [a, b] = f
    f_y = np.vectorize(lambda x: a*x + b)(train_x)
    line = plt.plot(train_x, f_y)
    plt.setp(line, color=(cr,cg,cb))

plt.plot(train_x, train_y, 'ro')
green_line = mpatches.Patch(color='red', label='Data Points')
plt.legend(handles=[green_line])

plt.show() #answer not shown for some reason



       
    
    
    
    
    
    
    
    
    
    
    
    