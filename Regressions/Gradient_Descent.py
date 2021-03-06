import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

data = pd.read_csv("/content/drive/My Drive/CCPP.csv")

data.head()

X = data.drop(columns="PE")
y = data.PE
X

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
X

def hypothesis(theta, X):
    return X.dot(theta)

def cost_function(X, Y, theta):
 m = len(Y)
 J = np.sum((hypothesis(theta, X) - Y) ** 2)/(2 * m)
 return J

def batch_gradient_descent(X, Y, theta, alpha, iter):
 cost_history = [0] * iter
 m = len(Y)
 
 for i in range(iter):
  # Hypothesis function
  h = hypothesis(theta, X)
  # Our Loss
  loss = h - Y
  # Gradient
  gradient = X.T.dot(loss) / m
  # Update theta
  theta = theta - alpha * gradient
  # Update Cost
  cost = cost_function(X, Y, theta)
  cost_history[i] = cost
 
 return theta, cost_history

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.3,  random_state = 42)
X_train = np.c_[np.ones(len(X_train),dtype='int64'),X_train]
X_test = np.c_[np.ones(len(X_test),dtype='int64'),X_test]

# Initial Coefficients
theta = np.zeros(X_train.shape[1])
# small learning rate
alpha = 0.005
# large number of iterations
iter_ = 4000
newtheta, cost_history = batch_gradient_descent(X_train, y_train, theta, alpha, iter_)

newtheta

plt.plot(cost_history)

def prediction(x_test, newtheta):
  return x_test.dot(newtheta)

y_pred = prediction(X_test,newtheta)
y_pred

def r2(y_pred,y):
 sst = np.sum((y-y.mean())**2)
 ssr = np.sum((y_pred-y)**2)
 r2 = 1-(ssr/sst)
 return(r2)

r2(y_pred,y_test)

optimal_iterations = prediction(X_test[3],newtheta)
optimal_iterations

"""'Therefore, the optimal number of iterations for sufficient convergence is 439 iterations.'"""

