import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalize this to improve optimization performance
x_train, x_test = x_train / 255.0, x_test / 255.0

"""
Number representation:    5
One-hot encoding:        [5]   [4]    [3]    [2]    [1]    [0]  
Array/vector:             1     0      0      0      0      0 
"""
print("categorical labels")
print(y_train[0:5])

# make labels one hot encoded
y_train = tf.one_hot(y_train, 10)
y_test = tf.one_hot(y_test, 10)

print("one hot encoded labels")
print(y_train[0:5])

print("number of training examples:" , x_train.shape[0])
print("number of test examples:" , x_test.shape[0])

#this allows you to define batch sizes as part of the dataset
#this allows you to iterate through subsets (batches) of the data during training
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(50)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(50)

#To make the input useful to us, we need the 2D images (28x28) to be arranged in a 1D vector using a consistent strategy
# showing an example of the Flatten class and operation
from tensorflow.keras.layers import Flatten
flatten = Flatten(dtype='float32')

"original data shape"
print(x_train.shape)

"flattened shape"
print(flatten(x_train).shape)

#Assigning bias and weights to null tensors
# Weight tensor
W = tf.Variable(tf.zeros([784, 10], tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10], tf.float32))


#adding weights and biases to inputs
def forward(x):
    return tf.matmul(x,W) + b #matrix multiplication of x and W as well as adding biases b

#Softmax Regression

#Softmax sample example
# a sample softmax calculation on an input vector
vector = [1, 0.2, 8]
softmax = tf.nn.softmax(vector)
print("softmax calculation")
print(softmax.numpy())
print("verifying normalization")
print(tf.reduce_sum(softmax))
print("finding vector with largest value (label assignment)")
print("category", tf.argmax(softmax).numpy())

#now we can define the output layer
def activate(x):
    return tf.nn.softmax(forward(x))

#Let's create a Model function for convenience.
def model(x):
    x = flatten(x)
    return activate(x)

#cost fction:
def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))
# addition of 1e-10 to prevent errors in zero calculations

# current loss function for unoptimized model
cross_entropy(y_train, model(x_train)).numpy()

#Type of optimization: Gradient Descent
optimizer = tf.keras.optimizers.SGD(learning_rate=0.25)

#Now we define the training step. This step uses GradientTape to automatically 
#compute deriviatives of the functions we have manually created and applies them using the SGD optimizer.
def train_step(x, y ):
    with tf.GradientTape() as tape:
        #compute loss function
        current_loss = cross_entropy( y, model(x))
        # compute gradient of loss 
        #(This is automatic! Even with specialized funcctions!)
        grads = tape.gradient( current_loss , [W,b] )
        # Apply SGD step to our Variables W and b
        optimizer.apply_gradients( zip( grads , [W,b] ) )     
    return current_loss.numpy()



#We have already divided our full dataset into batches of 50 each using the Datasets API. 
#Now we can iterate through each of those batches to compute a gradient. 
#Once we iterate through all of the batches in the dataset, we complete an epoch, or a full traversal of the dataset.

# zeroing out weights in case you want to run this cell multiple times
# Weight tensor
W = tf.Variable(tf.zeros([784, 10],tf.float32))
# Bias tensor
b = tf.Variable(tf.zeros([10],tf.float32))

loss_values=[]
accuracies = []
epochs = 10

for i in range(epochs):
    j=0
    # each batch has 50 examples
    for x_train_batch, y_train_batch in train_ds:
        j+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if j%500==0: #reporting intermittent batch statistics
            print("epoch ", str(i), "batch", str(j), "loss:", str(current_loss) ) 
    
    # collecting statistics at each epoch...loss function and accuracy
    #  loss function
    current_loss = cross_entropy( y_train, model( x_train )).numpy()
    loss_values.append(current_loss)
    correct_prediction = tf.equal(tf.argmax(model(x_train), axis=1),
                                  tf.argmax(y_train, axis=1))
    #  accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
    accuracies.append(accuracy)
    print("end of epoch ", str(i), "loss", str(current_loss), "accuracy", str(accuracy) ) 


#Test and Plots
    
#Here we compute a summary statistic on the test dataset
correct_prediction_train = tf.equal(tf.argmax(model(x_train), axis=1),tf.argmax(y_train,axis=1))
accuracy_train = tf.reduce_mean(tf.cast(correct_prediction_train, tf.float32)).numpy()

correct_prediction_test = tf.equal(tf.argmax(model(x_test), axis=1),tf.argmax(y_test, axis=1))
accuracy_test = tf.reduce_mean(tf.cast(correct_prediction_test, tf.float32)).numpy()

print("training accuracy", accuracy_train)
print("test accuracy", accuracy_test)

#The next two plots show the performance of the optimization at each epoch.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 6)
#print(loss_values)
plt.figure(0)
plt.plot(loss_values,'-ro')
plt.title("loss per epoch")
plt.xlabel("epoch")
plt.ylabel("loss")

plt.figure(1)
plt.plot(accuracies,'-ro')
plt.title("accuracy per epoch")
plt.xlabel("epoch")
plt.ylabel("accuracy")

###################################################################################

#Part 2

#Improve model accuracy with Simple Deep Neural Network with Dropout (more than 1 hidden layer):

#Create general parameters for the model
width = 28 # width of the image in pixels 
height = 28 # height of the image in pixels
flat = width * height # number of pixels in one image 
class_output = 10 # number of possible classifications for the problem

"""The input image is 28 pixels by 28 pixels, 1 channel (grayscale). 
In this case, the first dimension is the batch number of the image, and can be of any size (so we set it to -1).
 The second and third dimensions are width and height, and the last one is the image channels."""

#Converting images of the data set to tensors
x_image_train = tf.reshape(x_train, [-1,28,28,1])  
x_image_train = tf.cast(x_image_train, 'float32') 

x_image_test = tf.reshape(x_test, [-1,28,28,1]) 
x_image_test = tf.cast(x_image_test, 'float32') 

#creating new dataset with reshaped inputs
train_ds2 = tf.data.Dataset.from_tensor_slices((x_image_train, y_train)).batch(50)
test_ds2 = tf.data.Dataset.from_tensor_slices((x_image_test, y_test)).batch(50)

#reducing dataset size due to complexity otherwise
x_image_train = tf.slice(x_image_train,[0,0,0,0],[10000, 28, 28, 1])
y_train = tf.slice(y_train,[0,0],[10000, 10])


#Convolutional Layer 1
#Defining kernel weight and bias

"""We define a kernel here. The Size of the filter/kernel is 5x5; Input channels is 1 (grayscale); 
and we need 32 different feature maps (here, 32 feature maps means 32 different filters are applied on each image. 
So, the output of convolution layer would be 28x28x32)."""

#In this step, we create a filter / kernel tensor of shape [filter_height, filter_width, in_channels, out_channels]
W_conv1 = tf.Variable(tf.random.truncated_normal([5, 5, 1, 32], stddev=0.1, seed=0))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32])) # need 32 biases for 32 outputs

def convolve1(x):
    return(
        tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

#go through all outputs convolution layer, convolve1, 
#and wherever a negative number occurs, we swap it out for a 0. It is called ReLU activation Function.
def h_conv1(x): return(tf.nn.relu(convolve1(x)))
    
#It partitions the input image into a set of rectangles and, and then find the maximum value for that region.
#he input is a matrix of size 28x28x32, and the output would be a matrix of size 14x14x32.
def conv1(x):
    return tf.nn.max_pool(h_conv1(x), ksize=[1, 2, 2, 1], 
                          strides=[1, 2, 2, 1], padding='SAME')


#Convolutional Layer 2
#Weights and Biases of kernels

"""here, the input image is [14x14x32], the filter is [5x5x32], we use 64 filters of size [5x5x32], 
and the output of the convolutional layer would be 64 convolved image, [14x14x64]."""

W_conv2 = tf.Variable(tf.random.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[64])) #need 64 biases for 64 outputs

def convolve2(x): 
    return( 
    tf.nn.conv2d(conv1(x), W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)
    
def h_conv2(x):  return tf.nn.relu(convolve2(x))

def conv2(x):  
    return(
    tf.nn.max_pool(h_conv2(x), ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME'))
    
#output size is 64 matrix of [7x7]    



#Fully Connected Layer

"""Fully connected layers take the high-level filtered images from previous layer, 
that is all 64 matrices, and convert them to a flat array.
So, each matrix [7x7] will be converted to a matrix of [49x1], 
and then all of the 64 matrix will be connected, which make an array of size [3136x1]."""


#Flattening Second Layer
def layer2_matrix(x): return tf.reshape(conv2(x), [-1, 7 * 7 * 64])

#Weights and Biases between layer 2 and 3
#Composition of the feature map from the last layer (7x7) multiplied by the number of feature maps (64); 
#1024 outputs to Softmax layer

W_fc1 = tf.Variable(tf.random.truncated_normal([7 * 7 * 64, 1024], stddev=0.1, seed = 2))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[1024])) # need 1024 biases for 1024 outputs

#matrix mult:
def fcl(x): return tf.matmul(layer2_matrix(x), W_fc1) + b_fc1

#apply relu
def h_fc1(x): return tf.nn.relu(fcl(x))

#Third layer completed

#Dropout Layer, Optional phase for reducing overfitting
"""At each training step in a mini-batch, some units get switched off randomly 
so that it will not interact with the network. That is, it weights cannot be updated, nor affect the learning of the 
other network nodes. This can be very useful for very large neural networks to prevent overfitting."""

keep_prob=0.5
def layer_drop(x): return tf.nn.dropout(h_fc1(x), keep_prob)

#Weights and Biases
#In last layer, CNN takes the high-level filtered images and translate them into votes using softmax. 
#Input channels: 1024 (neurons from the 3rd Layer); 10 output features

W_fc2 = tf.Variable(tf.random.truncated_normal([1024, 10], stddev=0.1, seed = 2)) #1024 neurons
b_fc2 = tf.Variable(tf.constant(0.1, shape=[10])) # 10 possibilities for digits [0,1,2,3,4,5,6,7,8,9]

#matrix mult
def fc(x): return tf.matmul(layer_drop(x), W_fc2) + b_fc2

#Apply Softmax
#softmax allows us to interpret the outputs of fcl4 as probabilities.
def y_CNN(x): return tf.nn.softmax(fc(x))


#lets start applying fctions and training

#Cost fct:
def cross_entropy(y_label, y_pred):
    return (-tf.reduce_sum(y_label * tf.math.log(y_pred + 1.e-10)))

#optmization:
"""It is obvious that we want minimize the error of our network which is calculated by cross_entropy metric. 
To solve the problem, we have to compute gradients for the loss (which is minimizing the cross-entropy) 
and apply gradients to variables."""

optimizer = tf.keras.optimizers.Adam(1e-4)

variables = [W_conv1, b_conv1, W_conv2, b_conv2, 
             W_fc1, b_fc1, W_fc2, b_fc2, ]

def train_step(x, y):
    with tf.GradientTape() as tape:
        current_loss = cross_entropy( y, y_CNN( x ))
        grads = tape.gradient( current_loss , variables )
        optimizer.apply_gradients( zip( grads , variables ) )
        return current_loss.numpy()

#results = []
#increment = 1000
#for start in range(0,60000,increment):
#    s = tf.slice(x_image_train,[start,0,0,0],[start+increment-1, 28, 28, 1])
#    t = y_CNN(s)
#    results.append(t)

#Define prediction
#Do you want to know how many of the cases in a mini-batch has been classified correctly? lets count them.
correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1), tf.argmax(y_train, axis=1))

#It makes more sense to report accuracy using average of correct cases.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float32'))


#Run session, train

loss_values=[]
accuracies = []
epochs = 1

for i in range(epochs):
    j=0
    # each batch has 50 examples
    for x_train_batch, y_train_batch in train_ds2:
        j+=1
        current_loss = train_step(x_train_batch, y_train_batch)
        if j%50==0: #reporting intermittent batch statistics
            correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
            #  accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
            print("epoch ", str(i), "batch", str(j), "loss:", str(current_loss),
                     "accuracy", str(accuracy)) 
            
    current_loss = cross_entropy( y_train, y_CNN( x_image_train )).numpy()
    loss_values.append(current_loss)
    correct_prediction = tf.equal(tf.argmax(y_CNN(x_image_train), axis=1),
                                  tf.argmax(y_train, axis=1))
    #  accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
    accuracies.append(accuracy)
    print("end of epoch ", str(i), "loss", str(current_loss), "accuracy", str(accuracy) )  

#evaluation metrics
    
j=0
acccuracies=[]
# evaluate accuracy by batch and average...reporting every 100th batch
for x_train_batch, y_train_batch in train_ds2:
        j+=1
        correct_prediction = tf.equal(tf.argmax(y_CNN(x_train_batch), axis=1),
                                  tf.argmax(y_train_batch, axis=1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)).numpy()
        #accuracies.append(accuracy)
        if j%100==0:
            print("batch", str(j), "accuracy", str(accuracy) ) 
import numpy as np
print("accuracy of entire set", str(np.mean(accuracies)))            

#Vizualization
kernels = tf.reshape(tf.transpose(W_conv1, perm=[2, 3, 0,1]),[32, -1])

import utils1
import imp
imp.reload(utils1)
from utils1 import tile_raster_images
import matplotlib.pyplot as plt
from PIL import Image

image = Image.fromarray(tile_raster_images(kernels.numpy(), img_shape=(5, 5) ,tile_shape=(4, 8), tile_spacing=(1, 1)))
### Plot image
plt.rcParams['figure.figsize'] = (18.0, 18.0)
imgplot = plt.imshow(image)
imgplot.set_cmap('gray')  



#first layer output
import numpy as np
plt.rcParams['figure.figsize'] = (5.0, 5.0)
sampleimage = [x_image_train[0]]
plt.imshow(np.reshape(sampleimage,[28,28]), cmap="gray")

#ActivatedUnits = sess.run(convolve1,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
keep_prob=1.0
ActivatedUnits = convolve1(sampleimage)
                           
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 6
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")

#second layer output
    
#ActivatedUnits = sess.run(convolve2,feed_dict={x:np.reshape(sampleimage,[1,784],order='F'),keep_prob:1.0})
ActivatedUnits = convolve2(sampleimage)
filters = ActivatedUnits.shape[3]
plt.figure(1, figsize=(20,20))
n_columns = 8
n_rows = np.math.ceil(filters / n_columns) + 1
for i in range(filters):
    plt.subplot(n_rows, n_columns, i+1)
    plt.title('Filter ' + str(i))
    plt.imshow(ActivatedUnits[0,:,:,i], interpolation="nearest", cmap="gray")



