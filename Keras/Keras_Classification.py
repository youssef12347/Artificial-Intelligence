import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical # for transforming ouput into binary array
import matplotlib.pyplot as plt

# import the data
from keras.datasets import mnist

# read the data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train.shape # nb of images in dataset, size of each image

plt.imshow(X_train[0])


#With conventional neural networks, we cannot feed in the image as input as is. 
#So we need to flatten the images into one-dimensional vectors, each of size 1 x (28 x 28) = 1 x 784.

num_pixels = X_train.shape[1] * X_train.shape[2] # find size of one-dimensional vector

X_train = X_train.reshape(X_train.shape[0], num_pixels).astype('float32') # flatten training images
X_test = X_test.reshape(X_test.shape[0], num_pixels).astype('float32') # flatten test images

# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255

# we need to divide our target variable into categories. We use the to_categorical function
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

num_classes = y_test.shape[1]
print(num_classes)



# Classification
# define classification model
def classification_model():
    # create model
    model = Sequential()
    model.add(Dense(num_pixels, activation='relu', input_shape=(num_pixels,)))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(num_classes, activation='softmax')) #softmax only applies to output layer
    
    # compile model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# build the model
model = classification_model()

# fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=2)

# evaluate the model
scores = model.evaluate(X_test, y_test, verbose=0)
print(scores)

#accuracy and error
print('Accuracy: {}% \n Error: {}'.format(scores[1], 1 - scores[1]))        


#ometimes, you cannot afford to retrain your model everytime you want to use it, 
#especially if you are limited on computational resources and training your model can take a long time. 
#Therefore, with the Keras library, you can save your model after training.
model.save('classification_model.h5')

"""if want to use saved model again"""
#from keras.models import load_model
#pretrained_model = load_model('classification_model.h5')















