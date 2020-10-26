import numpy as np
import tensorflow as tf
tf.compat.v1.disable_eager_execution() 
sess = tf.compat.v1.Session()
from tensorflow import keras
from tensorflow.keras import layers
from keras.activations import *

LSTM_CELL_SIZE = 4  # output size (dimension), which is same as hidden size in the cell

lstm_cell = tf.compat.v1.nn.rnn_cell.BasicLSTMCell(LSTM_CELL_SIZE, state_is_tuple=True)
state = (tf.zeros([1,LSTM_CELL_SIZE]),)*2
print(state)

sample_input = tf.constant([[3,2,2,2,2,2]],dtype=tf.float32)
print (sess.run(sample_input))

with tf.compat.v1.variable_scope("LSTM_sample1"):
    output, state_new = lstm_cell(sample_input, state)
sess.run(tf.compat.v1.global_variables_initializer())
print (sess.run(state_new))

print (sess.run(output))

#########Stacked LSTM (2)
#a 2-layer LSTM. In this case, the output of the first layer will become the input of the second.
sess = tf.compat.v1.Session()
input_dim = 6

cells = []

#first layer
LSTM_CELL_SIZE_1 = 4 #4 hidden nodes
cell1 = tf.compat.v1.nn.rnn_cell.LSTMCell(LSTM_CELL_SIZE_1)
cells.append(cell1)

#second layer
LSTM_CELL_SIZE_2 = 5 #5 hidden nodes
cell2 = tf.compat.v1.nn.rnn_cell.LSTMCell(LSTM_CELL_SIZE_2)
cells.append(cell2)

#stack em
stacked_lstm = tf.compat.v1.nn.rnn_cell.MultiRNNCell(cells)

#Now we can create the RNN from stacked_lstm:
# Batch size x time steps x features.
data = tf.compat.v1.placeholder(tf.float32, [None, None, input_dim])
output, state = tf.dynamic_rnn(stacked_lstm, data, dtype=tf.float32) #does not work in tf 2.0 only 1.0

#Lets say the input sequence length is 3, and the dimensionality of the inputs is 6. 
#The input should be a Tensor of shape: [batch_size, max_time, dimension], in our case it would be (2, 3, 6).

sample_input = [[[1,2,3,4,3,2], [1,2,1,1,1,2],[1,2,2,2,2,2]],[[1,2,3,4,3,2],[3,2,2,1,1,2],[0,0,0,0,3,2]]]
sample_input

#check the output:
output


sess.run(tf.compat.v1.global_variables_initializer())
sess.run(output, feed_dict={data: sample_input})









