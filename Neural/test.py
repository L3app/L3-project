#!/usr/bin/env python
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


learning_rate = 0.001
batch_no = 1
display_step = 10
num_input = 1
num_output = num_input

# Network Parameters
n_hidden_1 = 200 # 1st layer number of neurons
n_hidden_2 = 200 # 2nd layer number of neurons
n_hidden_3 = 200

""" def generate_test_point():
    x = random.uniform(-20, 20)

    out = x*x

    return ( np.array([ x ]), np.array([ out ]) )

  # Generate a bunch of data points and then package them up in the array format needed by
  # tensorflow
def generate_batch_data( num ):
     xs = []
     ys = []

     for i in range(num):
       x, y = generate_test_point()

       xs.append( x )
       ys.append( y )

     return (np.array(xs), np.array(ys) )"""



#==============================================================================
# # Define the input function for training
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'xval': a}, y=b,
#     batch_size=500, num_epochs=None, shuffle=True)
#==============================================================================
X = tf.placeholder("float", [batch_no, num_input])
Y = tf.placeholder("float", [batch_no, num_output])

# Define the neural network
#==============================================================================
# def neural_net(x_dict):
#     # TF Estimator input is a dict, in case of multiple inputs
#     x = x_dict['xval']
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.layers.dense(x, n_hidden_1, activation = tf.sigmoid)
#     layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation = tf.sigmoid)
#     # Hidden fully connected layer with 256 neurons
#     layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation = tf.sigmoid)
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.layers.dense(layer_3, 100)
#     return out_layer
#==============================================================================

weights = {
            'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
            'out': tf.Variable(tf.random_normal([n_hidden_3, num_output]))
            }
biases = {
            'b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'b3': tf.Variable(tf.random_normal([n_hidden_3])),
            'out': tf.Variable(tf.random_normal([num_output]))
            }

def neural_net(x):
    layer_1 = tf.sigmoid(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer

# Define the model function (following TF Estimator Template)
#==============================================================================
# def model_fn(features, labels, mode):
# 
#     # Build the neural network
#     logits = neural_net(features)
# 
#     # Predictions
#     #pred_classes = tf.argmax(logits, axis=1)
#     #pred_probas = tf.nn.softmax(logits)
# 
# #    # If prediction mode, early return
# #    if mode == tf.estimator.ModeKeys.PREDICT:
# #        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)
# 
#     # Define loss and optimizer
#     print(logits.get_shape())
#     print(labels.get_shape())
#     loss_op = tf.reduce_mean(tf.square(tf.subtract(logits,labels)))
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
#     train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())
# 
#     # Evaluate the accuracy of the model
#     acc_op = tf.metrics.mean_absolute_error(labels=labels, predictions=logits)
# 
#     # TF Estimators requires to return a EstimatorSpec, that specify
#     # the different ops for training, evaluating, ...
#     estim_specs = tf.estimator.EstimatorSpec(
#       mode=mode,
#       predictions=logits,
#       loss=loss_op,
#       train_op=train_op,
#       eval_metric_ops={'accuracy': acc_op})
# 
#     return estim_specs
#==============================================================================

logits = neural_net(X)
loss_op = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
accuracy_2 = tf.reduce_mean(tf.subtract(logits,Y))


saver = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "tmp1/model.ckpt")
  batch_1 = np.empty([batch_no, num_input])
  batch_1[0,0] = 0.2
  prediction = sess.run(logits, feed_dict={X: batch_1})
  print(prediction)
  print(prediction[0,0])

