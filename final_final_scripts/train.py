#!/usr/bin/env python
# Import MNIST data
import csv
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

filename = input('data filename: ')
filename2 = input('new knowledge filename: ')
filename3 = input('old knowledge filename: ')


with open(filename) as f:
    lis=[line.split(',') for line in f] 
    lis1 = [float(x[0].rstrip()) for x in lis[1:len(lis) - 1]]
    lis2 = [float(x[1].rstrip()) for x in lis[1:len(lis) - 1]]
    lis3 = [float(x[2].rstrip()) for x in lis[1:len(lis) - 1]]
    lis4 = [float(x[3].rstrip()) for x in lis[1:len(lis) - 1]]
    lis5 = [float(x[4].rstrip()) for x in lis[1:len(lis) - 1]]
    lis6 = [float(x[5].rstrip()) for x in lis[1:len(lis) - 1]]
    lis7 = [float(x[6].rstrip()) for x in lis[1:len(lis) - 1]]

# Parameters
learning_rate = 0.001
num_steps = 10000
batch_no = len(lis1)
display_step = 10
num_input = 5
num_output = 2

# Network Parameters
n_hidden_1 = 400 # 1st layer number of neurons
n_hidden_2 = 300 # 2nd layer number of neurons

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

w1 = tf.Variable(tf.random_normal([num_input, n_hidden_1]))
w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
w3 = tf.Variable(tf.random_normal([n_hidden_2, num_output]))
b1 = tf.Variable(tf.random_normal([n_hidden_1]))
b2 = tf.Variable(tf.random_normal([n_hidden_2]))
b3 = tf.Variable(tf.random_normal([num_output]))

def neural_net(x):
    layer_1 = tf.nn.relu(tf.add(tf.matmul(x, w1), b1))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, w2), b2))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.tanh(tf.add(tf.matmul(layer_2, w3), b3))
    out_layer = tf.multiply(out_layer, 0.5)
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
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
accuracy_2 = tf.reduce_mean(tf.subtract(logits,Y))
init = tf.global_variables_initializer()

# Build the Estimator
#==============================================================================
# model = tf.estimator.Estimator(model_fn)
# 
# # Train the Model
# model.train(input_fn, steps=num_steps)
# 
# c = np.empty([1,100])
# d = np.empty([1,100])
# 
# for i in range(1):
#     x, y = generate_batch_data(1)
#     x = np.squeeze(x)
#     y = np.squeeze(y)
#     x = x/20
#     c[:,i] = x
#     d[:,i] = y
# 
# # Evaluate the Model
# # Define the input function for evaluating
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={'xval': c}, y=d,
#     batch_size=1, shuffle=False)
# # Use the Estimator 'evaluate' method
# diction = model.evaluate(input_fn)
# for keys,values in diction.items():
#     print(keys)
#     print(values)
#==============================================================================
saver = tf.train.Saver({"w1": w1, "w2": w2, "w3": w3, "b1": b1, "b2": b2, "b3": b3 })
saver1 = tf.train.Saver({"w1": w1, "w2": w2, "w3": w3, "b1": b1, "b2": b2, "b3": b3 })
with tf.Session() as sess:
    sess.run(init)
    counter = 0
    x = 0
    saver1.restore(sess, filename3)
    for step in range(1, numsteps + 1):
    #for step in range(1, 10):
        batch_x = np.empty([batch_no, num_input])
        batch_y = np.empty([batch_no, num_input])
        """counter = counter + 1
        for i in range(batch_no):
            x, y = generate_batch_data(num_input)
            x = np.squeeze(x)
            y = np.squeeze(y)
            x = x/20
            y = y/400
            a[i,:] = x
            b[i,:] = y """
        """if counter == 21:
           counter = 1
           x = x + 1 
        batch_x[0,0] = lis2[x]
        batch_y[0,0] = lis1[x]"""
        """if step <= num_steps:
           batch_x[0,0] = lis2[step - 1]
           batch_y[0,0] = lis1[step - 1]
        else:
           batch_x[0,0] = lis2[step - (num_steps + 1)]
           batch_y[0,0] = lis1[step - (num_steps + 1)]"""
        batch_x = np.array([lis1,
                            lis2,
                            lis5,
                            lis6,
                            lis7])
        batch_x = np.transpose(batch_x)
        batch_y = np.array([lis3,
                            lis4])
        batch_y = np.transpose(batch_y)
        if step == 1:
            print('batch x: ', batch_x)
            print('batch y: ', batch_y)
        sess.run(train_op, feed_dict={X: batch_x, Y:batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
    print("Optimization Finished!")
     
    """c = np.empty([batch_no, num_input])
    d = np.empty([batch_no, num_input])

    for i in range(batch_no):
         x, y = generate_batch_data(num_input)
         x = np.squeeze(x)
         y = np.squeeze(y)
         x = x/20
         y = y/400
         c[i,:] = x
         d[i,:] = y"""
    save_path = saver.save(sess, filename2)
    """batch_c = np.empty([batch_no, num_input])
    batch_d = np.empty([batch_no, num_input])
    batch_c[0,0] = lis2[256]
    batch_d[0,0] = lis1[256]"""
