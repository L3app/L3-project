#!/usr/bin/env python
import csv
import rospy
import math
import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

with open("testall.csv") as f:
    lis=[line.split(',') for line in f] 
lis1 = [float(x[0].rstrip()) for x in lis[1:len(lis) - 1]]
lis2 = [float(x[1].rstrip()) for x in lis[1:len(lis) - 1]]


learning_rate = 0.001
num_steps = len(lis1)
batch_no = 1
display_step = 10
num_input = 1
num_output = num_input

# Network Parameters
n_hidden_1 = 200 # 1st layer number of neurons
n_hidden_2 = 200 # 2nd layer number of neurons
n_hidden_3 = 200

X = tf.placeholder("float", [batch_no, num_input])
Y = tf.placeholder("float", [batch_no, num_output])

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
    layer_2 = tf.sigmoid(tf.add(tf.matmul(layer_1, weights['h2']), biases['b2']))
    layer_3 = tf.sigmoid(tf.add(tf.matmul(layer_2, weights['h3']), biases['b3']))
    out_layer = tf.add(tf.matmul(layer_3, weights['out']), biases['out'])
    return out_layer

logits = neural_net(X)
loss_op = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
accuracy_2 = tf.reduce_mean(tf.subtract(logits,Y))

saver = tf.train.Saver()
saver1 = tf.train.Saver()

with tf.Session() as sess:
  saver.restore(sess, "tmp/model.ckpt")
  counter = 0
  x = 0
  for step in range(1, (500 * num_steps) + 1):
    #for step in range(1, 10):
        batch_x = np.empty([batch_no, num_input])
        batch_y = np.empty([batch_no, num_input])
        batch_x[0,0] = lis2[(step % len(lis2)) - 1]
        batch_y[0,0] = lis1[(step % len(lis1)) - 1]
        sess.run(train_op, feed_dict={X: batch_x, Y:batch_y})
        if step % display_step == 0 or step == 1:
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))
  print("Optimization Finished!")
  save_path = saver1.save(sess, "tmp1/model.ckpt")
  batch_c = np.empty([batch_no, num_input])
  batch_d = np.empty([batch_no, num_input])
  batch_c[0,0] = lis2[256]
  batch_d[0,0] = lis1[256]
  print("Testing Accuracy:", \
        sess.run(accuracy_2, feed_dict={X: batch_c,
                                      Y: batch_d}))
  print batch_c
  print batch_d
  print logits.eval(feed_dict={X: batch_c})
