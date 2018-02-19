#!/usr/bin/env python

import tensorflow as tf
import numpy as np
import argparse
import pprint as pp
import rospy
import math
import sys
import time
import random
import subprocess
from mavros_msgs.msg import OpticalFlowRad 
from mavros_msgs.msg import State  
from sensor_msgs.msg import Range  
from sensor_msgs.msg import Imu  
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty 
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TwistStamped 
from mavros_msgs.srv import *  
from collections import deque

n_hidden_1 = 400
n_hidden_2 = 300
w1 = tf.Variable(tf.random_normal([4, n_hidden_1]))
w2 = tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
w3 = tf.Variable(tf.random_normal([n_hidden_2, 2]))
b1 = tf.Variable(tf.random_normal([n_hidden_1]))
b2 = tf.Variable(tf.random_normal([n_hidden_2]))
b3 = tf.Variable(tf.random_normal([2]))

saver = tf.train.Saver({"w1": w1, "w2": w2, "w3": w3, "b1": b1, "b2": b2, "b3": b3 })
def main():
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, "tmp/model.ckpt")
        x1 = sess.run(w1)
        print('w1: ', x1)
        x1 = sess.run(w2)
        print('w2: ', x1)
        x1 = sess.run(w3)
        print('w3: ', x1)
        x1 = sess.run(b1)
        print('b1: ', x1)
        x1 = sess.run(b2)
        print('b2: ', x1)
        x1 = sess.run(b3)
        print('b3: ', x1)

        saver.restore(sess, "tmpact/model.ckpt")
        x1 = sess.run(w1)
        print('w1: ', x1)
        x1 = sess.run(w2)
        print('w2: ', x1)
        x1 = sess.run(w3)
        print('w3: ', x1)
        x1 = sess.run(b1)
        print('b1: ', x1)
        x1 = sess.run(b2)
        print('b2: ', x1)
        x1 = sess.run(b3)
        print('b3: ', x1)

if __name__ == '__main__':
    main()







    
    





