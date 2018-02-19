#!/usr/bin/env python
####          WARNING: NOT WELL ORGANIZED HARD TO UNDERSTAND         #####################

#import statements========================================================================
import csv
import rospy
import math
import sys
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
from mavros_msgs.msg import OpticalFlowRad 
from mavros_msgs.msg import State  
from sensor_msgs.msg import Range  
from sensor_msgs.msg import Imu  
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TwistStamped 
from mavros_msgs.srv import *   
#=========================================================================================

#neural network parameters for restoration================================================
learning_rate = 0.001
batch_no = 1
display_step = 10
num_input = 1
num_output = num_input
n_hidden_1 = 200 
n_hidden_2 = 200 
n_hidden_3 = 200
X = tf.placeholder("float", [batch_no, num_input])
Y = tf.placeholder("float", [batch_no, num_output])
#=========================================================================================

#neural network architecture==============================================================
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
#=========================================================================================

#construct neural net=====================================================================
logits = neural_net(X)
loss_op = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate)
train_op = optimizer.minimize(loss_op)
accuracy = tf.reduce_mean(tf.square(tf.subtract(logits,Y)))
accuracy_2 = tf.reduce_mean(tf.subtract(logits,Y))
saver = tf.train.Saver()
#=========================================================================================

#control drone class======================================================================
class velControl:
    def __init__(self, attPub):  
        self._attPub = attPub
        self._setVelMsg = TwistStamped()
        self._targetVelX = 0
        self._targetVelY = 0
        self._targetVelZ = 0
        self._AngVelX = 0
        self._AngVelY = 0
        self._AngVelZ = 0

      
    def setVel(self, coordinates, coordinates1):
        self._targetVelX = float(coordinates[0])
        self._targetVelY = float(coordinates[1])
        self._targetVelZ = float(coordinates[2])
        self._AngVelX = float(coordinates1[0])
        self._AngVelY = float(coordinates1[1])
        self._AngVelZ = float(coordinates1[2])
        rospy.logwarn("Target velocity is \nx: {} \ny: {} \nz: {}".format(self._targetVelX,self._targetVelY, self._targetVelZ))

     
    def publishTargetPose(self, stateManagerInstance):
        self._setVelMsg.header.stamp = rospy.Time.now()    
        self._setVelMsg.header.seq = stateManagerInstance.getLoopCount()
        self._setVelMsg.header.frame_id = 'fcu'
        self._setVelMsg.twist.linear.x = self._targetVelX
        self._setVelMsg.twist.linear.y = self._targetVelY
        self._setVelMsg.twist.linear.z = self._targetVelZ
        self._setVelMsg.twist.angular.x = self._AngVelX
        self._setVelMsg.twist.angular.y = self._AngVelY
        self._setVelMsg.twist.angular.z = self._AngVelZ
        
        self._attPub.publish(self._setVelMsg) 
#=========================================================================================

#retrieve state of drone class============================================================
class stateManager: 
    def __init__(self, rate):
        self._rate = rate
        self._loopCount = 0
        self._isConnected = 0
        self._isArmed = 0
        self._mode = None
     
    def incrementLoop(self):
        self._loopCount = self._loopCount + 1
     	
    def getLoopCount(self):
        return self._loopCount
     
    def stateUpdate(self, msg):
        self._isConnected = msg.connected
        self._isArmed = msg.armed
        self._mode = msg.mode
        rospy.logwarn("Connected is {}, armed is {}, mode is {} ".format(self._isConnected, self._isArmed, self._mode)) 
     
    def armRequest(self):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            modeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode) 
            modeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("Service mode set faild with exception: %s"%e)
     
    def offboardRequest(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool) 
            arm(True)
        except rospy.ServiceException as e:   
           print("Service arm failed with exception :%s"%e)

    def waitForPilotConnection(self):   
        rospy.logwarn("Waiting for pilot connection")
        while not rospy.is_shutdown():  
            if self._isConnected:   
                rospy.logwarn("Pilot is connected")
                return True
            self._rate.sleep
        rospy.logwarn("ROS shutdown")
        return False
#=========================================================================================
 
# functions to process subscriber messages================================================
def distanceCheck(msg):
    global range1 
    print("d")
    range1 = msg.range 
        
def timer(msg):
    global timer1
    #print("t")
    timer1 = msg.header.stamp.secs

def velfinder(msg):
    global velx, vely, velz
    #print("v")
    velx = msg.twist.linear.x
    vely = msg.twist.linear.y
    velz = msg.twist.linear.z

def callback(msg):
    global x
    global y
    #print("c")
    x = msg.integrated_x
    y = msg.integrated_y
 
def gyrocheck(msg):
    global x1
    global y1
    global z1
    #print("g")
    x2 = msg.orientation.x
    y2 = msg.orientation.y
    z2 = msg.orientation.z
    w = msg.orientation.w
    x1, y1, z1 = quaternion_to_euler_angle(w, x2, y2, z2)
#=========================================================================================


#miscellaneous function :)================================================================
 
def quaternion_to_euler_angle(w, x, y, z):
	ysqr = y * y
	
	t0 = +2.0 * (w * x + y * z)
	t1 = +1.0 - 2.0 * (x * x + ysqr)
	X = math.degrees(math.atan2(t0, t1))
	
	t2 = +2.0 * (w * y - z * x)
	t2 = +1.0 if t2 > +1.0 else t2
	t2 = -1.0 if t2 < -1.0 else t2
	Y = math.degrees(math.asin(t2))
	
	t3 = +2.0 * (w * z + x * y)
	t4 = +1.0 - 2.0 * (ysqr + z * z)
	Z = math.degrees(math.atan2(t3, t4))
	
	return X, Y, Z        
 
def PID(y, yd, Ki, Kd, Kp, ui_prev, e_prev, limit):
     # error
     e = yd - y
     # Integrator
     ui = ui_prev + 1.0 / Ki * e
     # Derivative
     ud = 1.0 / Kd * (e - e_prev)	
     #constraint on values, resetting previous values	
     ui = ui/8
     ud = ud/8
     e_prev = e
     ui_prev = ui
     u = Kp * (e + ui + ud)
     print("U: ", u)
     if u > limit:
         u = limit
     if u < -limit:
         u = -limit
     return u, ui_prev, e_prev
#=========================================================================================

#main run function========================================================================
def main():
    print "hello"
#import sensor variables==================================================================
    tol = 0.1
    global range1
    range1 = 0
    global x, y
    x, y = 0, 0
    global x1, y1, z1 
    x1, y1, z1 = 0, 0, 0
    global timer1
    timer1 = 0
    global velx, vely, velz
    velx, vely, velz = 0, 0, 0
#=========================================================================================  

#utility==================================================================================
    rospy.init_node('navigator')   
    rate = rospy.Rate(20) 
    stateManagerInstance = stateManager(rate) 
#=========================================================================================

#Subscriptions============================================================================
    rospy.Subscriber("/mavros/state", State, stateManagerInstance.stateUpdate)  
    rospy.Subscriber("/mavros/distance_sensor/hrlv_ez4_pub", Range, distanceCheck)  
    rospy.Subscriber("/mavros/px4flow/raw/optical_flow_rad", OpticalFlowRad, callback)     
    rospy.Subscriber("/mavros/imu/data", Imu, gyrocheck)
    rospy.Subscriber("/mavros/local_position/odom", Odometry, timer)
    rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, velfinder)
#=========================================================================================

#Publishers===============================================================================
    velPub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=2)
    controller = velControl(velPub) 
    stateManagerInstance.waitForPilotConnection()  
#=========================================================================================

#PID variables============================================================================ 
    u = 0

    ui_prev1 = 0
    e_prev1 = 0
    u1 = 0

    ui_prev3 = 0
    e_prev3 = 0
    u3 = 0
#=========================================================================================
   
#start tensorflow session=================================================================
    with tf.Session() as sess:
       saver.restore(sess, "tmp/model.ckpt")
#=========================================================================================

#=========================================================================================

#start loop, set drone settings===========================================================
       while not rospy.is_shutdown():
           controller.publishTargetPose(stateManagerInstance)
           stateManagerInstance.incrementLoop()
           rate.sleep()   
#=========================================================================================

#start publishing velocities after 100 iterations to account for offboard control=========
           if stateManagerInstance.getLoopCount() > 100:
               if abs(range1) > 1.45 and term == 0:
                   timer2 = timer1
                   term = 1
               elif abs(timer2-timer1) < 5:
                   controller.setVel([0,0,0],[0,0,0]) 
               elif abs(1.5 - range1) < tol and abs(timer2-timer1) > 5:
                   controller.setVel([0.5,u1,u],[0,0,u3])
               else:
                   controller.setVel([0,u1,u],[0,0,u3])
#=========================================================================================

#tensorflow prediction of optimal z velocity==============================================
               batch_1 = np.empty([1, 1])
               batch_1[0,0] = range1
               prediction = sess.run(logits, feed_dict={X: batch_1})
               u = prediction[0,0]
#=========================================================================================
               
#PID predictions==========================================================================
               u1, ui_prev1, e_prev1 = PID(x1, 0, 1, 1, 1, ui_prev1, e_prev1, 0.075)
               u3, ui_prev3, e_prev3 = PID(z1, 0, 1, 1, 1, ui_prev3, e_prev3, 0.1)
#=========================================================================================

#request offboard control, keep control after shutdown request============================       
               stateManagerInstance.offboardRequest()  
               stateManagerInstance.armRequest()  
       rospy.spin()     
#=========================================================================================

