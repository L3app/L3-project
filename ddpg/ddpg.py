#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:05:02 2017
@author: dylan
"""

import tensorflow as tf
import numpy as np
import tflearn
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
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TwistStamped 
from mavros_msgs.srv import *  
from collections import deque


class ReplayBuffer(object):

    def __init__(self, buffer_size, random_seed=123):
        """
        The right side of the deque contains the most recent experiences 
        """
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        if self.count < self.buffer_size: 
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size):
        batch = []

        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])

        return s_batch, a_batch, r_batch, t_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
# ===========================
#   Actor and Critic DNNs
# ===========================

class ActorNetwork(object):

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, batch_size):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.batch_size = batch_size

        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[
            len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # This gradient will be provided by the critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(
            self.scaled_out, self.network_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
            apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(
            self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        net = tflearn.fully_connected(net, 300)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(
            net, self.a_dim, activation='tanh', weights_init=w_init)
        # Scale output to -action_bound to action_bound
        scaled_out = tf.multiply(out, self.action_bound)
        return inputs, out, scaled_out

    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars


class CriticNetwork(object):


    def __init__(self, sess, state_dim, action_dim, learning_rate, tau, gamma, num_actor_vars):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.learning_rate = learning_rate
        self.tau = tau
        self.gamma = gamma

        # Create the critic network
        self.inputs, self.action, self.out = self.create_critic_network()

        self.network_params = tf.trainable_variables()[num_actor_vars:]

        # Target Network
        self.target_inputs, self.target_action, self.target_out = self.create_critic_network()

        self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

        # Op for periodically updating target network with online network
        # weights with regularization
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) \
            + tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]

        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])

        # Define loss and optimization Op
        self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
        self.optimize = tf.train.AdamOptimizer(
            self.learning_rate).minimize(self.loss)

        self.action_grads = tf.gradients(self.out, self.action)

    def create_critic_network(self):
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
        net = tflearn.fully_connected(inputs, 400)
        net = tflearn.layers.normalization.batch_normalization(net)
        net = tflearn.activations.relu(net)

        # Add the action tensor in the 2nd hidden layer
        # Use two temp layers to get the corresponding weights and biases
        t1 = tflearn.fully_connected(net, 300)
        t2 = tflearn.fully_connected(action, 300)

        net = tflearn.activation(
            tf.matmul(net, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')

        # linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net, 1, weights_init=w_init)
        return inputs, action, out

    def train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.out, self.optimize], feed_dict={
            self.inputs: inputs,
            self.action: action,
            self.predicted_q_value: predicted_q_value
        })

    def predict(self, inputs, action):
        return self.sess.run(self.out, feed_dict={
            self.inputs: inputs,
            self.action: action
        })

    def predict_target(self, inputs, action):
        return self.sess.run(self.target_out, feed_dict={
            self.target_inputs: inputs,
            self.target_action: action
        })

    def action_gradients(self, inputs, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.inputs: inputs,
            self.action: actions
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

# ===========================
#   Tensorflow Summary Ops
# ===========================

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars

def log(func):
	print(func.__name__)


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

 
def distanceCheck(msg):
    global range1 
    print("d")
    range1 = msg.range 
        


###################################################################################################

#convert imu reading to body fixed angles
 
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
        

#receive time message
 
def timer(msg):
    global timer1
    #print("t")
    timer1 = msg.header.stamp.secs
    
#receive velocity message
 
def velfinder(msg):
    global velx, vely, velz
    # print("v")
    velx = msg.twist.linear.x
    vely = msg.twist.linear.y
    velz = msg.twist.linear.z
 
def callback(msg):
    global x
    global y
    #print("c")
    x = msg.integrated_x
    y = msg.integrated_y

#receive quaternion angles

 
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

#PID function
 
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




# ===========================
#   Agent Training
# ===========================

def main():
    with tf.Session() as sess:
        
        #import sensor variables
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
        
    
        
    
        
        np.random.seed(int(1234))
        tf.set_random_seed(int(1234))

        state_dim = 4
        action_dim = 2
        action_bound = 0.5

        actor = ActorNetwork(sess, state_dim, action_dim, action_bound,
                             float(0.0001), float(0.001),                                     
                             int(64))

        critic = CriticNetwork(sess, state_dim, action_dim,
                               float(0.0001), float(0.001),
                               float(0.99),
                               actor.get_num_trainable_vars())
        
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))

        summary_ops, summary_vars = build_summaries()

        sess.run(tf.global_variables_initializer())
        writer = tf.summary.FileWriter('./results/tf_ddpg', sess.graph)
    
        # Initialize target network weights
        actor.update_target_network()
        critic.update_target_network()
    
        # Initialize replay memory
        replay_buffer = ReplayBuffer(int(1000000), int(1234))
        while not rospy.is_shutdown():
        
        
            for i in range(int(50000)):
        
                subprocess.call(['./bashopen.sh'])

                rospy.init_node('navigator')   
                rate = rospy.Rate(40) 
                stateManagerInstance = stateManager(rate) 
		
				#Subscription
                rospy.Subscriber("/mavros/state", State, stateManagerInstance.stateUpdate)  
                rospy.Subscriber("/mavros/distance_sensor/hrlv_ez4_pub", Range, distanceCheck)  
                rospy.Subscriber("/mavros/px4flow/raw/optical_flow_rad", OpticalFlowRad, callback)     
                rospy.Subscriber("/mavros/imu/data", Imu, gyrocheck)
                rospy.Subscriber("/mavros/local_position/odom", Odometry, timer)
                rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, velfinder)
		
				#Publishers
                velPub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=2) 
                controller = velControl(velPub) 
                stateManagerInstance.waitForPilotConnection()  
		
				#PID hover variables 
                ui_prev = 0.25
                e_prev = 0
                u = 0.25
		
				#PID stable x variables
                ui_prev1 = 0
                e_prev1 = 0
                u1 = 0
			   
				#PID stable y variables
                ui_prev2 = 0
                e_prev2 = 0
                u2 = 0
		
				#PID stable z variables
                ui_prev3 = 0
                e_prev3 = 0
                u3 = 0
		
                terminal = 0
				
                xdist = 0
				
				#timer variable
                time1 = timer1
                ep_reward = 0
                ep_ave_max_q = 0
                
                for k in range(100):
                    
                    controller.publishTargetPose(stateManagerInstance)
                    stateManagerInstance.incrementLoop()
                    rate.sleep()
                    
                for j in range(int(1000)):
                    
                    controller.publishTargetPose(stateManagerInstance)
                    stateManagerInstance.incrementLoop()
                    rate.sleep()
                    s = [velz,velx,range1,x1]
                    a = actor.predict(np.reshape(s, (1, actor.s_dim))) + actor_noise()
                    #stable x pid
                    u1, ui_prev1, e_prev1 = PID(x1, 0, 1, 1, 1, ui_prev1, e_prev1, 0.075)
                    #stable z pid
                    u3, ui_prev3, e_prev3 = PID(z1, 0, 1, 1, 1, ui_prev3, e_prev3, 0.1)
                    
                    controller.setVel([a[0][0],u1,a[0][1]],[0,0,u3])
                    
                    rospy.sleep(0.025)
                    
        
                    
                    
                    r = velx - abs(1.5 - range1)
                    if timer1 - time1 > 30:
                        terminal = 1 
                    s2 = [velz,velx,range1,x1]
                    replay_buffer.add(np.reshape(s, (actor.s_dim,)), np.reshape(a, (actor.a_dim,)), r,
                                      terminal, np.reshape(s2, (actor.s_dim,)))
        
                    # Keep adding experience to the memory until
                    # there are at least minibatch size samples
                    if replay_buffer.size() > int(64):
                        s_batch, a_batch, r_batch, t_batch, s2_batch = \
                            replay_buffer.sample_batch(int(64))
        
                        # Calculate targets
                        target_q = critic.predict_target(
                            s2_batch, actor.predict_target(s2_batch))
        
                        y_i = []
                        for k in range(int(64)):
                            if t_batch[k]:
                                y_i.append(r_batch[k])
                            else:
                                y_i.append(r_batch[k] + critic.gamma * target_q[k])
        
                        # Update the critic given the targets
                        predicted_q_value, _ = critic.train(
                            s_batch, a_batch, np.reshape(y_i, (int(64), 1)))
        
                        ep_ave_max_q += np.amax(predicted_q_value)
        
                        # Update the actor policy using the sampled gradient
                        a_outs = actor.predict(s_batch)
                        grads = critic.action_gradients(s_batch, a_outs)
                        actor.train(s_batch, grads[0])
        
                        # Update target networks
                        actor.update_target_network()
                        critic.update_target_network()
        
                    ep_reward += r
        
                    if terminal:
        
                        summary_str = sess.run(summary_ops, feed_dict={
                            summary_vars[0]: ep_reward,
                            summary_vars[1]: ep_ave_max_q / float(j)
                        })
        
                        writer.add_summary(summary_str, i)
                        writer.flush()
        
                        print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), \
                                i, (ep_ave_max_q / float(j))))
                        break
                    stateManagerInstance.offboardRequest()  
                    stateManagerInstance.armRequest()
                subprocess.call(['./bashclose.sh'])
        rospy.spin() 
if __name__ == '__main__':
    main()
