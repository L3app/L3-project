#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 23:05:02 2017
@author: dylan
"""

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
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TwistStamped 
from mavros_msgs.srv import *  
from collections import deque

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

    def waitForPilotConnection(self, i):   
        rospy.logwarn("Waiting for pilot connection")
        print('1', i)
        while not rospy.is_shutdown():  
            print('2', i)
            if self._isConnected:
                print('3', i)
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





def main():
    
    for i in range(int(4)):
        #open gazebo
        time.sleep(5)
        if i ==0:
          subprocess.call(['./bashopen.sh'])
        else:
          subprocess.call(['./bashopen_second.sh'])
        
        #setup ROS
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
        stateManagerInstance.waitForPilotConnection(i)  #this part gives an infinite loop as drone not returning as connected

        #wait for 100 iterations to send position data before switching to offboard
        for j in range(int(200)):
            if rospy.is_shutdown():
                while rospy.is_shutdown():
                    rospy.spin()
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()
            if stateManagerInstance.getLoopCount() > 100:
                stateManagerInstance.offboardRequest()  
                stateManagerInstance.armRequest()
    subprocess.call(['./bashclose.sh'])
    time.sleep(5)

if __name__ == '__main__':
    main()
