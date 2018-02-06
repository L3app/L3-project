#!/usr/bin/env python

#import statements:
import rospy
import math
import sys
import time
from mavros_msgs.msg import OpticalFlowRad #import optical flow message structure
from mavros_msgs.msg import State  #import state message structure
from sensor_msgs.msg import Range  #import range message structure
#rostopic info /mavros/local_position/pose = geometry_msgs/PoseStamped

from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose #import position message structures
from geometry_msgs.msg import TwistStamped #used to set velocity messages
from mavros_msgs.srv import *   #import for arm and flight mode setting






class velControl:
    def __init__(self, attPub):  #attPub = attitude publisher
        self._attPub = attPub
        self._setVelMsg = TwistStamped()
        self._targetVelX = 0
        self._targetVelY = 0
        self._targetVelZ = 0

    
    def setVel(self, coordinates):
        self._targetVelX = float(coordinates[0])
        self._targetVelY = float(coordinates[1])
        self._targetVelZ = float(coordinates[2])
        #rospy.logwarn("Target velocity is \nx: {} \ny: {} \nz: {}".format(self._targetVelX,self._targetVelY, self._targetVelZ)) #a bit of spouting


    def publishTargetPose(self, stateManagerInstance):
        self._setVelMsg.header.stamp = rospy.Time.now()    #construct message to publish with time, loop count and id
        self._setVelMsg.header.seq = stateManagerInstance.getLoopCount()
        self._setVelMsg.header.frame_id = 'fcu'

        self._setVelMsg.twist.linear.x = self._targetVelX
        self._setVelMsg.twist.linear.y = self._targetVelY
        self._setVelMsg.twist.linear.z = self._targetVelZ
        
        self._attPub.publish(self._setVelMsg) 
        
        
        
        
        
class stateManager: #class for monitoring and changing state of the controller
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
        rospy.logwarn("Connected is {}, armed is {}, mode is {} ".format(self._isConnected, self._isArmed, self._mode)) #some status info

    def armRequest(self):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            modeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode) #get mode service and set to offboard control
            modeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("Service mode set faild with exception: %s"%e)
    
    def offboardRequest(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool) #get arm command service and arm
            arm(True)
        except rospy.ServiceException as e:   #except if failed
            print("Service arm failed with exception :%s"%e)


    def waitForPilotConnection(self):   #wait for connection to flight controller
        rospy.logwarn("Waiting for pilot connection")
        while not rospy.is_shutdown():  #while not shutting down
            if self._isConnected:   #if state isConnected is true
                rospy.logwarn("Pilot is connected")
                return True
            self._rate.sleep
        rospy.logwarn("ROS shutdown")
        return False

def heightCheck(msg):
    global range    #import global range
    global recordedHeight
    #print(msg.range) #for debugging
    range = msg.range #set range = recieved range
    if type(msg.range) == float:
		recordedHeight = msg.range

def opticalCheck(msg):
    global xReceivedIntegratedFlow
    global yReceivedIntegratedFlow
    xReceivedIntegratedFlow = msg.integrated_x
    yReceivedIntegratedFlow = msg.integrated_y

def poseCheck(msg):
    global xDistance
    global yDistance
    xDistance = msg.pose.position.x
    yDistance = msg.pose.position.y

def main():
    rospy.init_node('navigator')   # make ros node
    


    rate = rospy.Rate(10) # rate will update publisher at 20hz, higher than the 2hz minimum before tieouts occur 10 RUNS BETTER
    stateManagerInstance = stateManager(rate) #create new statemanager

    #Subscriptions
    rospy.Subscriber("/mavros/state", State, stateManagerInstance.stateUpdate)  #get autopilot state including arm state, connection status and mode
    rospy.Subscriber("/mavros/distance_sensor/hrlv_ez4_pub", Range, heightCheck)  #get current distance from ground 
    rospy.Subscriber("/mavros/px4flow/raw/optical_flow_rad", OpticalFlowRad, opticalCheck)  #subscribe to position messages
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, poseCheck)
    global range #import global range variable
    global recordedHeight
    global xReceivedIntegratedFlow
    global yReceivedIntegratedFlow
    global xDistance
    global yDistance
    
    flightPhase = 0
    finalDistance = 6


    #Publishers
    velPub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=2) ###Change to atti


    controller = velControl(velPub) #create new controller class and pass in publisher and state manager
    stateManagerInstance.waitForPilotConnection()   #wait for connection to flight controller
    recordedHeight = 0.0
    xReceivedIntegratedFlow = 0.0
    yReceivedIntegratedFlow = 0.0
    yDesiredDistance = 0.0
    xDesiredDistance = 0.0
    zHeight = 0.0

    while not rospy.is_shutdown():
        if flightPhase == 0:
            controller.setVel([0,0,0])
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()    #sleep at the set rate
        if flightPhase == 1:
            zHeight = zHeight + 0.02
            controller.setVel([2*(xDesiredDistance - xDistance),(yDesiredDistance - yDistance),zHeight-recordedHeight])
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()    #sleep at the set rate
        if flightPhase == 2:
            xDesiredDistance = xDesiredDistance + 0.05
            #print(xDesiredDistance)
            controller.setVel([2*(xDesiredDistance - xDistance),(yDesiredDistance - yDistance),2*(zHeight-recordedHeight)])
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()    #sleep at the set rate    
            #print(xDistance)    
        if flightPhase == 3:
            #xDesiredDistance = xDesiredDistance + 0.002
            #print(xDesiredDistance)
            zHeight = zHeight - 0.005 - zHeight*0.05
            controller.setVel([2*(xDesiredDistance - xDistance),(yDesiredDistance - yDistance),2*(zHeight-recordedHeight)])
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()    #sleep at the set rate      
            
        if flightPhase == 4:
            controller.setVel([0,0,-0.1])
            controller.publishTargetPose(stateManagerInstance)
            stateManagerInstance.incrementLoop()
            rate.sleep()    #sleep at the set rate  
        
        if stateManagerInstance.getLoopCount() > 30:   #need to send some position data before we can switch to offboard mode otherwise offboard is rejected
            if flightPhase == 0:
                flightPhase = 1
                #zHeight = 1.5
            stateManagerInstance.offboardRequest()  #request control from external computer
            stateManagerInstance.armRequest()   #arming must take place after offboard is requested
        if zHeight >= 1.5: #stateManagerInstance.getLoopCount() > 100:   #need to send some position data before we can switch to offboard mode otherwise offboard is rejected
            if flightPhase == 1:
                zHeight = 1.5
                flightPhase = 2 
        if flightPhase == 2:
            if xDesiredDistance >= finalDistance:
                flightPhase = 3
                xDesiredDistance = finalDistance
        if flightPhase == 3:
            if zHeight <= 0.1:
                flightPhase = 4
        
    rospy.spin()    #keeps python from exiting until this node is stopped


if __name__ == '__main__':
    main()



