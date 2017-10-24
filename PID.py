#!/usr/bin/env python
import rospy
import math
import sys
import time
from mavros_msgs.msg import OpticalFlowRad 
from mavros_msgs.msg import State  
from sensor_msgs.msg import Range  
from sensor_msgs.msg import Imu  
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TwistStamped 
from mavros_msgs.srv import *   






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

    def setAngVel(self, coordinates):
        self._AngVelX = float(coordinates[0])
        self._AngVelY = float(coordinates[1])
        self._AngVelZ = float(coordinates[2])
        rospy.logwarn("Target velocity is \nx: {} \ny: {} \nz: {}".format(self._AngVelX,self._AngVelY, self._AngVelZ))
    
    def setVel(self, coordinates):
        self._targetVelX = float(coordinates[0])
        self._targetVelY = float(coordinates[1])
        self._targetVelZ = float(coordinates[2])
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
        #rospy.logwarn("Connected is {}, armed is {}, mode is {} ".format(self._isConnected, self._isArmed, self._mode)) 

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
        #rospy.logwarn("Waiting for pilot connection")
        while not rospy.is_shutdown():  
            if self._isConnected:   
                #rospy.logwarn("Pilot is connected")
                return True
            self._rate.sleep
        #rospy.logwarn("ROS shutdown")
        return False


def distanceCheck(msg):
    global range1 
    range1 = msg.range 

def timer(msg):
    global timer1
    timer1 = msg.header.stamp.secs
    
def velfinder(msg):
    global velx, vely, velz
    velx = msg.twist.linear.x
    vely = msg.twist.linear.y
    velz = msg.twist.linear.z

def callback(msg):
    global x
    global y
    x = msg.integrated_x
    y = msg.integrated_y

def gyrocheck(msg):
    global x1
    global y1
    global z1
    x2 = msg.orientation.x
    y2 = msg.orientation.y
    z2 = msg.orientation.z
    w = msg.orientation.w
    x1, y1, z1 = quaternion_to_euler_angle(w, x2, y2, z2)

def PID(y, yc, Ti, Td, Kp, ui_prev, e_prev):
     # Error between the desired and actual output
     e = yc - y
     # Integration Input
     ui = ui_prev + 1.0 / Ti * e
     # Derivation Input
     ud = 1.0 / Td * (e - e_prev)
     # Adjust previous values		
     e_prev = e
     if ui > 1:
         u = 1
     if ui < -1:
         u = -1
     ui_prev = ui
     # Calculate input for the system
     u = Kp * (e + ui + ud)
     if u > 1:
         u = 1
     if u < -1:
         u = -1
     return u, ui_prev, e_prev

def main():
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

    rospy.init_node('navigator')   
    rate = rospy.Rate(20) 
    stateManagerInstance = stateManager(rate) 

    #Subscriptions
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
    ui_prev = 0.5
    e_prev = 0
    u = 0.5

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

    #timer variable
    time1 = timer1

    while not rospy.is_shutdown():

        #hover pid
        controller.setVel([0,0,u])
        u, ui_prev, e_prev = PID(range1, 2, 1, 1, 1, ui_prev, e_prev)  
        
        """while timer1 - time1 < 5:
            print(timer1 - time1)
            controller.setVel([0,0,0.5])"""


        #stable x pid
        controller.setAngVel([u1,0,0])
        u1, ui_prev1, e_prev1 = PID(x1, 0, 1, 1, 1, ui_prev1, e_prev1)

        #stable y pid
        controller.setAngVel([0,u2,0])
        u2, ui_prev2, e_prev2 = PID(y1, 0, 1, 1, 1, ui_prev2, e_prev2)
 
        #stable z pid
        controller.setAngVel([0,0,u3])
        u3, ui_prev3, e_prev3 = PID(z1, 0, 1, 1, 1, ui_prev3, e_prev3)
        
        controller.publishTargetPose(stateManagerInstance)
        stateManagerInstance.incrementLoop()
        rate.sleep()   
 
        print("loop: " ,stateManagerInstance.getLoopCount(), " distance: ", range1, " u input: ", u, " zvel: ", velz, " angx: ", x1, " angvelx: ", u1, " angy: ", y1, " angvely: ", u2, " angx: ", z1, " angvelx: ", u3) 
        if stateManagerInstance.getLoopCount() > 100:   
            stateManagerInstance.offboardRequest()  
            stateManagerInstance.armRequest()  

    rospy.spin()  


if __name__ == '__main__':
    main()



