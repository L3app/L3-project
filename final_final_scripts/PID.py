#!/usr/bin/env python
import csv
import rospy
import math
import numpy
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
from mavros_msgs.msg import OpticalFlowRad
from mavros_msgs.msg import State
from sensor_msgs.msg import Range
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TwistStamped
from mavros_msgs.srv import *

warnings.filterwarnings("ignore", ".*GUI is implemented.*")


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
        # rospy.logwarn("Target velocity is \nx: {} \ny: {} \nz: {}".format(self._targetVelX,self._targetVelY, self._targetVelZ))

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
        # rospy.logwarn("Connected is {}, armed is {}, mode is {} ".format(self._isConnected, self._isArmed, self._mode))

    def armRequest(self):
        rospy.wait_for_service('/mavros/set_mode')
        try:
            modeService = rospy.ServiceProxy('/mavros/set_mode', mavros_msgs.srv.SetMode)
            modeService(custom_mode='OFFBOARD')
        except rospy.ServiceException as e:
            print("Service mode set faild with exception: %s" % e)

    def offboardRequest(self):
        rospy.wait_for_service('/mavros/cmd/arming')
        try:
            arm = rospy.ServiceProxy('/mavros/cmd/arming', mavros_msgs.srv.CommandBool)
            arm(True)
        except rospy.ServiceException as e:
            print("Service arm failed with exception :%s" % e)

    def waitForPilotConnection(self):
        rospy.logwarn("Waiting for pilot connection")
        while not rospy.is_shutdown():
            if self._isConnected:
                rospy.logwarn("Pilot is connected")
                return True
            self._rate.sleep
        rospy.logwarn("ROS shutdown")
        return False

def myhook():
  print("KILL ME NOW!")



def distanceCheck(msg):
    global range1
    # print("d")
    range1 = msg.range


# convert imu reading to body fixed angles

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


# receive time message

def timer(msg):
    global timer1
    # print("t")
    timer1 = msg.header.stamp.secs


# receive velocity message

def velfinder(msg):
    global velx, vely, velz
    # print("v")
    velx = msg.twist.linear.x
    vely = msg.twist.linear.y
    velz = msg.twist.linear.z


def callback(msg):
    global x
    global y
    # print("c")
    x = msg.integrated_x
    y = msg.integrated_y


# receive quaternion angles


def gyrocheck(msg):
    global x1
    global y1
    global z1
    # print("g")
    x2 = msg.orientation.x
    y2 = msg.orientation.y
    z2 = msg.orientation.z
    w = msg.orientation.w
    x1, y1, z1 = quaternion_to_euler_angle(w, x2, y2, z2)


def PosCheck(msg):
    global xpos
    global ypos
    global zpos
    xpos = msg.pose.position.x
    ypos = msg.pose.position.y
    zpos = msg.pose.position.z


# PID function

def PID(y, yd, Ki, Kd, Kp, ui_prev, dh, limit, dt):
    # error
    e = yd - y
    # Integrator
    ui = ui_prev + Ki * e * dt
    # Derivative
    ud = Kd * (dh / dt)
    # constraint on values, resetting previous values

    ui_prev = ui
    u = Kp * (e + ui + ud)
    # print("U: ", u)
    if u > limit:
        u = limit
    if u < -limit:
        u = -limit
    return u, ui_prev


def main():
    # import sensor variables
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
    global xpos, ypos, zpos
    xpos, ypos, zpos = 0, 0, 0

    # Importing PID values from file written by optimizer.py
    with open("test.csv") as h:
        lis = [line.split(',') for line in h]
        # print lis
        lis1 = [float(x[0].rstrip()) for x in lis[1:len(lis)]]
    print(lis1)
    # print lis1[1]
    # print lis1[2]

    yDesiredDistance = 0.0
    xDesiredDistance = 0.0
    zHeight = 0.0
    flightPhase = 0
    finalDistance = 6.0
    yGain = 3  # Alan: the y direction (drift) proportional gain
    xGain = 3
    yDesiredDistance = 0.0  # Alan: The desired y coordinate
    global xDistance
    global yDistance
    xDistance, yDistance = 0.0, 0.0

    rospy.init_node('navigator')
    rate = rospy.Rate(20)
    stateManagerInstance = stateManager(rate)

    # Subscriptions
    rospy.Subscriber("/mavros/state", State, stateManagerInstance.stateUpdate)
    rospy.Subscriber("/mavros/distance_sensor/hrlv_ez4_pub", Range, distanceCheck)
    rospy.Subscriber("/mavros/px4flow/raw/optical_flow_rad", OpticalFlowRad, callback)
    rospy.Subscriber("/mavros/imu/data", Imu, gyrocheck)
    rospy.Subscriber("/mavros/local_position/pose", PoseStamped, PosCheck)
    rospy.Subscriber("/mavros/local_position/odom", Odometry, timer)
    rospy.Subscriber("/mavros/local_position/velocity", TwistStamped, velfinder)

    # Publishers
    velPub = rospy.Publisher("/mavros/setpoint_velocity/cmd_vel", TwistStamped, queue_size=2)
    controller = velControl(velPub)
    stateManagerInstance.waitForPilotConnection()

    # PID hover variables
    ui_z_prev = 0
    e_prev = 0
    u_z = 0.25

    # PID stable x variables
    ui_prev1 = 0
    e_prev1 = 0
    u1 = 0

    # PID stable y variables
    ui_prev2 = 0
    e_prev2 = 0
    u2 = 0

    # PID stable z variables
    ui_prev3 = 0
    e_prev3 = 0
    u3 = 0

    # timer variable
    time = 0.0
    time1 = timer1
    timer2 = timer1

    neu_dict = {'dist': [], 'xvel': [], 'zvel': [], 'pitch': [], 'PIDz': [], 'PIDx': [], 'theta': []}

    switch = 0
    switch1 = 0
    xcontrol = 0
    deltax = 0
    deltaz = 0
    z = 0
    x = 0

    # Birk: Stuff for z-position

    h = 0  # calculated from range1
    altitude = 0  # calculated by integrating u_z
    altitude_prev = 0

    # Stuff for recording variables to plot at th end
    rec_t = list()
    rec_range1 = list()
    rec_u_z = list()

    # Storing instantaneous errors to generate total deviation from target(s) for optimizer.py
    time_err_dict = {'time': [], 'error': []}

    while not rospy.is_shutdown() and stateManagerInstance.getLoopCount() < 1250:
        h_prev = h
        h = range1
        dh = h - h_prev

        time_prev = time
        time = float(rospy.get_time())
        dt = time - time_prev

        print("getLoopCount is ", stateManagerInstance.getLoopCount())
        print("Flight phase is: ", flightPhase)

        altitude_prev = altitude
        altitude = altitude_prev + u_z * dt

        controller.publishTargetPose(stateManagerInstance)
        stateManagerInstance.incrementLoop()
        rate.sleep()  # sleep at the set rate

        # rest on ground phase
        if flightPhase == 0:
            controller.setVel([0, 0, 0], [0, 0, 0])

        if stateManagerInstance.getLoopCount() == 100:
            time_init = float(rospy.get_time())

        # change from state 0 to 1 if count >100
        if stateManagerInstance.getLoopCount() > 100:  # need to send some position data before we can switch to offboard mode otherwise offboard is rejected

            # Printing instance (loop) information
            print("Coordinated (x,y,z): ", [xpos, ypos, range1])
            print("Desired height is ", zHeight)
            # print("The time difference is ", dt)

            # Recording to plot variables
            rec_t.append(time - time_init)
            rec_range1.append(range1)
            rec_u_z.append(u_z)

            # Moving to flight phase 1, take-off
            if flightPhase == 0:
                flightPhase = 1
                print("Moving to flight phase: ", flightPhase)

                stateManagerInstance.offboardRequest()  # request control from external computer
                stateManagerInstance.armRequest()  # arming must take place after offboard is requested
                zHeight = 1.5
                # zHeight = zHeight + 0.02 # Birk: Why add 0.02, Alan?

            elif flightPhase == 1:
                # Using PID to determine z-velocity, u_z
                u_z, ui_z_prev = PID(range1, zHeight, lis1[0], lis1[1], lis1[2], ui_z_prev, dh, 0.5, dt)

                # Send velocities to controller
                controller.setVel([xGain * (xDesiredDistance - xpos), yGain * (yDesiredDistance - ypos), u_z],
                                  [0, 0, 0])

                # change from phase 1 to 2 if at z=1.5m
                if zHeight >= 1.5:  # stateManagerInstance.getLoopCount() > 100:   #need to send some position data before we can switch to offboard mode otherwise offboard is rejected
                    zHeight = 1.5  # Desired height
                    flightPhase = 2
                    print("Moving to flight phase: ", flightPhase)
                    timer2 = timer1


            elif flightPhase == 2:
                controller.setVel(
                    [xGain * (xDesiredDistance - xpos), yGain * (yDesiredDistance - ypos), u_z],
                    [0, 0, 0])
                # change from phase 2 to 3 after two seconds
                if (timer1 - timer2) > 2:
                    flightPhase = 3
                    print("Moving to flight phase: ", flightPhase)
                    # print(xpos)

            # forward flight phase
            elif flightPhase == 3:

                # Using PID to determine z-velocity, u_z
                u_z, ui_z_prev = PID(range1, zHeight, lis1[0], lis1[1], lis1[2], ui_z_prev, dh, 0.5, dt)

                # Calculating x-velocity
                xcontrol = 0.5 - (0.2 / 0.1) * numpy.clip(abs(range1 - zHeight), 0,
                                                          0.1)  # this line is unstable when x0 > 0.4, at least on VM workstation (use player!)

                # Send velocities to controller
                controller.setVel([xcontrol, yGain * (yDesiredDistance - ypos), u_z], [0, 0, 0])

                # change from phase 3 to 4 after reaching finalDistance (=6m likely)
                if xpos >= finalDistance:
                    flightPhase = 4
                    print("Moving to flight phase: ", flightPhase)
                    xDesiredDistance = finalDistance


            # landing A phase
            elif flightPhase == 4:
                zHeight = zHeight * 0.95 - 0.005  # Birk: Are these values from somewhere?

                # Send velocities to controller
                # What is the idea behind the z-velocity here? Could we use u_z=PID here?
                controller.setVel(
                    [xGain * (xDesiredDistance - xpos), yGain * (yDesiredDistance - ypos), 2 * (zHeight - range1)],
                    [0, 0, 0])

                # change from phase 4 to 5 when z = 0.2m
                if zHeight <= 0.2:
                    flightPhase = 5
                    print("Moving to flight phase: ", flightPhase)

            # landing B phase
            elif flightPhase == 5:
                controller.setVel([0, 0, -0.2], [0, 0, 0])

                # stop velocity signal an change to phase 6
                if range1 <= 0.1:
                    flightPhase = 6
                    print("Moving to flight phase: ", flightPhase)
                    timer2 = timer1

                    # wait 2 seconds
            elif flightPhase == 6:
                print("Success!")
                controller.setVel([0, 0, 0], [0, 0, 0])

                if (timer1 - timer2) > 2:
                    flightPhase = 7
                    print("Moving to flight phase: ", flightPhase)

            # shut down unit
            elif flightPhase == 7:
                break
                

                # Recording instantanteous error for optimizer summation
            time_err_dict['time'].append((float(rospy.get_time()) - time_init))
            time_err_dict['error'].append(abs(zHeight - range1))

        with open("test1.csv", "wb") as f:
            writer = csv.writer(f)
            writer.writerow(time_err_dict.keys())
            writer.writerows(zip(*time_err_dict.values()))
    rospy.on_shutdown(myhook)

            # rospy.spin()

            # plt.subplot(2, 1, 1)
            # plt.ion()
            # plt.plot(rec_t, rec_range1)
            # plt.subplot(2, 1, 2)
            # plt.plot(rec_t, rec_u_z)
            # plt.show()


if __name__ == '__main__':
    main()

