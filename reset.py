#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  3 13:21:43 2018

@author: johnson
"""

import rospy
from std_srvs.srv import Empty 
# getting empty service message which the reset world service uses
# initialising the service node
rospy.init_node('reset_world')
# making sure the service is running
#the service is /gazebo/reset world
rospy.wait_for_service('/gazebo/reset_world')
# creating connection between service and message
reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
#executing connection 
# now run the script in a .launuch file
reset_world()
# change to the correct directory and run "roslauch reset_world reset.launch"
