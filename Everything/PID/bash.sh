#!/bin/bash     
cd ~/src/Firmware
gnome-terminal -e './gazeboVmWork.sh "make posix_sitl_default gazebo_iris_opt_flow"'
sleep 9s
echo "Launch Drone Node"
cd ~/src/Firmware
gnome-terminal -e './rosLaunchDrone.sh'
sleep 5s
echo "Launch Navigator"
cd ~/Desktop/L3-project-master/everything/PID/
gnome-terminal -e 'python PID4.py'
sleep 18s
killall -15 "gazeboVmWork.sh"
killall -15 "roslaunch"
exit 0


