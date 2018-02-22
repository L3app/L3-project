#!/bin/bash     
cd ~/src/Firmware
gnome-terminal -e './gazeboVmWork.sh "make posix_sitl_default gazebo_iris_opt_flow"'
sleep 8s
echo "Launch Drone Node"
cd ~/src/Firmware
gnome-terminal -e './rosLaunchDrone.sh'
sleep 6s
echo "Launch Navigator"
cd ~/Desktop/L3-project-master/final_final_scripts/
gnome-terminal -e 'python PID.py'
sleep 40s
killall -15 "gazeboVmWork.sh"
killall -15 "roslaunch"
exit 0


