#!/bin/bash
cd ~/src/Firmware
gnome-terminal -e './gazeboVmWork.sh "make posix_sitl_default gazebo_iris_opt_flow"'
sleep 20s
cd ~/src/Firmware
gnome-terminal -e './rosLaunchDrone.sh'
sleep 20s
exit 0
