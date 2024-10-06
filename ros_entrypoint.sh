#!/bin/bash
set -e
# Setup ROS2 environment
. "/opt/ros/$ROS_DISTRO/setup.bash"
# Setup micro-ROS environment
. "/microros_ws/install/local_setup.bash"
# Setup ros_ws environment
. "/ros_ws/install/local_setup.bash"
# Execute the passed commands (if any)
exec "$@"