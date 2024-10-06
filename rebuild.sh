#!/bin/bash
set -e
# Save the current directory
ORIGINAL_DIR=$(pwd)
# Setup ROS2 environment
. "/opt/ros/$ROS_DISTRO/setup.bash"
# Setup micro-ROS environment
. "/microros_ws/install/local_setup.bash"
# Navigate to the ros_ws directory and run colcon build
cd /ros_ws
colcon build --symlink-install
# Return to the original directory
cd "$ORIGINAL_DIR"
# Setup ros_ws environment
. "/ros_ws/install/local_setup.bash"
# Execute the passed commands (if any)
# exec "$@"
# Run bash
exec bash