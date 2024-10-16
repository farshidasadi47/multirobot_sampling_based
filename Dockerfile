########################################################################
# This is Dockerfile for multi_robot_sampling_based robotics project.
# This Dockerfile was greatly inspired by the guide on docker and ROS2
# from roboticseabass.com.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## base image ##################################################
ARG ROS_DISTRO=humble
# Use an official ROS2 image
FROM osrf/ros:${ROS_DISTRO}-desktop AS base
# Set environment variables for ROS2
ENV ROS_DISTRO=${ROS_DISTRO}
ENV DEBIAN_FRONTEND=noninteractive
# Set bash as shell with arg -c to run commands with RUN
SHELL ["/bin/bash", "-c"]
ENV SHELL="/bin/bash"
# Create a workspace and download the micro-ROS
WORKDIR /microros_ws
RUN git clone -b $ROS_DISTRO \
 https://github.com/micro-ROS/micro_ros_setup.git src/micro_ros_setup
# Update dependencies using rosdep
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && sudo apt update && rosdep update \
 && rosdep install --from-paths src --ignore-src -y
# Install pip
RUN apt-get update && apt-get upgrade -y \
  && apt-get install -y python3-pip ffmpeg python3-tk
# Use Cyclone DDS as middleware
RUN apt-get update && apt-get install -y --no-install-recommends \
 ros-${ROS_DISTRO}-rmw-cyclonedds-cpp
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
# Build micro-ROS tools
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && colcon build --symlink-install
# Download micro-ROS-Agent packages
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && . /microros_ws/install/local_setup.bash \
 && ros2 run micro_ros_setup create_agent_ws.sh
# Build micro-ROS-agent step
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && . /microros_ws/install/local_setup.bash \
 && ros2 run micro_ros_setup build_agent.sh
# Create a workspace
RUN mkdir -p /ros_ws/src/multirobot_sampling_based
WORKDIR /ros_ws
COPY .  /ros_ws/src/multirobot_sampling_based
# Build workspace
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && colcon build --symlink-install
# Source ROS 2 installation, micro-ROS, and workspace
RUN echo ". /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
 && echo ". /microros_ws/install/local_setup.bash" >> ~/.bashrc \
 && echo ". /ros_ws/install/local_setup.bash" >> ~/.bashrc
# Copy entry point
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh
COPY ./rebuild.sh /rebuild.sh
RUN chmod +x /rebuild.sh
########## overlay #####################################################
FROM base AS overlay
# Remove system-installed matplotlib to avoid conflicts.
RUN apt-get remove -y python3-matplotlib python-matplotlib-data || true
# Add the Python dependencies
RUN pip install numpy==1.26.4 \
    matplotlib==3.9.2 \
    opencv-python==4.10.0.84 \
    scipy==1.14.1 \
    faiss-cpu==1.8.0.post1 \
    triangle==20230923 \
    python-fcl==0.7.0.6 \
    seaborn==0.13.2 \
    scikit-video==1.1.11 \
    pillow==11.0.0 \
    black \
    ipykernel
# Build the package
WORKDIR /ros_ws
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && colcon build --symlink-install \
 && . /ros_ws/install/local_setup.bash
# These commands add ROS2 packages to PYTHONPATH and PATH.
RUN echo '. /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc \
    && echo '. /microros_ws/install/local_setup.bash' >> ~/.bashrc \
    && echo '. /ros_ws/install/local_setup.bash' >> ~/.bashrc
########## adruino #####################################################
FROM base AS arduino
# Dev container arguments
ARG USERNAME=devuser
ARG GROUPNAME=${USERNAME}
ARG UID=1000
ARG GID=${UID}
# Create new user and home directory
RUN groupadd --gid $GID $GROUPNAME \
 # Ad user, asign group id, and create its home directory
 && useradd --uid ${UID} --gid ${GID} --create-home ${USERNAME}\ 
 # Give devuser ability execute commands as root.
 && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}
# Set HOME directory
ENV HOME=/home/${USERNAME}
# Set ownership of home directory
RUN chown -R ${UID}:${GID} /home
# Download arduino cli
WORKDIR ${HOME}
RUN curl -fsSL https://github.com/arduino/arduino-cli/releases/download/v1.0.4/arduino-cli_1.0.4_Linux_64bit.tar.gz \
  -o arduino-cli.tar.gz
# Install arduino cli
RUN mkdir -p ./arduino-cli \
  && tar -xzf arduino-cli.tar.gz -C ./arduino-cli \
  && rm arduino-cli.tar.gz
# Install arduino duo board
WORKDIR ${HOME}/arduino-cli
RUN ./arduino-cli core update-index \
  && ./arduino-cli core install arduino:sam@1.6.12
# Install libraries
RUN ./arduino-cli lib install "Regexp@0.1.0"
# Download micro ROS for arduino
WORKDIR ${HOME}/Arduino/libraries
RUN curl -fsSL https://github.com/micro-ROS/micro_ros_arduino/archive/refs/tags/v2.0.7-humble.tar.gz -o micro_ros_arduino.tar.gz \
  && tar -xzf micro_ros_arduino.tar.gz \
  && rm micro_ros_arduino.tar.gz
# Patch the Arduino SAM platform
WORKDIR ${HOME}/.arduino15/packages/arduino/hardware/sam/1.6.12/
RUN curl -fsSL https://raw.githubusercontent.com/micro-ROS/micro_ros_arduino/main/extras/patching_boards/platform_arduinocore_sam.txt -o platform.txt
# Write a sript to compile and upload code to arduino
# Create a shell script in /root with the necessary commands
RUN echo "#!/bin/bash" >> ${HOME}/compile.sh
RUN echo "arduino-cli compile --fqbn arduino:sam:arduino_due_x /ros_ws/src/multirobot_sampling_based/multirobot_sampling_based/helmholtz_arduino/helmholtz_arduino.ino" >> ${HOME}/compile.sh
RUN echo "#!/bin/bash" >> ${HOME}/upload.sh
RUN echo "arduino-cli upload -p /dev/ttyACM0 --fqbn arduino:sam:arduino_due_x /ros_ws/src/multirobot_sampling_based/multirobot_sampling_based/helmholtz_arduino/helmholtz_arduino.ino" >> ${HOME}/upload.sh
# Make the shell script executable
RUN chmod +x ${HOME}/compile.sh ${HOME}/upload.sh
# Add the arduino-cli directory to PATH
RUN echo "export PATH=$PATH:${HOME}/arduino-cli" >> ~/.bashrc
# Set the default command to run when the container starts
WORKDIR ${HOME}
########## dev #########################################################
FROM overlay AS dev
# Dev container arguments
ARG USERNAME=devuser
ARG GROUPNAME=${USERNAME}
ARG UID=1000
ARG GID=${UID}
# Create new user and home directory
RUN groupadd --gid $GID $GROUPNAME \
 # Ad user, asign group id, and create its home directory
 && useradd --uid ${UID} --gid ${GID} --create-home ${USERNAME}\ 
 # Give devuser ability execute commands as root.
 && echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
 && chmod 0440 /etc/sudoers.d/${USERNAME}
# Set HOME directory
ENV HOME=/home/${USERNAME}
# Set ownership of home directory
RUN chown -R ${UID}:${GID} /home
# Set ownership of ros_ws
RUN chown -R ${UID}:${GID} /ros_ws/
# Rebuild workspaces
WORKDIR /ros_ws
RUN . /opt/ros/$ROS_DISTRO/setup.bash \
 && colcon build --symlink-install
# These commands add ROS2 packages to PYTHONPATH and PATH.
RUN echo '. /opt/ros/$ROS_DISTRO/setup.bash' >> ~/.bashrc \
    && echo '. /microros_ws/install/local_setup.bash' >> ~/.bashrc \
    && echo '. /ros_ws/install/local_setup.bash' >> ~/.bashrc
# Set the user
USER ${USERNAME}