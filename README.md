# multirobot_sampling_based
This repository contains the code for the research paper:
> Sampling-based motion planning for multiple heterogeneuous magnetic robots under global input using RRT<sup>*</sup>

It performs motion planning for a class of heterogeneous robots using an adapdation of RRT.

Videos related to the examples in the paper can be seen below:

Example 1: Animation of moving 10 robots through obstacles and aligning them on specified positions | Example 3: Moving 4 robots through a passage and aligning them on specified positions | Example 3: Animation of moving 4 robots through a passage and aligning them on specified positions
---|---|---
[![Video 1](https://img.youtube.com/vi/XgpuL6UDEKw/0.jpg)](https://www.youtube.com/watch?v=XgpuL6UDEKw) | [![Video 2](https://img.youtube.com/vi/OFOvsfrkxKw/0.jpg)](https://www.youtube.com/watch?v=OFOvsfrkxKw) | [![Video 3](https://img.youtube.com/vi/KzKwsqhUuUU/0.jpg)](https://www.youtube.com/watch?v=KzKwsqhUuUU)

## Structure of the code
The repository is a ROS2 package and the main code is in `swarm` folder.

The structure of this folder is:
```
multirobot_sampling_based/
    helmholtz_arduino/
    paper/
    rrtparam/
    model.py
    rrt.py
    ratio.py
    localization.py
    closedloop.py
    rosclosed.py
```
- `helmholtz_arduino/` contain arduino code for the microcontroller that runs coil DC drivers. The code uses `micro-ROS` to communicate through serial port with computer. 
You may need to change it based on the microcontroller and DC driver that you use. To work with `micro-ros` see [micro-ROS for Arduino](https://github.com/micro-ROS/micro_ros_arduino).
- `paper/` for scalability analysis and processing experimental results for the paper.
- `model.py` includes classes that define robot specifications and simulators for visualizing motion plans.
- `rrt.py` module includes classes for obstacle triangulation, collision detection, and motion planning using the Adapted RRT<sup>*</sup> method proposed in our paper.
- `localization.py` is the image processing module that identifies and localizes robots.
- `closedloop.py` is the module that gets the motion plan and produces the low level motions needed to execute the command. 
It also contains methods to perform closedloop control as describged in our previous paper.
- `rosclosed.py` contains ROS2  processes to run the image processing and hardware communications. It uses all above modules in it.
The `localization.py`, `closedloop.py`, and `rosclosed.py` modules are required only for experiments involving hardware.

The `model.py` and `rrt.py` modules are sufficient for running simulations and will be described more.
## Dependencies
The python dependencies of `model.py` and `planner.py` can be installed as:
```
pip install numpy==1.26.4
pip install matplotlib==3.9.2
pip install opencv-python==4.10.0.84
pip install scipy==1.14.1
pip install faiss-cpu==1.8.0.post1
pip install triangle==20230923
pip install python-fcl==0.7.0.6
```

## Interfaces
### `model.py` module
This module contains two main classes:
- `SwarmSpecs` which is mainly used to store specifictions of robot group and boundaries of the workspace.
```python
import model  # Import model module.
# Defining specificiations of robot group.
length = np.array(
    [[10, 9, 7], [10, 7, 5], [10, 5, 9]]
) # Each row corresponds to a robot, each column corresponds to a mode.
# Defining obstacle outline contours.
# Each obstacle is defined by array of vertices of its external surronding contour.
obstacle_contours = [
    np.array([[-5, -25], [-5, -90], [5, -90], [5, -25]], dtype=float),
    np.array([[-5, 90], [-5, 25], [5, 25], [5, 90]], dtype=float),
]
# Construct SwarmSpecs instance.
specs = model.SwarmSpecs(length, obstacle_contours=obstacle_contours)
```
- `Simulation` which is mainly used to visualize a given motion plan.
``` python
# Create simulation instance.
simulation = model.Simulation(specs)
# Initial position of robots.
pos = [-20.0, 0.0, 0.0, 0.0, 20.0, 0.0]
# Some pseudo displacement.
# Here each row is a pseudo displacement given as
    [pseudo displacement length, pseudo displacement angle, pseudo displacement mode]
cmds = [
        [50, np.radians(90), 1],
        [50, np.radians(180), 2],
        [10, np.radians(270), -1],
        [30, np.radians(-45), 0],
]
# Simulate the system.
simulation.simulate(cmds, pos)
# Draw simulation results.
file_name = None # Change it to a string to save the visualization.
# Plot the simulation.
simulation.simplot(
    step=10,  # Discretization length.
    plot_length=1000,  # How many steps to draw.
    boundary=True,  # Draw the boundary of workspace.
)
# Animate the simulation
anim = simulation.simanimation(
    anim_length=1100,  # How many steps to animate.
    vel=30,  # How fast the robot zero moves in mm/s.
    boundary=True,  # Draw boundary of workspace.
    last_section=True,  # Only animate the last section at any moment to prevent visual clutter.
)
plt.show()
```
### `rrt.py` module
```python
import model  # Import model module.
import rrt  # Import planning module.
# Build specs of robots and obstacles.
specs = model.SwarmSpecs.robo(4)  # Prebuild specification of 4 robot system.
# Obstacle contours.
obstacle_contours = [
    np.array([[-5, -30], [-5, -100], [5, -100], [5, -30]], dtype=float),
    np.array([[-5, 100], [-5, 30], [5, 30], [5, 100]], dtype=float),
]
# Build obstacle triangulation instance.
obstacles = rrt.Obstacles(specs, obstacle_contours)
# Calculate obstacle triangular mesh.
obstacle_contours = obstacles.get_cartesian_obstacle_contours()
mesh, mesh_contours = obstacles.get_obstacle_mesh()
# Add obstacles to specification for future visualization.
specs.set_obstacles(obstacle_contours=obstacle_contours)
# Build collision detection instance.
collision = rrt.Collision(mesh, specs)
# Start and goal positions.
pose_i = np.array([40, 45, 40, 15, 40, -15, 40, -45], dtype=float)
pose_f = np.array([-40, 45, -40, 15, -40, -15, -40, -45], dtype=float)
# Buils Adapted RRT<sup>*</sup> instance.
planner = rrt.RRTS(
    specs,
    collision,
    obstacle_contours,
    tol_cmd=tol_cmd,  # Tolerance for removing negligible pseudodisplacements.
    goal_bias=goal_bias,  # Bias for sampling goal position during tree expansion.
    max_size=20000,  # Maximum number of tree nodes.
)
# Search for motion plan, do not visualize the search process.
rrt.plans(pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False)
# Process the command.
cmds = planner.cmds
# Add mode change and modify tumbling (mode 0) if planning for multi-face millirobots.
# cmds = rrt.post_process(rrt.cmds, ang=10)
# Convert the motion plan to polar coordinate.
cmds = model.cartesian_to_polar(cmds)
# Build simulation instance.
simulation = model.Simulation(specs)
# Simulate the system
poses, cmds = simulation.simulate(cmds, pose_i)
# Plot simulation results.
simulation.simplot(
    step=10,  # Discretization length.
    plot_length=1000,  # How many steps to plot.
    boundary=True,  # Draw workspace boundary.
)
# Animate the simulation result.
anim = simulation.simanimation(
    anim_length=1100,  # Ho many steps to animate.
    vel=30,  # How fast the robot zero moves in mm/s.
    boundary=True,  # Draw boundary.
    last_section=True,  # Only show the last step at any instance of animation.
)
```
## Docker
The project includes a dockerfile for easier execution.
### VS Code Dev Containers Extension
The recommended way to run the project is by using the VS Code Dev Containers extension following these steps:
- Install Docker and VS Code.
- Install the `Dev Containers` extension of VS Code.
- In the root directory of the project, build the project using the following command:
  ```
  docker compose --profile "*" build --no-cache
  ```
- In the root directory, open the Dev Containers by clicking the button in the bottom left corner of VS Code and selecting `Reopen in Container`.
  If encountering any error, select `Rebuild Container` and then reopen it.
- Once inside the container, run files as needed.
