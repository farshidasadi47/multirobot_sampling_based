# %%
########################################################################
# This files hold examples of the paper.
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import os
import time
import logging

import numpy as np
from matplotlib import pyplot as plt

try:
    from multirobot_sampling_based import model, rrt
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    import model, rrt

round_down = model.round_down
wrap_2pi = model.wrap_2pi
define_colors = model.define_colors


########## Examples ####################################################
def example_1(tol_cmd=0.01, goal_bias=0.04, max_size=8500):
    np.random.seed(42)  # Keep for consistency, but can be removed.
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo10()
    # Obstacle contours.
    obstacle_contours = [
        np.array(
            [[-45, 5], [-45, -200], [-55, -200], [-55, 5]],
            dtype=float,
        ),
        np.array([[55, 200], [55, -5], [45, -5], [45, 200]], dtype=float),
    ]
    obstacles = rrt.Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)
    collision = rrt.Collision(mesh, specs, with_coil=False)
    pose = np.array(
        [
            100,
            75,
            100,
            70,
            100,
            60,
            100,
            50,
            100,
            40,
            100,
            30,
            100,
            20,
            100,
            10,
            100,
            0,
            100,
            -10,
        ],
        dtype=float,
    )
    _ = collision.is_collision(pose)

    N = 100
    pose_i = np.array(
        [
            100,
            90,
            100,
            70,
            100,
            50,
            100,
            30,
            100,
            10,
            100,
            -10,
            100,
            -30,
            100,
            -50,
            100,
            -70,
            100,
            -90,
        ],
        dtype=float,
    )
    pose_f = np.array(
        [
            -100,
            90,
            -100,
            70,
            -100,
            50,
            -100,
            30,
            -100,
            10,
            -100,
            -10,
            -100,
            -30,
            -100,
            -50,
            -100,
            -70,
            -100,
            -90,
        ],
        dtype=float,
    )
    #
    planner = rrt.RRTS(
        specs,
        collision,
        obstacle_contours,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=max_size,
    )
    start_time = time.time()
    planner.plans(pose_i, pose_f, np.arange(10), anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    rrt.logger.debug(
        f"The runtime of the test() function is {runtime} seconds"
    )
    cmds = model.cartesian_to_polar(planner.cmds)
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    poses, cmds = simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    file_name = None  # "example_1"
    simulation.plot_selected([0, 9], file_name=file_name)
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        vel=10,
        anim_length=1100,
        boundary=True,
        last_section=True,
        file_name=file_name,
    )


def example_3(tol_cmd=15.0, goal_bias=0.09):
    np.random.seed(42)  # Keep for consistency, but can be removed.
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo(4)
    specs.set_space(rcoil=90)
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-5, -30], [-5, -100], [5, -100], [5, -30]], dtype=float),
        np.array([[-5, 100], [-5, 30], [5, 30], [5, 100]], dtype=float),
    ]
    obstacles = rrt.Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)
    collision = rrt.Collision(mesh, specs)
    #
    pose_i = np.array([40, 45, 40, 15, 40, -15, 40, -45], dtype=float)
    # pose_i = np.array([-43,40, -43,18, -45,-10, -36,-45], dtype=float)
    pose_f = np.array([-40, 45, -40, 15, -40, -15, -40, -45], dtype=float)
    #
    planner = rrt.RRTS(
        specs,
        collision,
        obstacle_contours,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=20000,
    )
    start_time = time.time()
    # Find plan.
    planner.plans(
        pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False
    )
    end_time = time.time()
    runtime = end_time - start_time
    rrt.logger.debug(
        f"The runtime of the test() function is {runtime} seconds"
    )
    # Process the command.
    cmds = planner.cmds
    # Uncomment if you want mode change.
    # cmds = planner.post_process(planner.cmds, ang=0)  # Add mode change.
    cmds = model.cartesian_to_polar(cmds)  # Convert to polar.
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    poses, cmds = simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    file_name = None  # "example_3"
    simulation.plot_selected([0, 3])
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        vel=10,
        anim_length=1100,
        boundary=True,
        last_section=True,
        file_name=file_name,
    )


########## test section ################################################
if __name__ == "__main__":
    # example_1()
    plt.show()
