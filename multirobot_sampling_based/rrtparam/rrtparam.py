# %%
########################################################################
# This files used for studying rrt parameters.
# This is part of sampling based motion planning for heterogeneous
# magnetic robots.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import os
import time
from itertools import product
import json
import subprocess
from concurrent.futures import as_completed, ThreadPoolExecutor
import re
from collections import defaultdict
from multiprocessing import cpu_count

import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

try:
    from multirobot_sampling_based import model
    from multirobot_sampling_based.rrt import Obstacles, Collision, RRT, RRTS
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    import model
    from rrt import Obstacles, Collision, RRT, RRTS

N_CPU = cpu_count() // 2
script_dir = os.path.dirname(__file__)
rrt_subprocess_path = os.path.join(script_dir, "rrt_subprocess.py")


########## classes and functions #######################################
def rrt3(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    np.random.seed()
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo3()
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-5, -25], [-5, -90], [5, -90], [5, -25]], dtype=float),
        np.array([[-5, 90], [-5, 25], [5, 25], [5, 90]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs)
    pose = np.array([-60, 0, 0, -30, 60, 20], dtype=float)
    _ = collision.is_collision(pose)

    N = 100
    pose_i = np.array([60.0, 40.0, 60.0, 0.0, 60.0, -40.0], dtype=float)
    pose_f = np.array([-60.0, 40.0, -60.0, 0.0, -60.0, -40.0], dtype=float)
    poses = np.linspace(pose_i, pose_f, N + 1)

    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i, pose_f, [0, 1, 2], anim_online=False, plot=False, log=False
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def rrt4(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo4()
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-5, -25], [-5, -90], [5, -90], [5, -25]], dtype=float),
        np.array([[-5, 90], [-5, 25], [5, 25], [5, 90]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs)
    pose = np.array([50, 40, 50, 15, 50, -15, 50, -40], dtype=float)
    _ = collision.is_collision(pose)
    #
    N = 100
    pose_i = np.array([50, 40, 50, 15, 50, -15, 50, -40], dtype=float)
    pose_f = np.array([-50, 40.0, -50, 15, -50, -15, -50, -40], dtype=float)
    poses = np.linspace(pose_i, pose_f, N + 1)

    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i,
            pose_f,
            [0, 1, 2, 3, 4],
            anim_online=False,
            plot=False,
            log=False,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def errt4(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo(4)
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-5, -30], [-5, -100], [5, -100], [5, -30]], dtype=float),
        np.array([[-5, 100], [-5, 30], [5, 30], [5, 100]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs)
    #
    pose_i = np.array([40, 45, 40, 15, 40, -15, 40, -45], dtype=float)
    pose_f = np.array([-40, 45.0, -40, 15, -40, -15, -40, -45], dtype=float)
    #
    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i,
            pose_f,
            [0, 1, 2, 3, 4],
            anim_online=False,
            plot=False,
            log=False,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def rrt5(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo5()
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-5, -25], [-5, -90], [5, -90], [5, -25]], dtype=float),
        np.array([[-5, 90], [-5, 25], [5, 25], [5, 90]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs)
    pose = np.array([50, 40, 50, 20, 50, 0, 50, -20, 50, -40], dtype=float)
    _ = collision.is_collision(pose)
    #
    N = 100
    pose_i = np.array([50, 50, 50, 25, 50, 0, 50, -25, 50, -50], dtype=float)
    pose_f = np.array(
        [-50, 50.0, -50, 25, -50, 0, -50, -25, -50, -50], dtype=float
    )
    poses = np.linspace(pose_i, pose_f, N + 1)

    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i,
            pose_f,
            [0, 1, 2, 3, 4],
            anim_online=False,
            plot=False,
            log=False,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def rrt10big(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    np.random.seed()
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo10()
    specs.set_space(lbx=-200, ubx=200, lby=-150, uby=150, rcoil=300)
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-75, 5], [-75, -200], [-65, -200], [-65, 5]], dtype=float),
        np.array([[65, 200], [65, -5], [75, -5], [75, 200]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs, with_coil=False)
    pose = np.array(
        [
            140,
            75,
            140,
            70,
            140,
            60,
            140,
            50,
            140,
            40,
            140,
            30,
            140,
            20,
            140,
            10,
            140,
            0,
            140,
            -10,
        ],
        dtype=float,
    )
    _ = collision.is_collision(pose)

    N = 100
    pose_i = np.array(
        [
            140,
            90,
            140,
            70,
            140,
            50,
            140,
            30,
            140,
            10,
            140,
            -10,
            140,
            -30,
            140,
            -50,
            140,
            -70,
            140,
            -90,
        ],
        dtype=float,
    )
    pose_f = np.array(
        [
            -140,
            90,
            -140,
            70,
            -140,
            50,
            -140,
            30,
            -140,
            10,
            -140,
            -10,
            -140,
            -30,
            -140,
            -50,
            -140,
            -70,
            -140,
            -90,
        ],
        dtype=float,
    )

    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i,
            pose_f,
            np.arange(11),
            anim_online=False,
            plot=False,
            log=False,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def rrt10(
    max_size,
    tol_cmd=1e-2,
    goal_bias=0.05,
    **kwargs,
):
    np.random.seed()
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo10()
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-45, 5], [-45, -200], [-55, -200], [-55, 5]], dtype=float),
        np.array([[55, 200], [55, -5], [45, -5], [45, 200]], dtype=float),
    ]
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification.
    specs.set_obstacles(obstacle_contours=obstacle_contours)

    collision = Collision(mesh, specs, with_coil=False)
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
    rrt = RRT(
        specs,
        collision,
        obstacle_contours,
        max_size=max_size,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
    )
    for _ in range(3):
        start_time = time.time()
        rrt.plan(
            pose_i,
            pose_f,
            np.arange(11),
            anim_online=False,
            plot=False,
            log=False,
        )
        end_time = time.time()
        runtime = end_time - start_time
        print(f"The runtime of the test() function is {runtime} seconds")
        paths = rrt._paths
        # Check if collision detection was faulty.
        if len(paths):
            # Check collision multiple time to avoid any fault.
            checks = []
            for _ in range(5):
                checks.append(collision.is_collision_path(paths[-1]["poses"]))
            # If there was a fault, repeat one more time.
            if max(checks):
                print("Repeating due to error in collision avoidance.")
                continue
        break
    #
    values_iterations = [(p["value"], p["i"]) for p in rrt._paths]
    result = {
        "values_iterations": values_iterations,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
    }
    return result


def write_results(results, file_name):
    file_name += r".json"
    with open(file_name, "w") as file:
        json.dump(results, file, indent=4)


def get_statistics(data):
    means, stds, imeans, istds, ns = [], [], [], [], []
    for value, iteration in zip(data["values"], data["iterations"]):
        value = np.array(value)
        mask = value > -1
        n = np.sum(value > -1).item()
        if n > 0:
            means.append(np.mean(value[mask]))
            stds.append(np.std(value[mask]))
            imeans.append(np.mean(iteration))
            istds.append(np.std(iteration))
        else:
            means.append(-1)
            stds.append(0)
            imeans.append(-1)
            istds.append(0)
        ns.append(n)
    return {
        "means": means,
        "stds": stds,
        "ns": ns,
        "imeans": imeans,
        "istds": istds,
    }


def evaluate_single(planner, max_size, n=10, **params):
    all_values = []
    all_iterations = []
    for _ in range(n):
        result = planner(max_size, **params)
        if len(result["values_iterations"]):
            values_iterations = result.pop("values_iterations")
            values, iterations = list(zip(*values_iterations))
            all_values.append(values[-1])
            all_iterations.append(iterations[-1])
        else:
            _ = result.pop("values_iterations")
            all_values.append(-1)
            all_iterations.append(-1)
    result["values"] = all_values
    result["iterations"] = all_iterations
    return result


def evaluate(planner, max_size, n=10, file_name=None, **param_ranges):
    param_names = param_ranges.keys()
    parameters = [
        dict(zip(param_names, combination))
        for combination in product(*param_ranges.values())
    ]
    #
    all_results = []
    for params in parameters:
        result = evaluate_single(planner, max_size, n=n, **params)
        all_results.append(result)
        print(result)
    # Summarize the results.
    results = {
        name: [result[name] for result in all_results]
        for name in all_results[0].keys()
    }
    results.update(get_statistics(results))
    if file_name is not None:
        write_results(results, file_name)
    return results


def evaluate_single_subprocess(planner, max_size, n=10, **params):
    params_json = json.dumps(params)  # Convert params to JSON string
    results = []

    def subprocessor():
        try:
            process = subprocess.Popen(
                [
                    sys.executable,
                    rrt_subprocess_path,
                    "--planner_name",
                    planner.__name__,
                    "--max_size",
                    str(max_size),
                    "--params",
                    params_json,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate(timeout=5400)
            return stdout, stderr
        except Exception as e:
            process.kill()
            return None, f"An unexpected error occurred: {e}"

    with ThreadPoolExecutor(max_workers=N_CPU) as executor:
        futures = [executor.submit(subprocessor) for _ in range(n)]
        for future in as_completed(futures):
            stdout, stderr = future.result(timeout=5400)
            if stdout:
                matched = re.search(r"(\{.+\})$", stdout.decode("utf-8"))
                if matched:
                    result = json.loads(matched.group(0))
                    results.append(result)
                else:
                    print(f"{stdout = }")
            else:
                print(f"{stderr}")

    # Extract useful info.
    all_values = []
    all_iterations = []
    for result in results:
        if len(result["values_iterations"]):
            values_iterations = result.pop("values_iterations")
            values, iterations = list(zip(*values_iterations))
            all_values.append(values[-1])
            all_iterations.append(iterations[-1])
        else:
            _ = result.pop("values_iterations")
            all_values.append(-1)
            all_iterations.append(-1)
    result["values"] = all_values
    result["iterations"] = all_iterations
    return result


def evaluate_from_subprocess(
    planner, max_size, n=10, file_name=None, **param_ranges
):
    param_names = param_ranges.keys()
    parameters = [
        dict(zip(param_names, combination))
        for combination in product(*param_ranges.values())
    ]
    #
    all_results = []
    for params in parameters:
        result = evaluate_single_subprocess(planner, max_size, n=n, **params)
        all_results.append(result)
        print(result)
    # Summarize the results.
    results = {
        name: [result[name] for result in all_results]
        for name in all_results[0].keys()
    }
    results.update(get_statistics(results))
    if file_name is not None:
        write_results(results, file_name)
    return results


def plot_cmd_bias(data, values=True, log=False):
    name = None
    if isinstance(data, str):
        name = data
        # Read data from file.
        with open(data, "r") as file:
            data = json.load(file)

    tol_cmd = data["tol_cmd"]
    goal_bias = data["goal_bias"]
    if values:
        means = data["means"]
    else:
        means = data["imeans"]
    try:
        # Create a 3D plot
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")

        if log:
            tol_cmd = np.log10(tol_cmd)

        # Plot data
        ax.plot_trisurf(
            tol_cmd, goal_bias, means, cmap="viridis", edgecolor="none"
        )

        # Add labels and title
        ax.set_xlabel("tol_cmd" + (" (log scale)" if log else ""))
        ax.set_ylabel("goal_bias")
        ax.set_zlabel(f"Mean of {'Values' if values else'iterations'}")
        ax.set_title(f"3D Plot of tol_cmd, goal_bias, and Means {name}")
    except Exception:
        pass


def plot_means(data, choice, values=True, log=False, avg=True):
    choices = ["tol_cmd", "goal_bias"]
    Name = None
    if isinstance(data, str):
        # Read data from file.
        name = data
        with open(data, "r") as file:
            data = json.load(file)

    choice = choices[choice]
    chosen = data[choice]
    if values:
        means = data["means"]
    else:
        means = data["imeans"]
    ns = data["ns"]
    if avg:
        chosen_dict = defaultdict(list)
        chosen_dict_n = defaultdict(list)
        for x, m, n in zip(chosen, means, ns):
            chosen_dict[x].append(m)
            chosen_dict_n[x].append(n)

        chosen = []
        means = []
        for c, m in chosen_dict.items():
            chosen.append(c)
            n = np.array(chosen_dict_n[c], dtype=float)
            m = np.array(m, dtype=int)
            m = np.sum(m * n) / (np.sum(n) if np.sum(n) > 0 else 1)
            means.append(m)
    if log:
        chosen = np.log10(chosen)
    # Create a 2D plot
    plt.figure(figsize=(10, 7))

    plt.plot(chosen, means, marker="o", linestyle="-")
    # Add labels and title
    plt.xlabel(choice)
    plt.ylabel(f"Mean of {'values' if values else 'iterations'}")
    plt.title(
        f"Plot of Means vs {choice} {'(Log Scale)' if log else ''} {name}"
    )


def plot_box(data, choice, values=True):
    choices = ["tol_cmd", "goal_bias"]
    name = None
    if isinstance(data, str):
        name = data
        # Read data from file.
        with open(data, "r") as file:
            data = json.load(file)

    choice = choices[choice]
    chosen = data[choice]
    if values:
        datas = data["values"]
    else:
        datas = data["iterations"]
    # Flatten the chosen and values lists for boxplot
    chosen_flat = np.concatenate([[i] * len(j) for i, j in zip(chosen, datas)])
    datas_flat = np.concatenate(datas)
    # Filter for statistics.
    mask = datas_flat > 0.0
    f_datas_flat = datas_flat[mask]
    f_chosen_flat = chosen_flat[mask]

    # Create a boxplot
    plt.figure(figsize=(14, 8))
    sns.boxplot(
        x=f_chosen_flat, y=f_datas_flat, fill=False, whis=[0, 100], color="k"
    )
    sns.stripplot(x=chosen_flat, y=datas_flat, jitter=True, color="b", size=6)
    # Add labels and title
    plt.xlabel(choice)
    plt.ylabel(f"{'Values' if values else 'Iterations'}")
    plt.title(f"Box Plot of Values vs {choice} {name}")
    plt.xticks(rotation=90)
    plt.grid(axis="y", linestyle="--", linewidth=0.5)


def print_sorted(data, sort_by="ns"):
    name = None
    if isinstance(data, str):
        name = data
        # Read data from file.
        with open(data, "r") as file:
            data = json.load(file)
    inds = np.argsort(data[sort_by])[::-1]
    print(f"Reporting {name}")
    print(
        f"{'tol_cmd':<10} {'goal_bias':<10} {'means':<15} {'n':<5} {'minim':<7} {'maxim':<7}"
    )
    print("-" * 80)
    for i in inds:
        n = data["ns"][i]
        if n > 0:
            tol_cmd = data["tol_cmd"][i]
            goal_bias = data["goal_bias"][i]
            means = data["means"][i]
            iterations = [x for x in data["iterations"][i] if x > 0]
            minim = min(iterations)
            maxim = max(iterations)
            print(
                f"{tol_cmd:<10.4g} {goal_bias:<10.4g} {means:<15.2f} {n:<5} {minim:<7} {maxim:<7}"
            )
    print("*" * 80)


def log_range(start, stop, n=1):
    points = 10.0 ** np.arange(start, stop)
    arr = []
    for a, b in zip(points[:-1], points[1:]):
        arr.append(np.linspace(a, b, n + 1))
    return np.unique(np.concatenate(arr))


def eval_param_rrt3():
    file_name = "rrt301_5000_11"
    planner = rrt3
    max_size = 5000
    n = 11
    params = {"tol_cmd": 1e-2, "goal_bias": 0.05}
    param_ranges = {
        "tol_cmd": log_range(-2, 2, 2).tolist() + [20.0, 30.0],
        "goal_bias": log_range(-2, 0, 2).tolist() + [0.2, 0.3],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


def eval_param_rrt4():
    file_name = "rrt4_5000_11"
    planner = rrt4
    max_size = 5000
    n = 11
    params = {"tol_cmd": 1e-2, "goal_bias": 0.05}
    param_ranges = {
        "tol_cmd": log_range(-2, 2, 2).tolist() + [20.0, 30.0],
        "goal_bias": log_range(-2, 0, 2).tolist() + [0.2, 0.3],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


def eval_param_errt4():
    file_name = "errt41_10000_11"
    planner = errt4
    max_size = 10000
    n = 11
    params = {"tol_cmd": 1e-2, "goal_bias": 0.05}
    param_ranges = {
        "tol_cmd": log_range(-2, 2, 2).tolist() + [20.0, 30.0],
        "goal_bias": log_range(-2, 0, 2).tolist() + [0.2, 0.3],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


def eval_param_rrt5():
    file_name = "rrt5_10000_11"
    planner = rrt5
    max_size = 10000
    n = 11
    params = {"tol_cmd": 1e-2, "goal_bias": 0.05}
    param_ranges = {
        "tol_cmd": log_range(-2, 2, 2).tolist() + [20.0, 30.0],
        "goal_bias": log_range(-2, 0, 2).tolist() + [0.2, 0.3],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


def eval_param_rrt10big():
    file_name = "rrt10_200000_11"
    planner = rrt10big
    max_size = 200000
    n = 11
    params = {"tol_cmd": 1e-1, "goal_bias": 1e-3}
    param_ranges = {
        "tol_cmd": log_range(-2, 2, 2).tolist(),
        "goal_bias": log_range(-2, 0, 2).tolist() + [0.2, 0.3],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


def eval_param_rrt10():
    file_name = "rrt10_200000_11"
    planner = rrt10
    max_size = 200000
    n = 11
    params = {"tol_cmd": 1e-1, "goal_bias": 1e-3}
    param_ranges = {
        "tol_cmd": [0.01],
        "goal_bias": [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12],
    }
    start_time = time.time()
    results = evaluate_from_subprocess(
        planner, max_size, n=n, file_name=file_name, **param_ranges
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(results)
    print(f"The runtime of the test() function is {runtime} seconds")


########## test section ################################################
if __name__ == "__main__":
    # 3 robot.
    f_31 = "rrt31_5000_15.json"
    s_31 = "rrts31_5000_15.json"
    files_3 = [f_31, s_31]
    # 4 robot.
    f_41 = "rrt41_5000_8.json"
    f_41c = "rrt41_cmd01_20000_15.json"
    s_41c = "rrts41_cmd01_20000_15.json"
    files_4 = [f_41]
    # 4 robot experimental.
    ef_41 = "errt41_10000_11.json"  # 40, 45, ...
    ef_42 = "errt42_10000_11.json"  # 45, 45, ...
    ef_43 = "errt43_10000_11.json"  # 50, 45, ...
    ef_41c = "errt41_cmd01_20000_15.json"
    es_41c = "errts41_cmd01_20000_15.json"
    efiles_4 = [ef_41, ef_42, ef_41c, es_41c]
    # 5 robot.
    f_51 = "rrt51_50000_8.json"
    files_5 = [f_51]
    # 10 robot.
    f_101 = "rrt101_100000_10.json"
    f_101c = "rrt101_cmd01_200000_11.json"
    f_101cg = "rrt_101_cmd01_g04_300000_100.json"
    files_10 = [f_101]
    #
    files = files_10
    for file in files:
        # plot_cmd_bias(file, values=True, log=True)
        # plot_means(file, 1, values=False, log=False, avg=True)
        plot_box(file, 0, values=True)
        print_sorted(file, "means")
        pass
    plt.show()
