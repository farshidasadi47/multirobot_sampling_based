# %%
########################################################################
# This files hold classes and functions that studies scalability of the
# heterogeneuous robot system, using rrt.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys
import time
import json

import numpy as np
from scipy.optimize import curve_fit

np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["figure.figsize"] = [7.2, 8.0]
plt.rcParams.update({"font.size": 11})
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Computer Modern Roman"
# plt.rcParams["text.usetex"] = True

try:
    from multirobot_sampling_based.model import SwarmSpecs, msg_header
    from multirobot_sampling_based.rrt import RRT, RRTS
    from multirobot_sampling_based.ratio import Ratio
    from multirobot_sampling_based.rrtparam.rrtparam import (
        rrtn,
        evaluate_single_subprocess,
    )
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    from model import SwarmSpecs, msg_header
    from rrt import RRT, RRTS
    from ratio import Ratio
    from rrtparam.rrtparam import rrtn, evaluate_single_subprocess


########## Functions ###################################################
def best_velocity_combo(n_robot, vels):
    print(msg_header(f"{n_robot:>2d} robot best velocity combo"))
    ratios = Ratio(n_robot, n_robot, vels, repeatable=True)
    length, eig = ratios.get_best()
    if abs(eig).min() < 1e-2:
        raise ValueError("Eigen value < 0.01, change velocities.")
    print(f"{length}\n{eig}")
    return length, eig


def get_statistics(data):
    values = np.array(data["values"])
    iterations = data["iterations"]
    mask = values > -1
    n = np.sum(values > -1).item()
    if n > 0:
        mean = np.mean(values[mask])
        std = np.std(values[mask])
        imean = np.mean(iterations)
        istd = np.std(iterations)
    else:
        mean = -1
        std = 0
        imean = -1
        istd = 0
    return {
        "mean": mean,
        "std": std,
        "n_success": n,
        "success_ratio": n / len(values),
        "imean": imean,
        "istd": istd,
    }


def print_result(result):
    print(
        f"height: {result['height']:>7.2f}, "
        f"n_success: {result['n_success']:>3d}, "
        f"success_ratio: {result['success_ratio']:<5.2f}, "
        f"imean: {result['imean']:>8.2f}, "
        f"istd: {result['istd']:>8.2f}, "
        f"mean: {result['mean']:>8.2f}, "
        f"std: {result['std']:>8.2f}, "
    )


def scenario(
    length,
    height,
    max_size=1000,
    tol_cmd=1e-2,
    goal_bias=0.05,
    spacing=1.5,
    dmin=5,
    clr=5,
    n=10,
):
    planner = rrtn
    params = {
        "max_size": max_size,
        "tol_cmd": tol_cmd,
        "goal_bias": goal_bias,
        "length": length.tolist(),
        "height": height,
        "spacing": spacing,
        "dmin": dmin,
        "clr": clr,
    }
    start_time = time.time()
    result = evaluate_single_subprocess(planner, n=n, **params)
    end_time = time.time()
    runtime = end_time - start_time
    print(f"The runtime is {runtime} seconds")
    # Calculate statistics.
    result.update(get_statistics(result))
    return result


def solve_scenario(
    length,
    height_lb=15,
    height_ub=1000,
    threshold=0.9,
    max_size=1000,
    tol_cmd=1e-2,
    goal_bias=0.05,
    spacing=1.5,
    dmin=5,
    clr=5,
    n=10,
    tol_sol=1.0,
    iter_max=10000,
):
    rate = 1.0  # Contraction rate.
    #
    i = 1
    best_height = -1
    best_ratio = 0.0
    best_result = {}
    while i < iter_max:
        height = (height_lb + height_ub) / 2
        msg = (
            f"i:{i:>3d}, "
            f"height_lb: {height_lb:>7.2f}, "
            f"height: {height:>7.2f}, "
            f"height_ub: {height_ub:>7.2f}"
        )
        print("*" * 79)
        print(msg)
        result = scenario(
            length,
            height,
            max_size=max_size,
            tol_cmd=tol_cmd,
            goal_bias=goal_bias,
            spacing=spacing,
            dmin=dmin,
            clr=clr,
            n=n,
        )
        print_result(result)
        success_ratio = result.get("success_ratio", 0)
        # Check for success.
        if success_ratio >= threshold:
            # Decrease upper bound.
            height_ub = height_ub - rate * (height_ub - height)
            best_height = height
            best_ratio = success_ratio
            best_result = result
        else:
            # Increase lower bound.
            height_lb = height_lb + rate * (height - height_lb)
        # Check if stopping criteria is met.
        if (height_ub - height_lb) < tol_sol * 2:
            break
        #
        i += 1
    print("*" * 79)
    print(f"Final height: {best_height:>7.2f}")
    return best_height, best_ratio, best_result


def find_heights(
    n_robots,
    vels,
    threshold=0.9,
    max_size=200000,
    height_lb_adaptive=True,
    height_ub_mult=2.0,
    tol_cmd=1e-2,
    goal_bias=0.05,
    spacing=1.5,
    dmin=5,
    clr=5,
    n=30,
    tol_sol=1.0,
    iter_max=1000,
    filename=None,
):
    n_robots.sort()
    results = {}
    height_lb = dmin + 2 * clr
    for n_robot in n_robots:
        length, _ = best_velocity_combo(n_robot, vels)
        height, success_ratio, result = solve_scenario(
            length,
            height_lb=height_lb,
            height_ub=((n_robot - 1) * dmin + 2 * clr)
            * spacing
            * height_ub_mult,
            threshold=threshold,
            max_size=max_size,
            tol_cmd=tol_cmd,
            goal_bias=goal_bias,
            spacing=spacing,
            dmin=dmin,
            clr=clr,
            n=n,
            tol_sol=tol_sol,
            iter_max=iter_max,
        )
        results[n_robot] = {
            "height": height,
            "success_ratio": success_ratio,
            "result": result,
        }
        print(results[n_robot])
        #
        if height > 0 and height_lb_adaptive:
            height_lb = height - tol_sol
    # Write results if requested.
    if filename is not None:
        filename += r".json"
        with open(filename, "w") as file:
            json.dump(results, file, indent=4)
    return results


def print_vels():
    vels = [1, 2]
    lengths = {}
    for n_robot in range(3, 11):
        lengths[n_robot] = best_velocity_combo(n_robot, vels)


def plot(filename, figname=None, log=False):
    fontsize = 32
    markersize = 12
    # Read the data.
    with open(filename, "r") as file:
        data = json.load(file)
    n_robots = np.array([int(k) for k in data.keys()])
    heights = np.array([v["height"] for v in data.values()])
    #
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(n_robots, heights, c="b", marker="o", markersize=markersize, lw=2)
    # Set up figure.
    ax.set_xticks(n_robots)
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.set_xlabel(r"$\#$ robots", fontsize=fontsize)
    ax.set_ylabel(r"Workspace height$|_{\mathrm{mm}}$", fontsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    # Add legend.
    handles, labels = [], []
    handles += [
        plt.plot(
            [],
            [],
            ls="",
            markerfacecolor="b",
            marker="o",
            markersize=markersize,
        )[0]
    ]
    labels += ["Data point"]
    ax.legend(
        handles=handles,
        labels=labels,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize=fontsize,
        framealpha=0.8,
        facecolor="w",
        handletextpad=0.01,
        labelspacing=0.05,
        borderpad=0.2,
        borderaxespad=0.2,
        loc="upper left",
    )
    if figname is not None:
        fig_name = os.path.join(os.getcwd(), f"{figname}.pdf")
        fig.savefig(fig_name, bbox_inches="tight", pad_inches=0.05)
    return fig, ax


def test_height():
    n_robots = [3, 4, 5, 6, 7, 8, 9, 10, 11]
    vels = [2, 3]
    filename = "scalability_height"
    start_time = time.time()
    find_heights(
        n_robots,
        vels,
        threshold=0.9,
        max_size=200000,
        height_lb_adaptive=True,
        height_ub_mult=2.0,
        tol_cmd=1e-2,
        goal_bias=0.05,
        spacing=1.5,
        dmin=5,
        clr=5,
        n=30,
        tol_sol=1.0,
        iter_max=1000,
        filename=filename,
    )
    end_time = time.time()
    runtime = end_time - start_time
    print(f"The total runtime is {runtime} seconds.")


########## test section ################################################
if __name__ == "__main__":
    # test()
    # plot("scalability_height.json", "scalability_height")
    plt.show()
