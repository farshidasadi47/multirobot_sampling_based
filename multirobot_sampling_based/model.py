# %%
########################################################################
# This files hold classes and functions that simulates the milirobot
# system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
from itertools import combinations
import os

import matplotlib  # Move this up so matplotlib.use can be called early

matplotlib.use("TkAgg")  # Set the backend before importing pyplot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.legend_handler import HandlerTuple

np.set_printoptions(precision=4, suppress=True)

plt.rcParams["figure.figsize"] = [7.25 * 1.25, 7.25]
plt.rcParams.update({"font.size": 11})
plt.rcParams["font.family"] = ["serif"]
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["text.usetex"] = False
plt.rcParams["hatch.linewidth"] = 0.5


########## Functions ###################################################
def round_down(arr, dec=2):
    m = 10**dec
    return np.floor(np.asarray(arr) * m) / m


def wrap_pi(angles):
    """Wraps angles between -PI to PI."""
    angles = np.atleast_1d(angles)
    wrapped = np.remainder(angles + np.pi, 2 * np.pi)
    wrapped[wrapped < 0] += 2 * np.pi
    wrapped -= np.pi
    return wrapped


def wrap_2pi(angles):
    """Wraps angles between 0 to 2PI."""
    angles = np.atleast_1d(angles)
    wrapped = np.remainder(angles, 2 * np.pi)
    wrapped[wrapped < 0] += 2 * np.pi
    return wrapped


def cartesian_to_polar(cartesian):
    """Converts 2D array of cartesian pints to polar points."""
    cartesian = np.atleast_2d(cartesian)
    polar = np.zeros_like(cartesian)
    polar[:, -1] = cartesian[:, -1]
    #
    x = cartesian[:, 0]
    y = cartesian[:, 1]
    polar[:, 0] = np.sqrt(x**2 + y**2)
    polar[:, 1] = np.arctan2(y, x)
    return polar


def msg_header(msg):
    N = len(msg)
    start_width = 10
    line_width = 79
    if N > 0:
        msg = " ".join(
            [
                "*" * start_width,
                msg,
                "*" * (line_width - (start_width + 2 + N)),
            ]
        )
    else:
        msg = "*" * line_width
    return msg


def define_colors(self):
    self._colors = {
        "k": (0, 0, 0),
        "r": (0, 0, 255),
        "b": (255, 0, 0),
        "lime": (0, 255, 0),
        "fuchsia": (255, 0, 255),
        "y": (0, 255, 255),
        "c": (255, 255, 0),
        "crimson": (220, 20, 60),
        "mediumpurple": (147, 112, 219),
        "darkorange": (255, 140, 0),
        "lightcoral": (240, 128, 128),
    }
    self._cmaps = {
        "k": "Greys",
        "r": "Reds",
        "b": "Blues",
        "lime": "YlGn",
        "fuchia": "RdPu",
        "y": "Wistia",
        "c": "PuBu",
        "crimson": "Oranges",
        "mediumpurple": "PuBuGn",
        "darkorange": "YlOrBr",
        "lightcoral": "YlOrRd",
    }
    self._markers = [
        "o",
        "s",
        "D",
        "v",
        "^",
        "X",
        "P",
        "p",
        "*",
        "d",
        "H",
        "x",
    ]
    self._styles = [
        ("solid", "solid"),
        ("dashed", (0, (5, 5))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("dashd7.4otdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("loosely dotted", (0, (1, 10))),
        ("solid", "solid"),
        ("dashed", (0, (5, 5))),
        ("dashdotted", (0, (3, 5, 1, 5))),
        ("dashd7.4otdotted", (0, (3, 5, 1, 5, 1, 5))),
        ("loosely dotted", (0, (1, 10))),
        ("solid", "solid"),
        ("dashed", (0, (5, 5))),
    ]


########## Classes #####################################################
class SwarmSpecs:
    """This class stores specifications of swarm of milirobots."""

    def __init__(
        self,
        pivot_length: np.array,
        tumbling_length,
        uncertainty_scaling,
        *,
        obstacles=[],
        obstacle_contours=[],
        dmin=20.0,
        clearance=15.0,
        theta_inc=np.deg2rad(5),
        alpha_inc=np.deg2rad(5),
        rot_inc=np.deg2rad(5),
        pivot_inc=np.deg2rad(5),
        tumble_inc=np.deg2rad(2),
        theta_sweep=np.deg2rad(30),
        alpha_sweep=np.deg2rad(33),
    ):
        msg = '"pivot_length" should be a 2D numpy array.'
        assert pivot_length.ndim == 2, msg
        msg = "Robots should have even number of sides."
        assert pivot_length.shape[1] % 2 == 0, msg
        msg = "uncertainty scaling matric should be 2D numpy array."
        assert uncertainty_scaling.ndim == 2, msg
        #
        self.n_mode = pivot_length.shape[1] + 1
        self.n_robot = pivot_length.shape[0]
        # Construct leg (tumbling) matrix.
        self.leg_length = np.hstack(
            (
                np.ones((self.n_robot, 1)) * tumbling_length,
                pivot_length.astype(float),
            )
        )
        self.tumbling_length = tumbling_length
        self.uncertainty_scaling = uncertainty_scaling
        # Construct displacement ratio matrix.
        self.beta = np.zeros_like(self.leg_length)
        # beta := pivot_seperation_robot_j/pivot_seperation_robot_0
        for robot in range(self.n_robot):
            self.beta[robot] = self.leg_length[robot] / self.leg_length[0]
        # Construct joint controllability matrix W from beta.
        self.W = np.kron(self.beta, np.eye(2))
        # Construct control matrix B by reshaping W.
        self.B = self.W.T.reshape((self.n_mode, 2, -1)).transpose((0, 2, 1))
        # Construct uncertainty matrix E.
        self.E = np.array([np.diag(x) for x in self.uncertainty_scaling])
        # Control gains.
        self.K = -np.reshape(np.linalg.pinv(self.W), (self.n_mode, 2, -1))
        # Next mode when odd number of tumbling.
        self.next_mode_odd_tumble = np.arange(self.n_mode)
        self.next_mode_odd_tumble[1:] = np.roll(
            self.next_mode_odd_tumble[1:], self.n_mode // 2
        )
        # Mode change angles and displacements.
        mode_rel_ang = (
            np.arange(self.n_mode - 1) * 2 * np.pi / (self.n_mode - 1)
        )
        mode_rel_length = (
            self.tumbling_length * np.sqrt(2 - 2 * np.cos(mode_rel_ang))
        ) / 2
        self.mode_rel_ang = np.zeros((self.n_mode, self.n_mode), dtype=float)
        self.mode_rel_length = np.zeros(
            (self.n_mode, self.n_mode), dtype=float
        )
        for ind, (row_ang, row_len) in enumerate(
            zip(self.mode_rel_ang[1:], self.mode_rel_length[1:])
        ):
            row_ang[1:] = np.roll(mode_rel_ang, ind)
            row_len[1:] = np.roll(mode_rel_length, ind)
        # Other parameters.
        self.robot_pairs = list(combinations(range(self.n_robot), 2))
        # Plotting and vision markers.
        define_colors(self)
        self._colors = list(self._colors.keys())
        self._cmaps = list(self._cmaps.values())
        # Experimental execution parameters.
        self.theta_inc = theta_inc
        self.alpha_inc = alpha_inc
        self.rot_inc = rot_inc
        self.pivot_inc = pivot_inc
        self.tumble_inc = tumble_inc
        self.theta_sweep = theta_sweep
        self.alpha_sweep = alpha_sweep
        # Data points for letters.
        self._set_letters()
        # Set default space.
        self.set_space(dmin=dmin, clearance=clearance)
        # Set obstacles.
        self.set_obstacles(
            obstacles=obstacles, obstacle_contours=obstacle_contours
        )
        self._set_obstacle_scenario()

    def set_space(
        self,
        *,
        ubx=115,
        lbx=-115,
        uby=90,
        lby=-90,
        rcoil=100,
        dmin=None,
        clearance=None,
    ):
        """
        Sets space boundaries available.
        ----------
        Parameters
        ----------
        ubx: x axis upper limit
        lbx: x axis lower limit
        uby: y axis upper limit
        lby: y axis lower limit
        rcoil: radius of small magnetic coil
        """
        self.dmin = self.dmin if dmin is None else dmin
        self.clearance = self.clearance if clearance is None else clearance
        self.xclearance = self.clearance
        self.yclearance = self.clearance
        # Space boundaries
        self.bounds = ((lbx, ubx), (lby, uby))
        self.rcoil = rcoil
        # Optimization boundaries.
        oubx = ubx - self.xclearance
        ouby = uby - self.yclearance
        olbx = lbx + self.xclearance
        olby = lby + self.yclearance
        self.obounds = ((olbx, oubx), (olby, ouby))
        self.orcoil = rcoil - self.clearance

    def set_obstacles(self, *, obstacles=[], obstacle_contours=[]):
        self.obstacles = obstacles
        self.obstacle_contours = obstacle_contours

    def _set_obstacle_scenario(self):
        space_half = [
            np.array(
                [[-5, -25], [-5, -100], [5, -100], [5, -25]], dtype=float
            ),
            np.array([[-5, 100], [-5, 25], [5, 25], [5, 100]], dtype=float),
        ]
        self._scenarios = {
            3: {0: [], 1: space_half},
            4: {0: [], 1: space_half},
            5: {0: [], 1: space_half},
            10: {
                0: [],
                1: space_half,
                2: [
                    np.array(
                        [[-75, 5], [-75, -200], [-65, -200], [-65, 5]],
                        dtype=float,
                    ),
                    np.array(
                        [[65, 200], [65, -5], [75, -5], [75, 200]], dtype=float
                    ),
                ],
            },
        }

    @property
    def scenarios(self):
        return self._scenarios.get(self.n_robot, 3)

    def set_obstacle_scenario(self, scenario=0):
        self.obstacle_contours = self._scenarios[self.n_robot].get(
            scenario, []
        )

    def _set_letters(self):
        chars3 = dict()
        chars3["*"] = {
            "poses": [-40, 0, 0, 0, 40, 0],
            "shape": [0, 1, 1, 2, 999, 999],
            "steps": 3,
        }
        chars3["b"] = {
            "poses": [+40, +40, 0, 0, 0, +40],
            "shape": [0, 2, 1, 2, 999, 999],
            "steps": 3,
        }
        chars3["c"] = {
            "poses": [-40, 0, -40, 40, 0, 0],
            "shape": [0, 1, 0, 2, 999, 999],
            "steps": 3,
        }
        chars3["d"] = {
            "poses": [0, 0, 0, -40, -40, -40],
            "shape": [0, 1, 1, 2, 999, 999],
            "steps": 3,
        }
        chars3["e"] = {
            "poses": [+40, -40, 0, 0, 40, 0],
            "shape": [0, 2, 1, 2, 999, 999],
            "steps": 3,
        }
        chars3["f"] = {
            "poses": [0, 0, +40, 0, -40, 0],
            "shape": [0, 1, 0, 2, 999, 999],
            "steps": 3,
        }
        #
        chars4 = dict()
        chars4["*"] = {
            "poses": [-45, 0, -15, 0, 15, 0, 45, 0],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["1"] = {
            "poses": [0, 60, 0, 20, 0, -20, 0, -60],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["7"] = {
            "poses": [-40, 40, -40, -40, 0, 0, 40, 40],
            "shape": [0, 3, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["D"] = {
            "poses": [-40, 40, -40, -40, 40, 20, 40, -20],
            "shape": [0, 1, 0, 2, 1, 3, 2, 3],
            "steps": 3,
        }
        chars4["I"] = {
            "poses": [0, 60, 0, 20, 0, -20, 0, -60],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["J"] = {
            "poses": [-40, -20, 0, -40, 40, -20, 40, 40],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["L"] = {
            "poses": [-40, 40, -40, 0, -40, -40, 30, -40],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["N"] = {
            "poses": [-40, 40, -40, -40, 40, 40, 40, -40],
            "shape": [0, 1, 0, 3, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["T"] = {
            "poses": [-40, 40, 0, 40, 0, -40, 40, 40],
            "shape": [0, 1, 1, 2, 1, 3, 999, 999],
            "steps": 3,
        }
        chars4["V"] = {
            "poses": [-40, 40, -20, 0, 0, -40, 40, 40],
            "shape": [0, 1, 1, 2, 2, 3, 999, 999],
            "steps": 3,
        }
        chars4["Y"] = {
            "poses": [0, 0, 0, -50, 30, 40, -30, 40],
            "shape": [0, 1, 0, 2, 0, 3, 999, 999],
            "steps": 3,
        }
        chars4["Z"] = {
            "poses": [-30, 40, -40, -40, 40, 40, 40, -40],
            "shape": [0, 2, 1, 2, 1, 3, 999, 999],
            "steps": 3,
        }
        #
        chars5 = dict()
        chars5["*"] = {
            "poses": [-60, 0, -30, 0, 0, 0, 30, 0, 60, 0],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 4, 5],
            "steps": 3,
        }
        chars5["A"] = {
            "poses": [0, 40, -15, 0, -30, -40, 30, -40, 15, 0],
            "shape": [0, 1, 0, 4, 1, 2, 1, 4, 3, 4],
            "steps": 5,
        }
        chars5["B"] = {
            "poses": [-30, -40, -30, 0, -30, 40, 20, 20, 30, -30],
            "shape": [0, 2, 0, 4, 1, 3, 1, 4, 2, 3],
            "steps": 3,
        }
        chars5["C"] = {
            "poses": [-30, 0, 0, 40, 40, 40, 0, -40, 40, -40],
            "shape": [0, 1, 0, 3, 1, 2, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["D"] = {
            "poses": [-40, 40, 0, 40, 30, 0, -40, -40, 0, -40],
            "shape": [0, 1, 0, 3, 1, 2, 2, 4, 3, 4],
            "steps": 3,
        }
        chars5["E"] = {
            "poses": [-30, -30, -30, 30, -10, 0, 30, 30, 30, -30],
            "shape": [0, 2, 0, 4, 1, 2, 1, 3, 999, 999],
            "steps": 4,
        }
        chars5["F"] = {
            "poses": [-30, 40, -30, 0, -30, -40, 15, 0, 30, 40],
            "shape": [0, 1, 0, 4, 1, 2, 1, 3, 999, 999],
            "steps": 3,
        }
        chars5["I"] = {
            "poses": [0, 60, 0, 30, 0, 0, 0, -30, 0, -60],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 4,
        }
        chars5["J"] = {
            "poses": [-30, -20, 0, -40, 30, -20, 30, 20, 30, 60],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["K"] = {
            "poses": [-30, -40, -30, 0, -30, 40, 30, 30, 30, -40],
            "shape": [0, 1, 1, 2, 1, 3, 1, 4, 999, 999],
            "steps": 4,
        }
        chars5["L"] = {
            "poses": [-35, 50, -35, 0, -35, -50, 0, -50, 35, -50],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["M"] = {
            "poses": [-50, -40, -30, 40, 0, 0, 30, 40, 50, -40],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["N"] = {
            "poses": [-30, -40, -30, 40, 0, 0, 30, 40, 30, -40],
            "shape": [0, 1, 1, 2, 2, 4, 3, 4, 999, 999],
            "steps": 5,
        }
        chars5["O"] = {
            "poses": [-40, 40, -40, -10, 0, -40, 40, -10, 40, 40],
            "shape": [0, 1, 0, 4, 1, 2, 2, 3, 3, 4],
            "steps": 4,
        }
        chars5["P"] = {
            "poses": [-30, 40, -30, 0, -30, -40, 10, 0, 10, 40],
            "shape": [0, 1, 0, 4, 1, 2, 1, 3, 3, 4],
            "steps": 4,
        }
        chars5["R"] = {
            "poses": [-30, -40, -30, 0, -30, 40, 20, 20, 30, -40],
            "shape": [0, 1, 1, 2, 1, 3, 1, 4, 2, 3],
            "steps": 4,
        }
        chars5["S"] = {
            "poses": [-30, 30, 0, 50, 30, 30, -30, -40, 30, -40],
            "shape": [0, 1, 0, 4, 1, 2, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["T"] = {
            "poses": [-35, 35, 0, 35, 35, 35, 0, 0, 0, -35],
            "shape": [0, 1, 1, 2, 1, 3, 3, 4, 999, 999],
            "steps": 4,
        }
        chars5["U"] = {
            "poses": [-30, -30, 0, -50, 30, -30, -30, 30, 30, 30],
            "shape": [0, 1, 0, 3, 1, 2, 2, 4, 999, 999],
            "steps": 3,
        }
        chars5["V"] = {
            "poses": [-40, 40, -20, 0, 0, -40, 20, 0, 40, 40],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["W"] = {
            "poses": [-50, 40, -30, -40, 0, 0, 30, -40, 50, 40],
            "shape": [0, 1, 1, 2, 2, 3, 3, 4, 999, 999],
            "steps": 3,
        }
        chars5["X"] = {
            "poses": [-30, -40, -30, 40, 0, 0, 30, 40, 30, -40],
            "shape": [0, 2, 1, 2, 2, 3, 2, 4, 999, 999],
            "steps": 5,
        }
        chars5["Y"] = {
            "poses": [-35, 40, 0, 20, 0, -10, 0, -40, 35, 40],
            "shape": [0, 1, 1, 2, 2, 3, 1, 4, 999, 999],
            "steps": 4,
        }
        chars5["Z"] = {
            "poses": [-30, -40, -30, 40, 0, 0, 30, 40, 30, -40],
            "shape": [0, 2, 0, 4, 1, 3, 2, 3, 999, 999],
            "steps": 5,
        }
        #
        chars = {3: chars3, 4: chars4, 5: chars5}
        self.chars = None
        if self.n_robot in chars.keys():
            self.chars = chars.get(self.n_robot, 3)
            for char_dic in self.chars.values():
                char_dic["poses"] = np.array(char_dic["poses"], dtype=float)

    def get_letter(self, char, ang=0, roll=0):
        chars = self.chars
        scale = 1.0
        roll = roll % self.n_robot
        ang = np.deg2rad(ang)
        poses, shape, steps = chars.get(char, chars["*"]).values()
        poses = scale * poses[: 2 * self.n_robot]
        # Rolling robots position in the pattern
        poses = np.roll(poses, 2 * roll)
        shape = [
            (elem + roll) % self.n_robot if elem < 999 else elem
            for elem in shape
        ]
        # Rotation
        rot = np.array(
            [[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]]
        )
        poses = np.dot(rot, poses.reshape(-1, 2).T).T.flatten()
        return poses, shape, steps

    @classmethod
    def robo3(cls):
        pivot_length = np.array([[9, 7], [7, 5], [5, 9]])
        uncertainty_scaling = 0.2 * np.ones((3, 2 * 3))
        return cls(pivot_length, 10, uncertainty_scaling)

    @classmethod
    def robo4(cls):
        pivot_length = np.array(
            [[9, 7, 5, 3], [7, 5, 3, 9], [5, 3, 9, 7], [3, 9, 7, 5]]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 4))
        return cls(pivot_length, 10, uncertainty_scaling)

    @classmethod
    def robo5(cls):
        pivot_length = np.array(
            [
                [10, 6, 8, 6],
                [6, 8, 6, 10],
                [8, 6, 10, 10],
                [6, 10, 10, 6],
                [10, 10, 6, 8],
            ]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 5))
        return cls(pivot_length, 10, uncertainty_scaling)

    @classmethod
    def robo10(cls):
        pivot_length = np.array(
            [
                [2, 2, 3, 3, 3, 2, 3, 3, 2, 2],
                [2, 3, 3, 3, 2, 3, 3, 2, 2, 2],
                [3, 3, 3, 2, 3, 3, 2, 2, 2, 2],
                [3, 3, 2, 3, 3, 2, 2, 2, 2, 3],
                [3, 2, 3, 3, 2, 2, 2, 2, 3, 3],
                [2, 3, 3, 2, 2, 2, 2, 3, 3, 3],
                [3, 3, 2, 2, 2, 2, 3, 3, 3, 2],
                [3, 2, 2, 2, 2, 3, 3, 3, 2, 3],
                [2, 2, 2, 2, 3, 3, 3, 2, 3, 3],
                [2, 2, 2, 3, 3, 3, 2, 3, 3, 2],
            ],
            dtype=float,
        )
        uncertainty_scaling = np.zeros((11, 10 * 2))
        specs = cls(pivot_length, 1, uncertainty_scaling)
        specs.set_space(
            lbx=-150,
            ubx=150,
            lby=-100,
            uby=100,
            rcoil=300,
            dmin=5.0,
            clearance=5.0,
        )
        return specs

    @classmethod
    def robo3p(cls):
        pivot_length = np.array([[8.46, 6.67], [6.61, 4.75], [4.79, 8.18]])
        uncertainty_scaling = 0.2 * np.ones((3, 2 * 3))
        return cls(pivot_length, 11.44, uncertainty_scaling)

    @classmethod
    def robo4p(cls):
        pivot_length = np.array(
            [
                [7.48, 4.91, 4.90, 4.93],
                [4.95, 4.95, 4.91, 7.59],
                [4.85, 4.87, 7.42, 4.89],
                [4.86, 7.44, 4.82, 4.84],
            ]
        )
        uncertainty_scaling = np.array(
            [
                [1.55, 1.26, 1.07, 1.39, 1.02, 1.40, 1.16, 1.08],
                [2.60, 2.66, 3.98, 3.47, 3.93, 3.68, 4.17, 3.74],
                [3.19, 3.26, 4.24, 3.94, 3.84, 3.74, 2.53, 2.89],
                [3.93, 3.61, 4.85, 5.09, 2.69, 2.12, 4.40, 3.80],
                [4.11, 3.79, 3.67, 2.96, 4.71, 5.16, 5.07, 4.17],
            ]
        )
        return cls(pivot_length, 12.55, uncertainty_scaling)

    @classmethod
    def robo5p(cls):
        pivot_length = np.array(
            [
                [4.80, 8.00, 8.00, 5.00],
                [7.90, 8.00, 4.90, 5.00],
                [7.90, 4.90, 4.90, 5.10],
                [4.90, 4.90, 5.00, 8.00],
                [4.80, 4.90, 8.10, 8.20],
            ]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 5))
        return cls(pivot_length, 15.24, uncertainty_scaling)

    @classmethod
    def robo(cls, n_robot):
        robots = {3: "robo3p", 4: "robo4p", 5: "robo5p"}
        return getattr(cls, robots.get(n_robot, "robo3p"))()


class WorkSpace:
    """
    This class creates mesh representation of workspace and obstacles
    from given dimensions and external contour of obstacles.
    """

    def __init__(
        self,
        obstacle_contours=[],
        lbx=-115,
        ubx=115,
        lby=-90,
        uby=90,
        rcoil=100,
        with_coil=True,
    ):
        self.bounds = (
            np.array((lbx, ubx), dtype=float),
            np.array((lby, uby), dtype=float),
        )
        self.rcoil = rcoil
        self.with_coil = with_coil


class WorkSpaceImg(WorkSpace):
    """
    This class creates mesh representation of workspace and obstacles
    from given image and given scale.
    """

    def __init__(self, img, rcoil=100, with_coil=True, scale=0.48970705):
        self.img = img
        self.scale = scale  # mm = pix * scale
        self._set_bounds()
        self.bounds = self._cartesian_bounds()
        self.rcoil = rcoil
        self.with_coil = with_coil
        eps = 0.1

    def cart2pix(self, xy):
        xy = np.array(xy, dtype=float)
        xy = np.atleast_2d(xy)
        pix = np.zeros_like(xy, dtype=int)  # [[x_pix, y_pix], ...].
        pix[:, 0] = np.clip(xy[:, 0] / self.scale + self.center[0], 0, self.w)
        pix[:, 1] = np.clip(self.center[1] - xy[:, 1] / self.scale, 0, self.h)
        return pix

    def pix2cart(self, pix):
        pix = np.array(pix, dtype=int)
        pix = np.atleast_2d(pix)
        xy = np.zeros_like(pix, dtype=float)
        xy[:, -1] = -pix[:, -1]  # Angle, if present.
        xy[:, 0] = (pix[:, 0] - self.center[0]) * self.scale  # x coord.
        xy[:, 1] = (self.center[1] - pix[:, 1]) * self.scale  # y coord.
        return xy

    def _set_bounds(self):
        self.h, self.w = self.img.shape[:2]
        self.center = np.array([self.w / 2, self.h / 2], dtype=int)

    def _cartesian_bounds(self):
        half_width, half_height = self.pix2cart([self.w, self.h])[0]
        ubx = half_width
        uby = -half_height
        return np.array((-ubx, ubx)), np.array((-uby, uby))


class Simulation:
    """This class simulates and visualize given input."""

    def __init__(self, specs: SwarmSpecs, fontsize=20):
        self._obs_color = "slategray"
        self._fs_title = fontsize
        self._fs_axlable = 0.7 * self._fs_title
        self._fs_legend = 0.7 * self._fs_title
        self.specs = specs
        # Simulation variables.
        self._sim_mode = None
        self._sim_result = None

    def update_state(self, pos: np.ndarray, cmd: np.ndarray, stochastic=False):
        """
        This function updates the position, angle, and mode of swarm
        of milirobots based on given input.
        ----------
        Parameters
        ----------
        pos: numpy.ndarray
            Current position
        cmd: numpy.ndarray
            A numpy array as [r, phi, cmd_mode]
            r:    Distance to travel
            phi:  Angle to travel
            mode: Mode to travel
        mode: int, default=1
            Current mode of system.
        """
        n_state = self.specs.n_robot * 2
        r, phi, mode = cmd
        mode = int(mode)
        # Modify input based on requested motion mode.
        if mode < 0:
            mode = -mode
            # Mode change, r is irrelevant in this case.
            r = self.specs.mode_rel_length[self._sim_mode, mode]
            B = self.specs.B[0]
            E = np.zeros((n_state, n_state))
            self._sim_mode = mode
        elif mode == 0:
            # Tumbling, modify input based on tumbling distance.
            tumbles = np.round(r / self.specs.tumbling_length).astype(int)
            # r = tumbles * self.specs.tumbling_length
            B = self.specs.B[mode]
            E = self.specs.E[mode]
            # Update mode if needed.
            if tumbles % 2:
                # Odd number of tumblings.
                self._sim_mode = self.specs.next_mode_odd_tumble[
                    self._sim_mode
                ]
        else:
            # Pivot walking
            B = self.specs.B[mode]
            E = self.specs.E[mode]
            self._sim_mode = mode
        # Convert input to cartesian.
        cmd_x = r * np.cos(phi)
        cmd_y = r * np.sin(phi)
        next_pos = (
            pos
            + np.dot(B, [cmd_x, cmd_y])
            + np.dot(E, np.random.normal(size=n_state)) * stochastic
        )
        return next_pos

    def simulate(self, cmds, pos, mode=None):
        """Simulates the swarm for a given logical series of input."""
        cmds = np.atleast_2d(cmds)
        if mode is None:
            mode = int(abs(cmds[0, 2]))
        self._sim_mode = mode
        #
        poses = np.zeros((len(cmds), 2, self.specs.n_robot * 2))
        for i, cmd in enumerate(cmds):
            poses[i, 0] = pos
            pos = self.update_state(pos, cmd)
            poses[i, 1] = pos
        self._sim_result = (poses, cmds)
        return poses, cmds

    def simulate_closed_loop(
        self, plan, xi, xf, mode, max_error, max_iteration
    ):
        """
        Simulates the system using the provided receding horison planner.
        """
        self._sim_mode = mode
        cmds = []
        poses = []
        for iteration in range(max_iteration):
            msg = f"Performing step: {iteration + 1:3d}"
            print(msg_header(msg))
            poss = [xi]
            cmd = plan(xi, xf)
            xi = self.update_state(xi, cmd, stochastic=True)
            poss.append(xi)
            cmds.append(cmd)
            poses.append(poss)
            error = xi - xf
            print(f"Error at step {iteration + 1: 2d}: {poses[-1][-1] - xf}")
            if np.linalg.norm(error, np.inf) < max_error:
                break
        #
        cmds = np.array(cmds)
        poses = np.array(poses)
        self._sim_result = (poses, cmds)
        return poses, cmds

    def _trim_path(self, poses, cmds, length, step):
        sec_length = np.ceil(cmds[:, 0] / step).astype(int)
        sec_length = np.where(sec_length, sec_length, 1)
        cum_length = np.cumsum(sec_length)
        if length >= cum_length[-1]:
            return poses
        ind = np.sum(cum_length < length).item()
        if ind:
            base_length = cum_length[ind - 1]
        else:
            base_length = 0
        time_fraction = (length - base_length) / sec_length[ind]
        # Trim path.
        poses = poses[: ind + 1].copy()
        poses[-1][-1] = poses[-1][0] + time_fraction * (
            poses[-1][1] - poses[-1][0]
        )
        return poses

    def _simplot_set(self, ax, boundary=False):
        """Sets the plot configuration."""
        ((lbx, ubx), (lby, uby)) = self.specs.bounds
        if boundary is True:
            ax.set_ylim([lby, uby])
            ax.set_xlim([lbx, ubx])
            # Draw usable space boundaries
            ((lbx, ubx), (lby, uby)) = self.specs.bounds
            rectangle = plt.Rectangle(
                [
                    lbx + self.specs.xclearance,
                    lby + self.specs.yclearance,
                ],
                ubx - lbx - self.specs.xclearance * 2,
                uby - lby - self.specs.yclearance * 2,
                linestyle="--",
                linewidth=1,
                edgecolor="k",
                facecolor="none",
                zorder=1,
            )
            ax.add_patch(rectangle)
            """ coil = plt.Circle(
                (0, 0),
                radius=self.specs.rcoil,
                linestyle="--",
                linewidth=1,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(coil) """
        ax.set_xlabel("x axis", fontsize=self._fs_axlable)
        ax.set_ylabel("y axis", fontsize=self._fs_axlable)
        ax.xaxis.set_tick_params(labelsize=self._fs_axlable)
        ax.yaxis.set_tick_params(labelsize=self._fs_axlable)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.4)

    def _simplot_title(self, ax, i, i_tot, cmd_mode):
        if cmd_mode < 0:
            text = f"Step {i:3d} of {i_tot:3d}, changing to mode {-cmd_mode}"
        elif cmd_mode < 1:
            text = f"Step {i:3d} of {i_tot:3d}, tumbling, mode {cmd_mode}"
        else:
            text = f"Step {i:3d} of {i_tot:3d}, pivot-walking, mode {cmd_mode}"
        ax.set_title(text, fontsize=self._fs_title, pad=self._fs_title / 3)

    def _draw_obstacles(self, ax, circles=True):
        # Draw obstacle contours.
        for cnt in self.specs.obstacle_contours:
            polygon = Polygon(cnt, ec="k", lw=1, fc=self._obs_color, alpha=1.0)
            ax.add_patch(polygon)
        if circles:
            # Draw obstacles.
            for obstacle in self.specs.obstacles:
                circle = plt.Circle(
                    obstacle[:2],
                    radius=obstacle[-1],
                    linestyle="--",
                    linewidth=1,
                    edgecolor="k",
                    facecolor=self._obs_color,
                )
                ax.add_patch(circle)

    def _draw_robots(self, ax, pos, cmd_mode, start=True, single=False):
        """Draws robts circle in given position."""
        cmd_mode = int(cmd_mode)
        if cmd_mode < 0:
            cmd_mode = 0
        for robot in range(self.specs.n_robot):
            if start:
                ls = "--"
                lw = 1.0 if single else 0.5
            else:
                ls = None
                lw = 1.5 if single else 1.0
            ax.plot(
                pos[2 * robot],
                pos[2 * robot + 1],
                lw=1,
                c="k" if single else self.specs._colors[cmd_mode],
                marker=self.specs._markers[robot],
                mfc="none",
            )
            # Draw circle bounding the robots
            circle = plt.Circle(
                pos[2 * robot : 2 * robot + 2],
                radius=self.specs.dmin / 2,
                ls=ls,
                lw=lw,
                edgecolor="k",
                facecolor="none",
            )
            ax.add_patch(circle)

    def _draw_legend(self, ax):
        # Get the size of one point in data coordinates
        legend_obs = lambda: plt.Circle(
            (0, 0),
            radius=self.specs.dmin,
            edgecolor="k",
            facecolor=self._obs_color,
            alpha=0.8,
            lw=1,
        )
        legend_marker = lambda m, l: plt.plot(
            [], [], marker=m, color="k", mfc="none", ms=6, ls="none"
        )[0]
        legend_bc = lambda ls, lw: plt.scatter(
            [],
            [],
            marker="o",
            s=(15) ** 2,
            facecolor="none",
            edgecolor="k",
            linestyle=ls,
            linewidth=lw,
        )
        handles = [legend_obs()] if len(self.specs.obstacle_contours) else []
        labels = ["Obs"] if len(self.specs.obstacle_contours) else []
        # Add legend for robots.
        handles += [
            legend_marker(self.specs._markers[robot], f"R{robot:1d}")
            for robot in range(self.specs.n_robot)
        ]
        labels += [f"R{robot:1d}" for robot in range(self.specs.n_robot)]
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize=self._fs_legend,
            framealpha=0.8,
            facecolor="w",
            handletextpad=0.15,
            borderpad=0.15,
            borderaxespad=0.2,
        )

    def _draw_section(self, ax, poss, cmd_mode):
        if cmd_mode < 0:
            cmd_mode = 0
        marker = True
        for pos_i, pos_f in zip(poss[:-1], poss[1:]):
            for robot in range(self.specs.n_robot):
                ax.plot(
                    (pos_i[2 * robot], pos_f[2 * robot]),
                    ((pos_i[2 * robot + 1], pos_f[2 * robot + 1])),
                    lw=1,
                    c=self.specs._colors[cmd_mode],
                    marker=self.specs._markers[robot] if marker else None,
                    markevery=[0],
                    mfc="none",
                )
            marker = False

    def _simplot_plot(
        self, ax, plot_length, step, title=True, last_section=False
    ):
        """Plots the result of simulation for the given length."""
        # Draw obstacles.
        self._draw_obstacles(ax, circles=True)
        # Get the simulation results.
        poses, cmds = self._sim_result
        total_sections = len(cmds)
        # Draw initial positions and hold it.
        counter = 0
        self._draw_robots(ax, poses[0, 0], cmds[0, 2], start=True)
        # Trim path.
        poses_dense = self._trim_path(poses, cmds, plot_length, step)
        # Determine sections to draw.
        if last_section:
            idxs = [len(poses_dense) - 1]
        else:
            idxs = range(len(poses_dense))
        # Draw the path.
        for poss, cmd in zip(poses_dense[idxs], cmds[idxs]):
            # Get and adjust command mode.
            cmd_mode = int(cmd[2])
            #
            self._draw_section(ax, poss, cmd_mode)
        # Plot robots at the end stage.
        self._draw_robots(ax, poss[-1], cmd_mode, start=False)
        # Set title.
        if title:
            self._simplot_title(ax, idxs[-1] + 1, total_sections, cmd_mode)
        # ax.legend(handlelength=0, loc="upper right")
        self._draw_legend(ax)
        return ax

    def simplot(
        self,
        step=1000,
        plot_length=10000,
        title=True,
        boundary=False,
        last_section=False,
        file_name=None,
    ):
        """Plots the swarm motion for a given logical series of input."""
        _, cmds = self._sim_result
        cmds = np.atleast_2d(cmds)
        # Set the figure properties
        fig, ax = plt.subplots(constrained_layout=True)
        self._simplot_set(ax, boundary)
        # plot the figure
        self._simplot_plot(ax, plot_length, step, title, last_section)
        # Saving figure if requested.
        if file_name:
            file_name = os.path.join(os.getcwd(), f"{file_name}_simplot.pdf")
            fig.savefig(file_name, bbox_inches="tight", pad_inches=0.05)
        return fig, ax

    def _animate(self, i, ax, step, title, boundary, last_section):
        ax.clear()
        self._simplot_set(ax, boundary)
        self._simplot_plot(ax, i, step, title, last_section)
        return ax

    def simanimation(
        self,
        vel=30.0,
        anim_length=10000,
        title=True,
        boundary=False,
        last_section=False,
        file_name=None,
    ):
        """This function produces an animation from swarm transition
        for a given logical input series and specified length."""
        _, cmds = self._sim_result
        cmds = np.atleast_2d(cmds)
        """if step is None:
            step = self.specs.tumbling_length"""
        fps = 10
        interval = 1000 // fps
        fps = 1000 / interval
        step = vel / fps
        n_steps = int(
            np.ceil(np.linalg.norm(cmds[cmds[:, 2] >= 0, 0], ord=1) / step)
        )
        n_steps = n_steps + np.count_nonzero((cmds[:, 2] < 0)) + 1
        n_steps = np.ceil(cmds[:, 0] / step).astype(int)
        n_steps = np.where(n_steps, n_steps, 1)
        n_steps = np.sum(n_steps).item() + 1
        anim_length = min(anim_length, n_steps)
        # Set the figure properties
        dpi = 100
        fig, ax = plt.subplots(layout="tight")
        fig.set_size_inches(19.2, 10.8)
        fig.set_dpi(dpi)
        self._simplot_set(ax, boundary)
        # fig.tight_layout(pad=0.05)
        # Animating
        anim = animation.FuncAnimation(
            fig,
            self._animate,
            fargs=(ax, step, title, boundary, last_section),
            interval=interval,
            frames=range(anim_length + 1 + int(1 * fps)),
            # repeat=False,
        )
        # Saving animation.
        if file_name:
            file_name = os.path.join(os.getcwd(), f"{file_name}.mp4")
            anim.save(
                file_name,
                fps=fps,
                writer="ffmpeg",
                codec="libx264",
                bitrate=10000,
                extra_args=["-crf", "0"],
            )
        plt.show(block=False)
        plt.pause(0.01)
        return anim

    def _make_dense(self, poses, cmds, step):
        """Breaks path to shorter steps."""
        poses_dense = [poses[0, :1]]
        cmd_modes_dense = [(cmds[0, 2],)]
        for poss, cmd in zip(poses, cmds):
            n_steps = np.ceil(cmd[0] / step).astype(int)
            dense = np.linspace(poss[0], poss[1], n_steps + 1)[1:]
            poses_dense.append(dense)
            cmd_modes_dense.append(np.ones(len(dense)) * cmd[2])
        return np.vstack(poses_dense), np.hstack(cmd_modes_dense).astype(int)

    def _legend_single(self, ax):
        # Get the size of one point in data coordinates
        legend_marker = lambda m, l: plt.plot(
            [], [], marker=m, color="k", mfc="none", ms=6, ls="none"
        )[0]
        legend_bc = lambda ls, lw: plt.Circle(
            (0, 0),
            radius=self.specs.dmin / 2,
            color="k",
            ls=ls,
            lw=lw,
            fill=False,
        )
        legend_bc = lambda ls, lw: plt.scatter(
            [],
            [],
            marker="o",
            s=(15) ** 2,
            facecolor="none",
            edgecolor="k",
            linestyle=ls,
            linewidth=lw,
        )
        handles = [legend_bc("--", 1.0), legend_bc("-", 1.5)]
        labels = ["Start", "Goal"]
        # Add legend for robots.
        handles += [
            legend_marker(self.specs._markers[robot], f"R{robot:1d}")
            for robot in range(self.specs.n_robot)
        ]
        labels += [f"R{robot:1d}" for robot in range(self.specs.n_robot)]
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize=self._fs_legend,
            framealpha=0.8,
            facecolor="w",
            # handlelength= handlelength,
            handletextpad=0.15,
            # labelspacing= 0.01,
            borderpad=0.15,
            borderaxespad=0.2,
        )

    def plot_single(self, boundary=True, file_name=None):
        """Plots the swarm motion for a given logical series of input."""
        poses, cmds = self._sim_result
        cmds = np.atleast_2d(cmds)
        # Set the figure properties
        fig, ax = plt.subplots()  # constrained_layout=True)
        self._simplot_set(ax, boundary)
        # Draw obstacles.
        self._draw_obstacles(ax, circles=True)
        # Draw start and finish positions.
        self._draw_robots(ax, poses[0, 0], cmds[0, 2], start=True, single=True)
        self._draw_robots(
            ax, poses[-1, -1], cmds[-1, 2], start=False, single=True
        )
        # Dense the points and make line segments.
        poses_dense, cmd_modes_dense = self._make_dense(
            poses, cmds, self.specs.tumbling_length
        )
        segments = np.array([poses_dense[:-1], poses_dense[1:]]).transpose(
            1, 0, 2
        )
        #
        cnorm = mcolors.Normalize(
            vmin=0,
            vmax=100,
        )
        cmap = plt.get_cmap("plasma_r")
        lc = LineCollection(segments[:, :, :2], cmap=cmap, norm=cnorm)
        lc.set_array(np.linspace(0, 100, len(segments)))
        ax.add_collection(lc)
        # Add and adjust colorbar.
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad=0.04)
        cbar = fig.colorbar(lc, cax=cax, orientation="vertical")
        cbar.set_label(
            "% Progression",
            fontsize=self._fs_axlable,
            labelpad=-self._fs_title,
        )
        cbar.ax.tick_params(labelsize=self._fs_axlable)
        cbar.set_ticks([0, 100])
        self._legend_single(ax)
        if file_name:
            file_name = os.path.join(os.getcwd(), f"{file_name}_single.pdf")
            fig.savefig(file_name, bbox_inches="tight", pad_inches=0.05)

    def _legend_selected(self, ax):
        # Get the size of one point in data coordinates
        legend_marker = lambda m, c: plt.plot(
            [], [], marker=m, color=c, ms=6, ls="none"
        )[0]
        # Add legend for robots.
        handles = [
            legend_marker(
                self.specs._markers[robot], self.specs._colors[robot]
            )
            for robot in range(self.specs.n_robot)
        ]
        labels = [f"R{robot:01d}" for robot in range(self.specs.n_robot)]
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize=self._fs_legend,
            framealpha=0.8,
            facecolor="w",
            # handlelength= handlelength,
            handletextpad=0.15,
            # labelspacing= 0.01,
            borderpad=0.15,
            borderaxespad=0.2,
        )

    @staticmethod
    def custom_cmap(base_color):
        # Light version of the base color
        light_color = np.array(plt.cm.colors.to_rgba(base_color))
        light_color[0:3] = 1.0 - 0.1 * (1.0 - light_color[0:3])
        # Dark version of the base color
        dark_color = np.array(plt.cm.colors.to_rgba(base_color))
        dark_color[0:3] *= 1.0
        # Create a custom colormap
        return LinearSegmentedColormap.from_list(
            "custom_cmap", [light_color, dark_color]
        )

    def _draw_robots_selected(
        self,
        ax,
        pos,
        cmaps,
        cmap_range,
        start=True,
    ):
        """Draws robts circle in given position."""
        for robot in range(self.specs.n_robot):
            cmap = cmaps[robot]
            ax.plot(
                pos[2 * robot],
                pos[2 * robot + 1],
                lw=1,
                marker=self.specs._markers[robot],
                c=cmap(cmap_range[0]) if start else cmap(1.0),
                mec=None if start else "k",
                mew=None if start else 2,
                ms=9,
                zorder=4,
            )

    def plot_selected(self, robots, boundary=True, file_name=None):
        poses, cmds = self._sim_result
        cmds = np.atleast_2d(cmds)
        # Set the figure properties
        fig, ax = plt.subplots()  # constrained_layout=True)
        self._simplot_set(ax, boundary)
        # Draw obstacles.
        self._draw_obstacles(ax, circles=True)
        cmaps = [
            self.custom_cmap(self.specs._colors[i])
            for i in range(self.specs.n_robot)
        ]
        # cmaps = [plt.get_cmap(name) for name in self.specs._cmaps]
        # Draw start and end.
        self._draw_robots_selected(
            ax, poses[0, 0], cmaps, (0.5, 1.0), start=True
        )
        # Dense the points and make line segments.
        poses_dense, cmd_modes_dense = self._make_dense(
            poses, cmds, self.specs.tumbling_length
        )
        segments = np.array([poses_dense[:-1], poses_dense[1:]]).transpose(
            1, 0, 2
        )
        cmd_modes_dense = np.where(cmd_modes_dense < 0, 0, cmd_modes_dense)
        styles = [
            self.specs._styles[cmd_mode][1] for cmd_mode in cmd_modes_dense
        ]
        #
        cnorm = mcolors.Normalize(vmin=0, vmax=100, clip=True)
        for i, robot in enumerate(robots):
            cmap = cmaps[robot]
            lc = LineCollection(
                segments[:, :, 2 * robot : 2 * robot + 2],
                cmap=cmap,
                norm=cnorm,
                # ls=styles,
                joinstyle="miter",
            )
            lc.set_array(np.linspace(15, 75, len(segments)))
            ax.add_collection(lc)
        self._draw_robots_selected(
            ax, poses[-1, -1], cmaps, (0.5, 1.0), start=False
        )
        self._legend_selected(ax)
        if file_name:
            file_name = os.path.join(os.getcwd(), f"{file_name}_selected.pdf")
            fig.savefig(file_name, bbox_inches="tight", pad_inches=0.05)

    def _draw_robots_simselected(self, ax, pos, cmd_mode, start=True):
        """Draws robts circle in given position."""
        cmd_mode = int(cmd_mode)
        if cmd_mode < 0:
            cmd_mode = 0
        for robot in range(self.specs.n_robot):
            ax.plot(
                pos[2 * robot],
                pos[2 * robot + 1],
                lw=1,
                c=self.specs._colors[cmd_mode],
                marker=self.specs._markers[robot],
                # mfc="none",
            )

    def simplot_selected(self, robots, boundary=True, file_name=None):
        poses, cmds = self._sim_result
        cmds = np.atleast_2d(cmds)
        # Set the figure properties
        fig, ax = plt.subplots()  # constrained_layout=True)
        self._simplot_set(ax, boundary)
        # Draw obstacles.
        self._draw_obstacles(ax, circles=True)
        # Draw start and end.
        self._draw_robots_simselected(ax, poses[0, 0], cmds[0, 2], start=True)
        self._draw_robots_simselected(
            ax, poses[-1, -1], cmds[-1, 2], start=False
        )
        self._draw_legend(ax)
        # Dense the points and make line segments.
        for poss, cmd in zip(poses, cmds):
            cmd_mode = int(cmd[2])
            if cmd_mode < 0:
                cmd_mode = 0
            for i, robot in enumerate(robots):
                ax.plot(
                    poss[:, 2 * robot],
                    poss[:, 2 * robot + 1],
                    lw=1,
                    c=self.specs._colors[cmd_mode],
                    marker=self.specs._markers[robot],
                    markevery=[0],
                    mfc="none",
                )
        if file_name:
            file_name = os.path.join(
                os.getcwd(), f"{file_name}_simselected.pdf"
            )
            fig.savefig(file_name, bbox_inches="tight", pad_inches=0.05)


def test_simulation():
    specs = SwarmSpecs.robo3()
    obstacles = np.array(
        [
            [0 * 30.0, 66.0, 5],
            [0 * 30.0, 37.0, 5],
            [0 * -30.0, -66.0, 5],
            [0 * -30.0, -37.0, 5],
        ],
        dtype=float,
    )
    obstacle_contours = [
        np.array([[-5, -25], [-5, -90], [5, -90], [5, -25]], dtype=float),
        np.array([[-5, 90], [-5, 25], [5, 25], [5, 90]], dtype=float),
    ]
    specs.set_obstacles(
        obstacles=obstacles, obstacle_contours=obstacle_contours
    )
    #
    simulation = Simulation(specs)
    pos = [-20.0, 0.0, 0.0, 0.0, 20.0, 0.0]
    cmds = [
        [50, np.radians(90), 1],
        [50, np.radians(180), 2],
        [10, np.radians(270), -1],
        [30, np.radians(-45), 0],
    ]
    # Run simulation.
    # Simulate the system
    simulation.simulate(cmds, pos)
    # Draw simulation results.
    file_name = None
    simulation.plot_selected([0, 2], file_name=file_name)
    simulation.simplot_selected([0, 2], file_name=file_name)
    simulation.plot_single(file_name=file_name)
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        anim_length=1100,
        vel=30,
        title=True,
        boundary=True,
        last_section=True,
    )


def test_simulation10():
    # Build specs of robots and obstacles.
    specs = SwarmSpecs.robo10()
    obstacle_contours = [
        np.array([[-45, 5], [-45, -200], [-55, -200], [-55, 5]], dtype=float),
        np.array([[55, 200], [55, -5], [45, -5], [45, 200]], dtype=float),
    ]
    specs.set_obstacles(obstacle_contours=obstacle_contours)
    #
    simulation = Simulation(specs)
    pose = np.array(
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
    cmds = np.array(
        [
            [40.0,                 3.141592653589793,    0.0],
            [6.856422034651185,    0.7628293103291118,   1.0],
            [4.799999084472653,   -3.1415926476293277,   0.0],
            [0.20569250724993743, -2.3787626497488414,   1.0],
            [4.151068007440625,   -0.7208154026912864,   2.0],
            [18.571782349229544,   0.14513497650797944,  3.0],
            [12.178693191809739,  -0.628991789660273,    4.0],
            [4.553598735253524,    2.910111193019997,    5.0],
            [3.901655767268735,   -3.1415918205788196,   6.0],
            [5.85248219439389,    -3.141590500666528,    7.0],
            [5.852472414885069,    3.14159197114522,     8.0],
            [3.9016459184133283,   3.141592068916295,    9.0],
            [3.901644253819331,   -3.141591673919066,   10.0],
            [6.6021102314025795,   1.7466301652414196,   1.0],
            [0.07535865328247249,  2.7695127701398494,   2.0],
            [13.90225471923896,   -1.2055658486482186,   3.0],
            [5.177422548762381,    2.7134599042066077,   4.0],
            [0.07052540791473136, -2.5074088040438705,   5.0],
            [11.38601836137262,    1.1948717611943636,   6.0],
            [3.1928804168701688,   3.141592479753514,    0.0],
            [1.0738666457128327,  -2.1685485362377928,   1.0],
            [0.8981300357317582,   0.7640200141265421,   2.0],
            [1.5478935676423597,   2.8024402339480825,   3.0],
            [21.205180549394,     -0.919760323860679,    4.0],
            [1.6926459360780468,  -2.507415021600687,    5.0],
            [0.3415803843118469,  -1.9467212190806362,   6.0],
            [11.356013549805343,   3.1415923080974144,   0.0],
            [7.047079899566103,    0.6418961727135171,   1.0],
            [5.910712528628478,    2.873060225203758,    2.0],
            [8.33420495108884,    -1.472951124716287,    3.0],
            [0.7264083085745505,   2.13102891886042,     4.0],
            [0.4417775258031982,  -1.9467205270647434,   6.0],
            [21.951094336278953,   0.04090385336433795,  8.0],
            [19.071665738665413,   2.0799488445885883,   9.0],
            [10.581104417375242,  -1.2845200758813653,  10.0],
            [5.7905267138911585,   3.0851406287193406,   0.0],
            [4.595411800104623,   -2.191767610675001,    1.0],
            [2.719214307894974,    1.5668505958710426,   2.0],
            [22.37640621770848,   -1.48444762630519,     3.0],
            [3.256678705921383,   -0.13262173979571057,  4.0],
            [4.366507993657452,    0.8549454393978959,   5.0],
            [2.2962035527388163,  -1.8238324286863934,   6.0],
            [4.3665198362577256,   0.8549473124843617,   7.0],
            [13.06774131804103,    3.0165343152969686,   8.0],
            [7.341106279149471,    1.298394952845383,    9.0],
            [0.619693590706742,    1.6578488456789093,  10.0],
            [1.1343357873098667,  -2.8941539138254875,   0.0],
            [0.8172462212783779,  -2.326943471308606,    1.0],
            [14.905870391417444,  -1.2416433043780466,   2.0],
            [7.749696666242588,    2.1346741436756713,   3.0],
            [6.214395633014001,    2.3321080990788405,   4.0],
            [0.8994799202869519,  -1.8238325655296974,   6.0],
            [7.083402874874581,    1.3177565259680222,   7.0],
            [3.594744147985383,    2.0870807697230855,   8.0],
            [3.96770399431819,     1.2320030028334046,   9.0],
            [0.1630225830045953,   1.5610711929567072,  10.0],
            [1.4985693098345587,  -2.5412888629714154,   0.0],
            [0.07095784828684656, -2.6535946325731503,   1.0],
            [15.434155813071149,  -1.4939201014489119,   2.0],
            [2.468990887269235,   -2.368268817156088,    3.0],
            [0.9783231329072508,   2.14660419123629,     4.0],
            [7.0834171625818385,   1.3177534405293487,   5.0],
            [4.603902313661107,    2.087077706227266,    8.0],
            [4.584987246125035,    1.408308499183715,    9.0],
            [2.582276374048861,    1.5610707896029616,  10.0],
            [0.682090472618723,    2.907932154360328,    1.0],
            [1.0147105163174965,   1.470258313203723,    2.0],
            [12.770223724122374,  -1.4018414263720673,   3.0],
            [13.963748675897035,  -1.5091405111272802,   4.0],
            [3.0112032240006017,  -1.4877388188869454,   5.0],
            [19.96583933768452,    1.5091782822795372,   6.0],
            [1.6204605757446968,   3.004412161398081,    7.0],
            [0.9746120252455769,   1.616114108105094,    8.0],
            [0.5876569212013255,  -1.4070187359556758,   9.0],
            [3.379776336401267,    1.2471614554650825,  10.0],
            [8.097494782649697,   -2.0004735429006084,   0.0],
            [2.618879239561724,   -1.8923588595084828,   1.0],
            [38.75734430996545,   -1.0828955829229863,   3.0],
            [21.599399230028617,   2.1599868358309484,   4.0],
            [48.334979875726376,  -0.12460086576383093,  5.0],
            [30.252203121005984,   2.687725061196421,    6.0],
            [40.43712344691874,   -0.8757709672759438,   7.0],
            [42.47671175844863,   -2.914699615424994,    8.0],
            [61.34112363144684,    0.8662667513316995,   9.0],
            [4.3316098591072505,   2.502823981361153,   10.0],
            [36.80863434724782,    2.909237401664744,    0.0],
            [0.7946178326162305,  -2.263359356054459,    1.0],
            [0.8803861169662044,  -1.660863738301054,    2.0],
            [57.43964019826877,   -1.0961267297497579,   3.0],
            [14.682442798950852,   2.30930318626547,     4.0],
            [15.623772086994899,  -3.090756987622661,    5.0],
            [18.37727896726938,    1.2407998383609913,   6.0],
            [27.949247786257317,  -1.9163572403388236,   7.0],
            [12.354629363232961,  -0.06890755675473406,  8.0],
            [43.072059956308394,  -1.9766025501308957,   9.0],
            [52.968488778738646,   2.9092377215258267,   0.0],
            [10.557054380101578,  -2.2633583413616107,   1.0],
            [10.601924970924184,   1.8456703623562205,   2.0],
            [19.88605409378788,   -0.9358680490474487,   3.0],
            [30.080392199825607,   0.24996847708387496,  4.0],
            [9.113865527396207,   -3.090756922741306,    5.0],
            [7.917833040902303,   -1.279869718066164,    6.0],
            [49.144768963202374,   1.9595080667868585,   7.0],
            [20.066906145525607,  -0.06890716311267381,  8.0],
            [67.0123013288774,     2.0337308391684896,   9.0],
            [66.78948915863202,   -2.995380132690458,   10.0],
            [51.68316323879203,   -0.09661604988082098,  0.0],
            [3.34449162727901,     3.072689247622834,    1.0],
            [17.9875046535609,     2.0216316259502616,   2.0],
            [1.7199835371402024,   2.1018063963403315,   3.0],
            [34.879685991455176,  -2.936730107059196,    4.0],
            [7.435431374493771,   -3.1092955578520662,   5.0],
            [38.409992841938,     -1.3402138437934359,   6.0],
            [5.016737490828967,    3.072684453360976,    7.0],
            [0.997642985934143,   -1.4745571439621017,   9.0],
            [26.045703508285918,   0.15745119342315458, 10.0],
            [33.59078176914744,    1.8320643880325111,   1.0],
            [17.02783631195487,    2.101806362019636,    3.0],
            [0.9957010815226107,  -3.1092955870482726,   5.0],
            [13.49540176355124,   -1.3402137438098733,   6.0],
            [47.40799497806193,   -1.474557158248994,    9.0],
            [5.62205965578578,     0.15745111593509503, 10.0],
            [9.960700058751561,    2.9284285951718543,   0.0],
            [2.3513546258109677,  -1.309528298133053,    1.0],
            [68.96273619830542,    2.10180639904948,     3.0],
            [6.691113447503941,   -3.1092949577216586,   5.0],
            [39.032585659210525,  -1.4745571446362231,   9.0],
            [28.67250267767209,    0.15745132598028583, 10.0],
            [1.561971379396357,   -1.3095282433932005,   1.0],
            [42.14389955405062,    2.101806511731109,    3.0],
            [17.205722089204976,  -3.109295165400832,    5.0],
            [12.32608172089994,   -1.4745560885278333,   9.0],
            [3.185838019237176,    0.1574520156116761,  10.0],
            [9.496787200814852,   -1.3095284438575725,   1.0],
            [25.286341306057817,   2.1018064968573253,   3.0],
            [15.135504889762645,  -1.3095282617515491,   1.0],
            [16.857567901156926,   2.1018070057409233,   3.0],
            [5.045169925315994,   -1.3095270908670373,   1.0],
        ]
    )
    # Simulate the system.
    simulation.simulate(cmds, pose)
    # Draw simulation results.
    file_name = None
    simulation.plot_single(file_name=file_name)
    simulation.plot_selected([0, 9], file_name=file_name)
    simulation.simplot_selected([0, 9], file_name=file_name)
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
        title=False,
        file_name=file_name,
    )
    """ anim = simulation.simanimation(
        vel=30,
        anim_length=1100,
        boundary=True,
        last_section=True,
        file_name=file_name,
    ) """
    plt.show()


########## test section ################################################
if __name__ == "__main__":
    test_simulation()
    plt.show()
    pass
