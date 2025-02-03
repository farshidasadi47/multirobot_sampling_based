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
        length: np.array,
        uncertainty_scaling=None,
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
        self.set_length(length, uncertainty_scaling)
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

    def set_length(self, length, uncertainty_scaling=None):
        msg = '"pivot_length" should be a 2D numpy array.'
        assert length.ndim == 2, msg
        msg = "uncertainty scaling matric should be 2D numpy array."
        if uncertainty_scaling is not None:
            assert uncertainty_scaling.ndim == 2, msg
        #
        self.n_robot, self.n_mode = length.shape
        # Construct leg (tumbling) matrix.
        self.length = length.astype(float)
        self.tumbling_length = length[0, 0]
        self.uncertainty_scaling = uncertainty_scaling
        if uncertainty_scaling is None:
            uncertainty_scaling = np.zeros((self.n_robot, 2 * self.n_mode))
        self.uncertainty_scaling = uncertainty_scaling
        # Construct displacement ratio matrix.
        self.beta = np.zeros_like(self.length)
        # beta := length_robot_j/length_robot_0
        for robot in range(self.n_robot):
            self.beta[robot] = self.length[robot] / self.length[0]
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

    def set_space(
        self,
        *,
        ubx=115,
        lbx=-115,
        uby=90,
        lby=-90,
        rcoil=90,
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
                [[-5, -30], [-5, -100], [5, -100], [5, -30]], dtype=float
            ),
            np.array([[-5, 100], [-5, 30], [5, 30], [5, 100]], dtype=float),
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
        length = np.array([[10, 9, 7], [10, 7, 5], [10, 5, 9]])
        uncertainty_scaling = 0.2 * np.ones((3, 2 * 3))
        return cls(length, uncertainty_scaling)

    @classmethod
    def robo4(cls):
        length = np.array(
            [
                [10, 9, 7, 5, 3],
                [10, 7, 5, 3, 9],
                [10, 5, 3, 9, 7],
                [10, 3, 9, 7, 5],
            ]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 4))
        return cls(length, uncertainty_scaling)

    @classmethod
    def robo5(cls):
        length = np.array(
            [
                [10, 10, 6, 8, 6],
                [10, 6, 8, 6, 10],
                [10, 8, 6, 10, 10],
                [10, 6, 10, 10, 6],
                [10, 10, 10, 6, 8],
            ]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 5))
        return cls(length, uncertainty_scaling)

    @classmethod
    def robo10(cls):
        length = np.array(
            [
                [2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0],
                [2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
                [1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
                [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0],
                [2.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0],
                [1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0],
                [1.0, 2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0],
                [2.0, 2.0, 2.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0],
            ]
        )
        uncertainty_scaling = np.zeros((10, 10 * 2))
        specs = cls(length, uncertainty_scaling)
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
        length = np.array(
            [[11.44, 8.46, 6.67], [11.44, 6.61, 4.75], [11.44, 4.79, 8.18]]
        )
        uncertainty_scaling = 0.2 * np.ones((3, 2 * 3))
        return cls(length, uncertainty_scaling)

    @classmethod
    def robo4p(cls):
        """length = np.array(
            [
                [12.55, 7.48, 4.91, 4.90, 4.93],
                [12.55, 4.95, 4.95, 4.91, 7.59],
                [12.55, 4.85, 4.87, 7.42, 4.89],
                [12.55, 4.86, 7.44, 4.82, 4.84],
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
        )"""
        length = np.array(
            [
                [12.82, 7.56, 4.96, 4.91, 4.94],
                [12.82, 4.89, 4.95, 4.94, 7.58],
                [12.82, 4.82, 4.90, 7.54, 4.90],
                [12.82, 4.91, 7.64, 4.93, 4.86],
            ]
        )
        uncertainty_scaling = np.array(
            [
                [1.78, 2.61, 1.07, 1.39, 1.02, 1.40, 1.16, 1.08],
                [5.34, 3.20, 3.98, 3.47, 3.93, 3.68, 4.17, 3.74],
                [4.45, 4.48, 4.24, 3.94, 3.84, 3.74, 2.53, 2.89],
                [4.78, 3.92, 4.85, 5.09, 2.69, 2.12, 4.40, 3.80],
                [4.11, 3.79, 3.67, 2.96, 4.71, 5.16, 5.07, 4.17],
            ]
        )
        # return cls(length, uncertainty_scaling)
        return cls(length, uncertainty_scaling)

    @classmethod
    def robo5p(cls):
        length = np.array(
            [
                [15.24, 4.80, 8.00, 8.00, 5.00],
                [15.24, 7.90, 8.00, 4.90, 5.00],
                [15.24, 7.90, 4.90, 4.90, 5.10],
                [15.24, 4.90, 4.90, 5.00, 8.00],
                [15.24, 4.80, 4.90, 8.10, 8.20],
            ]
        )
        uncertainty_scaling = 0.2 * np.ones((5, 2 * 5))
        return cls(length, uncertainty_scaling)

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
        else:
            text = f"Step {i:3d} of {i_tot:3d}, moving at mode {cmd_mode}"
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
            [26.6666666666666,     3.141592653589793,   0.0],
            [13.333333333333343,   3.141592653589793,   1.0],
            [5.404427998324123,    1.0046575307874137,  2.0],
            [2.799999852401505,   -3.1415926341503844,  3.0],
            [0.9590407454884609,   0.3119980080871358,  4.0],
            [0.9333337800461822,   3.141592557240446,   5.0],
            [0.5324607323239825,   0.22229486640971174, 6.0],
            [0.533333133736365,    3.141592597722726,   7.0],
            [28.509013683695272,  -1.4382151320368242,  8.0],
            [4.029323590737861,    2.6422450077441546,  9.0],
            [0.26666681130548237,  3.1415924401298714,  1.0],
            [23.466055362269966,   1.2127459811716794,  2.0],
            [5.209955327897452,    3.015326391708767,   3.0],
            [28.997552591849836,  -1.585683613362379,   4.0],
            [31.279867676108083,   1.506106425691048,   5.0],
            [0.5451098274750116,   2.902488658421449,   6.0],
            [3.6024799416830158,  -1.870736235345758,   7.0],
            [53.4680038555213,    -1.338545087083283,   8.0],
            [13.52745703997336,    2.719674645604221,   9.0],
            [16.49910346512794,   -1.3372535057734667,  0.0],
            [6.794673187441156,    3.1415923658953706,  1.0],
            [30.25248544454732,    1.4193368633566572,  2.0],
            [2.593539324210232,   -2.717353761169013,   3.0],
            [9.366320475458465,   -1.5862455611883748,  4.0],
            [0.2511429642176665,   2.249713722885584,   5.0],
            [1.4525261598669241,  -1.6123706722367142,  6.0],
            [11.787010612987633,  -1.4215000003356524,  7.0],
            [4.437605077076877,    2.4722535737408355,  8.0],
            [27.04357329862019,   -1.5176179751642782,  9.0],
            [1.7414689442577398,   2.3973928366242188,  0.0],
            [30.805060188698366,  -1.0401392421303532,  1.0],
            [19.213274173528962,   3.0179766114566253,  2.0],
            [4.099821873398244,    2.305540265701558,   3.0],
            [0.4294051210311145,  -3.006926960897695,   4.0],
            [2.7597720834704793,  -1.9970845642546051,  5.0],
            [1.8644252233881606,   3.051001756307467,   6.0],
            [5.008311617389648,   -1.3184325079300119,  7.0],
            [5.214692557583672,    2.0395175100230682,  8.0],
            [0.23046275671821712,  2.1299645532865843,  9.0],
            [0.7543222536733798,   1.739781111755997,   0.0],
            [0.958091456400976,    1.547297396157708,   1.0],
            [27.184994442434594,  -0.3437654913752979,  2.0],
            [50.448490562368214,   3.0477628715159355,  3.0],
            [23.412243652544788,   0.9814006470534385,  4.0],
            [9.898029049206531,    2.3987212561297993,  5.0],
            [1.8418246541236636,   1.2418685136664431,  6.0],
            [0.5104037443474335,   2.9730335056438717,  7.0],
            [0.8169684806348785,   2.0395174043315127,  8.0],
            [9.171322276629251,   -0.30747160893004144, 9.0],
            [0.9201993412314649,  -0.26094598610514785, 0.0],
            [17.81635946183762,    2.1687332470942993,  1.0],
            [2.4613275334397513,   2.557998938746704,   2.0],
            [13.871956544670503,   0.5384813739210409,  3.0],
            [1.4538917752934024,  -1.5518392074604257,  4.0],
            [11.125239289400659,  -3.052863088455054,   5.0],
            [5.76574475814133,    -0.8439614157350984,  6.0],
            [30.463698032265416,  -1.3205621658917295,  7.0],
            [44.066717884271156,  -2.5301283276373647,  8.0],
            [1.7465527773109062,   2.791236292550659,   9.0],
            [72.19946944566195,    1.2744655504011306,  0.0],
            [8.322832468247212,    2.651076292918022,   1.0],
            [11.345033372656184,  -2.7558779760181964,  2.0],
            [7.78115490196021,     1.1933864680287931,  3.0],
            [8.565045189375141,   -1.3342486265817028,  4.0],
            [4.181154794613225,   -2.7363675957225344,  5.0],
            [1.7860460239402771,   2.9179109896099167,  6.0],
            [34.53866974617062,    1.9862242931527005,  7.0],
            [49.1874529308013,     2.6701013870500163,  8.0],
            [0.28840667243443263,  2.2994545330324416,  9.0],
            [37.72224399698821,   -1.990198779741956,   0.0],
            [8.299665149110597,    2.145839251607911,   1.0],
            [41.14702670723085,   -2.027792267446272,   2.0],
            [7.029782080547203,   -0.5525771436600074,  3.0],
            [14.620214118362663,   2.542199630186999,   4.0],
            [19.829528499990136,  -1.3016537547596063,  5.0],
            [0.408203492517575,   -1.4226408351887945,  6.0],
            [10.102563475356463,   1.9862241514603705,  7.0],
            [82.39541202855295,    1.0472009694844335,  8.0],
            [0.5710450995364913,   2.299454593359094,   9.0],
            [0.2215433833306217,  -1.990198479372172,   0.0],
            [10.937817280968064,  -2.02779280121752,    2.0],
            [13.042313747719216,  -0.5525770716787334,  3.0],
            [1.767653403886987,   -1.4583720426299318,  4.0],
            [0.32515024722143354, -2.7626213804968054,  5.0],
            [14.60399720429952,    2.8918666706870186,  6.0],
            [9.32544589132831,     1.9862245300691073,  7.0],
            [4.336591713947365,    1.0472029228883515,  8.0],
            [0.8394361702309846,   2.299454346674016,   9.0],
            [21.932794647946263,  -1.9901980906700705,  0.0],
            [21.279564455888007,  -0.5525769684417986,  3.0],
            [36.973208540984956,   2.5059923052305035,  4.0],
            [6.304020332373134,   -1.2790939097421925,  5.0],
            [23.341921905701266,   2.2994541986781187,  9.0],
            [8.023298657071642,   -1.279093910878154,   5.0],
            [3.799844061706656,    2.2994538813452134,  9.0],
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
