# %%
########################################################################
# This files hold classes and functions that designs motion plan for
# multiple magnetic robot system.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import time
import re
from typing import TypedDict
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import root
import faiss

import triangle as tr
import cv2 as cv
import fcl

try:
    from multirobot_sampling_based import model
except ModuleNotFoundError:
    import model

round_down = model.round_down
wrap_2pi = model.wrap_2pi
define_colors = model.define_colors

np.set_printoptions(precision=2, suppress=True)

# plt.rcParams["figure.figsize"] = [7.2, 8.0]
plt.rcParams.update({"font.size": 11})
plt.rcParams["font.family"] = ["serif"]
# plt.rcParams["font.serif"] = "Times New Roman"
plt.rcParams["mathtext.fontset"] = "cm"
# plt.rcParams["text.usetex"] = False
plt.rcParams["hatch.linewidth"] = 0.5

# Create a logger
logger = logging.getLogger("MyLogger")
logger.setLevel(logging.DEBUG)
# Create a console (stream) handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
# Create a logging format
log_formatter = logging.Formatter("%(message)s")
console_handler.setFormatter(log_formatter)
# Add handlers to the logger
logger.addHandler(console_handler)


########## classes and functions #######################################
def find_rep_rows(data, tol):
    def within_tolerance(row, rows, tol):
        rows = np.atleast_2d(rows)
        if len(rows):
            drows = (row - rows).reshape(len(rows), -1, 2)
            return np.where(
                np.all(np.linalg.norm(drows, axis=2) < tol, axis=1)
            )[0]
        else:
            return []

    N = len(data)
    groups = []
    inds = np.arange(N)
    unused = np.ones(N, dtype=bool)

    for i in inds:
        if unused[i]:
            unused[i] = False
            group = [i]
            repeated_inds = within_tolerance(data[i], data[unused], tol)
            repeated_inds = inds[unused][repeated_inds].tolist()
            group.extend(repeated_inds)
            unused[repeated_inds] = False
            if len(group) > 1:
                groups.append(group)

    return groups


class Obstacles:
    """
    This class creates mesh representation of workspace and obstacles
    from given dimensions and external contour of obstacles.
    """

    def __init__(
        self, specs, obstacle_contours=[], with_boundary=False, with_coil=False
    ):
        self._specs = specs
        self._with_boundary = with_boundary
        self._with_coil = with_coil
        eps = 0.1  # Convex approximation tolerance.
        self._contours = self._approximate_with_convex(obstacle_contours, eps)

    def get_obstacle_mesh(self):
        return self._get_mesh_obstacles(self._contours)

    def get_cartesian_obstacle_contours(self):
        return self._contours

    def _approximate_with_convex(self, contours, eps=0.1):
        """
        Approximates a given contour with its convex hull if the
        convexity defects are not more than eps*100 percentage.
        """
        approx_contours = []
        for cnt in contours:
            cnt = np.float32(cnt)
            if not cv.isContourConvex(cnt):
                hull = cv.convexHull(cnt)
                area_cnt = cv.contourArea(cnt)
                area_hull = cv.contourArea(hull)
                if area_hull < (1 + eps) * area_cnt:
                    cnt = hull
                else:
                    cnt = cv.approxPolyDP(cnt, 1, True)
            approx_contours.append(cnt.reshape(-1, 2))
        return approx_contours

    def _get_mesh_obstacles(self, contours):
        """
        Triangulates obstacles and returns the mesh and mech_contour.
        """
        mesh = []
        mesh_contours = []
        for cnt in contours:
            segments = np.vstack(
                (range(len(cnt)), np.roll(range(len(cnt)), -1))
            ).T
            tris = tr.triangulate(
                {"vertices": cnt.squeeze(), "segments": segments}, "p"
            )
            verts = tris["vertices"].astype(cnt.dtype)
            tris = tris["triangles"]
            mesh_contours.extend(list(verts[tris]))
            mesh.append((verts, tris, cv.isContourConvex(cnt.astype(int))))
        # Get space mesh.
        if self._with_boundary:
            mesh_space, mesh_contours_space = self._get_space_mesh()
            mesh += mesh_space
            mesh_contours += mesh_contours_space
        # Coil mesh.
        if self._with_coil:
            mesh_coil, mesh_contours_coil = self._get_coil_mesh()
            mesh += mesh_coil
            mesh_contours += mesh_contours_coil
        return mesh, mesh_contours

    def _get_space_mesh(self):
        (lbx, ubx), (lby, uby) = self._specs.bounds
        verts_lbx = np.array(
            [[1.5 * lbx, 0], [lbx, 1.5 * lby], [lbx, 1.5 * uby]]
        )
        verts_ubx = np.array(
            [[1.5 * ubx, 0], [ubx, 1.5 * uby], [ubx, 1.5 * lby]]
        )
        verts_lby = np.array(
            [[0, 1.5 * lby], [1.5 * ubx, lby], [1.5 * lbx, lby]]
        )
        verts_uby = np.array(
            [[0, 1.5 * uby], [1.5 * lbx, uby], [1.5 * ubx, uby]]
        )
        tris = np.array([[0, 1, 2]])
        mesh_contours = [verts_lbx, verts_ubx, verts_lby, verts_uby]
        #
        mesh = [(vert, tris, True) for vert in mesh_contours]
        return mesh, mesh_contours

    def _get_coil_mesh(self):
        """
        This approximates a circular coil with a polygon circumscribed
        around the circle and returns the mech and mesh contours.
        """
        N = 8  # Polygon vertices.
        r = self._specs.rcoil
        phi = np.pi / N
        # Inner polygon vertices, that circumscribes the coil circle.
        inner_vertices = [
            (r * np.cos(2 * phi * n + phi), r * np.sin(2 * phi * n + phi))
            for n in range(N)
        ]
        # Outer polygon vertices.
        r = r / np.cos(phi) * 1.2
        outer_vertices = [
            (r * np.cos(2 * phi * n), r * np.sin(2 * phi * n))
            for n in range(N)
        ]
        # Final mesh.
        verts = np.vstack((inner_vertices, outer_vertices))
        ind_inner = np.arange(N)
        ind_outer = np.arange(N) + N
        ind0 = np.tile(ind_inner, (2, 1)).T.flatten()
        ind1 = np.roll(np.vstack((ind_inner, ind_outer)).T.flatten(), -1)
        ind2 = np.roll(np.vstack((ind_outer, ind_outer)).T.flatten(), -2)
        tris = [tri for tri in np.vstack((ind0, ind1, ind2)).T]
        mesh_contours = list(verts[tris])
        mesh = [(verts, tris, False)]
        # Join to pairs of adjacent trianles.
        tris = [np.vstack((a, b)) for a, b in zip(tris[::2], tris[1::2])]
        idx_convex = []
        for tri in tris:
            _, idx = np.unique(tri, return_index=True)
            idx_convex.append(tri.flatten()[np.sort(idx)])
        tri_com = np.array([[0, 1, 2], [0, 3, 2]])
        mesh = [(verts[idx], tri_com, True) for idx in idx_convex]
        return mesh, mesh_contours


class Collision:
    def __init__(self, mesh, specs, with_coil=True):
        self.T0 = fcl.Transform()
        self.mesh = mesh
        self.rrob = specs.clearance
        self.rmin = specs.dmin / 2 * 0.9995  # To allow robots touching.
        self.dmin = self.rmin * 2
        self.dmin2 = self.dmin**2
        self.robot_pairs = specs.robot_pairs
        self._specs = specs
        self.ball0 = fcl.Sphere(self.rrob)
        self.obstacles = self._build_obstacles(mesh)
        self.cmanager, self.col_req, self.dis_req = self._build_cmanager()
        self._with_coil = with_coil

    def update_obstacles(self, mesh):
        self.mesh = mesh
        self.obstacles = self._build_obstacles(mesh)

    def _build_obstacles(self, mesh):
        obstacles = []
        for verts, tris, is_convex in mesh:
            if is_convex:
                faces = np.concatenate(
                    (3 * np.ones((len(tris), 1), dtype=np.int64), tris), axis=1
                ).ravel()
                obs = fcl.Convex(verts, len(tris), faces)
                obstacles.append(fcl.CollisionObject(obs))
            else:
                for tri in tris:
                    vert = verts[tri]
                    tri = np.array([[0, 1, 2]], dtype=int)
                    faces = np.concatenate(
                        (3 * np.ones((len(tri), 1), dtype=np.int64), tri),
                        axis=1,
                    ).ravel()
                    obs = fcl.Convex(vert, len(tri), faces)
                    obstacles.append(fcl.CollisionObject(obs))
        return obstacles

    def _build_cmanager(self):
        cmanager = fcl.DynamicAABBTreeCollisionManager()
        cmanager.registerObjects(self.obstacles)
        cmanager.setup()
        col_req = fcl.CollisionRequest(
            num_max_contacts=1000,
            enable_contact=True,
        )
        dis_req = fcl.DistanceRequest()
        return cmanager, col_req, dis_req

    def is_collision(self, pose):
        return (
            self._is_collision_boundary(pose)
            or self._is_collision_intrarobot(pose)
            or self._is_collision(pose.reshape(-1, 2))[0]
        )

    def is_collision_line(self, pose_i, pose_f):
        # Collision with obstacles.
        is_collision_a, time_collision_a = self._is_collision_line(
            pose_i, pose_f
        )
        # Collision between robots.
        is_collision_b, time_collision_b = self._is_collision_line_intrarobot(
            pose_i, pose_f
        )
        # Collision between convex boundary.
        is_collision_c, time_collision_c = self._is_collision_line_boundary(
            pose_i, pose_f
        )
        is_collision = is_collision_a or is_collision_b or is_collision_c
        time_collision = min(
            time_collision_a, time_collision_b, time_collision_c
        )
        return is_collision, time_collision

    def is_collision_lines(self, poses_i, poses_f):
        poses_i = np.atleast_2d(poses_i)
        poses_f = np.atleast_2d(poses_f)
        is_collision = False
        i_collision = None
        time_collision = 1.0
        #
        for i_collision, (pose_i, pose_f) in enumerate(zip(poses_i, poses_f)):
            is_collision, time_collision = self.is_collision_line(
                pose_i, pose_f
            )
            if time_collision < 1:
                break
        return is_collision, i_collision, time_collision

    def is_collision_path(self, poses):
        # First check vertices of the path, ignore start and end.
        if len(poses) > 2:
            is_collision, _ = self._is_collision_inc(poses[1:-1])
            if is_collision:
                return True
        # Now check path between points.
        is_collision, i_collision, time_collision = self.is_collision_lines(
            poses[:-1], poses[1:]
        )
        if is_collision and i_collision < len(poses) and time_collision < 1.0:
            return True
        else:
            return False

    def _is_collision_intrarobot(self, pose):
        """
        Checks if robots are coliding in current position.
        """
        is_collision = False
        pose = np.reshape(pose, (-1, 2))
        dpose = np.diff(pose[self._specs.robot_pairs], axis=1).squeeze()
        D2 = np.einsum("ij,ij->i", dpose, dpose)
        is_collision = (D2 < self.dmin2).any()
        return is_collision

    def _is_collision_boundary(self, pose):
        (lbx, ubx), (lby, uby) = self._specs.obounds
        rcoil2 = self._specs.rcoil**2
        pose = pose.reshape(-1, 2)
        #
        is_collision = np.logical_or(pose[:, 0] < lbx, pose[:, 0] > ubx).any()
        is_collision |= np.logical_or(pose[:, 1] < lby, pose[:, 1] > uby).any()
        #
        if self._with_coil:
            D2 = np.einsum("ij,ij->i", pose, pose)
            is_collision |= (D2 > rcoil2).any()
        return is_collision

    def _is_collision(self, poses):
        poses = np.atleast_2d(poses)
        n_poses = len(poses)
        # Pad last axis with zero.
        poses = np.concatenate((poses, np.zeros((n_poses, 1))), axis=1)
        # Setup robots collision object and manage.
        balls = [fcl.Sphere(self.rrob) for _ in range(n_poses)]
        robots = [
            fcl.CollisionObject(ball, fcl.Transform(pos))
            for ball, pos in zip(balls, poses)
        ]
        manager = fcl.DynamicAABBTreeCollisionManager()
        manager.registerObjects(robots)
        manager.setup()
        # Detect collision.
        cdata = fcl.CollisionData(request=self.col_req)
        self.cmanager.collide(manager, cdata, fcl.defaultCollisionCallback)
        # Find index of first collision.
        ind_last_collision_free = n_poses - 1
        if cdata.result.is_collision:
            ind_last_collision_free = (
                min(
                    [
                        balls.index(contact.o2)
                        for contact in cdata.result.contacts
                    ]
                )
                - 1
            )
        return cdata.result.is_collision, ind_last_collision_free

    def _is_collision_inc(self, poses):
        poses = np.atleast_2d(poses)
        n_poses = len(poses)
        # Reshape to (n_poses, n_robot, 2).
        poses = np.reshape(poses, (n_poses, -1, 2))
        ind_last_collision_free = n_poses - 1
        for ind, poss in enumerate(poses):
            is_collision, _ = self._is_collision(poss)
            if is_collision:
                ind_last_collision_free = ind - 1
                break
        return is_collision, ind_last_collision_free

    def _is_collision_line(self, pose_i, pose_f):
        # Reshape to (n_robot, 2).
        pose_i = np.reshape(np.atleast_1d(pose_i), (-1, 2))
        pose_f = np.reshape(np.atleast_1d(pose_f), (-1, 2))
        # Pad last axis with zero.
        pose_i = np.concatenate((pose_i, np.zeros((len(pose_i), 1))), axis=1)
        pose_f = np.concatenate((pose_f, np.zeros((len(pose_f), 1))), axis=1)
        time_collisions = [np.inf]
        is_collisions = [False]
        for pos_i, pos_f in zip(pose_i, pose_f):
            robot = fcl.CollisionObject(self.ball0, fcl.Transform(pos_i))
            for obstacle in self.obstacles:
                request = fcl.ContinuousCollisionRequest(gjk_solver_type=1)
                result = fcl.ContinuousCollisionResult()
                time_collisions.append(
                    fcl.continuousCollide(
                        robot,
                        fcl.Transform(pos_f),
                        obstacle,
                        self.T0,
                        request,
                        result,
                    )
                )
                is_collisions.append(result.is_collide)
        return max(is_collisions), min(round_down(time_collisions))

    def _is_collision_line_intrarobot(self, pose_i, pose_f):
        """
        This function checks collision between line segment and a circle
        at origin.
        """
        pose_i = np.reshape(pose_i, (-1, 2))
        pose_f = np.reshape(pose_f, (-1, 2))
        dpose_i = np.diff(pose_i[self.robot_pairs], axis=1).squeeze()
        dpose_f = np.diff(pose_f[self.robot_pairs], axis=1).squeeze()
        D = dpose_f - dpose_i
        #
        a = np.einsum("ij,ij->i", D, D)  # Row wise dot product.
        b = np.einsum("ij,ij->i", D, dpose_i)
        c = np.einsum("ij,ij->i", dpose_i, dpose_i) - self.dmin2
        delta = b**2 - a * c
        #
        time_collision_free = np.ones_like(a, dtype=float)
        t1 = np.ones_like(a, dtype=float)
        t2 = np.ones_like(a, dtype=float)
        #
        ind = delta > 0  # There is no collision for delta <=0.
        t1[ind] = (-b[ind] - delta[ind] ** 0.5) / a[ind]
        t2[ind] = (-b[ind] + delta[ind] ** 0.5) / a[ind]
        # If t1 > 1 or t2 <= 0, there is no collision.
        ind[np.logical_or(t1 > 1, t2 <= 0)] = False
        # Now the motion is collision free up to t1 if t1 >=0.
        time_collision_free[ind] = np.where(t1[ind] >= 0, t1[ind], -1)
        is_collision = max(ind)
        time_collision_free = min(round_down(time_collision_free))
        return is_collision, time_collision_free

    def _is_collision_line_boundary(self, pose_i, pose_f):
        bx, by = self._specs.obounds
        rcoil2 = self._specs.rcoil**2
        # Test if initial position is inside boundary.
        is_collision = self._is_collision_boundary(pose_i)
        if is_collision:
            # Initial position is outside boundary.
            time_collision_free = -1
        else:
            # Initial position is inside boundary.
            pose_i = np.reshape(pose_i, (-1, 2))
            pose_f = np.reshape(pose_f, (-1, 2))
            dpose = pose_f - pose_i
            # Checking crossing x boundary.
            mask = np.logical_not(np.isclose(dpose[:, 0], 0))
            t = np.ones_like(dpose, dtype=float)
            t[mask] = (bx - pose_i[mask, 0, None]) / dpose[mask, 0, None]
            ind = np.logical_and(t > 0, t < 1)
            t[~ind] = 1
            is_collision = ind.any()
            time_collision_free = round_down(t).min()
            # Checking crossing y boundary.
            mask = np.logical_not(np.isclose(dpose[:, 1], 0))
            t = np.ones_like(dpose, dtype=float)
            t[mask] = (by - pose_i[mask, 1, None]) / dpose[mask, 1, None]
            ind = np.logical_and(t > 0, t < 1)
            t[~ind] = 1
            is_collision |= ind.any()
            time_collision_free = min(time_collision_free, round_down(t).min())
            # Check crossing coil boundary if requested.
            if self._with_coil:
                a = np.einsum("ij,ij->i", dpose, dpose)
                b = np.einsum("ij,ij->i", dpose, pose_i)
                c = np.einsum("ij,ij->i", pose_i, pose_i) - rcoil2
                delta = b**2 - a * c
                #
                if (delta < 0).any():
                    is_collision = True
                    time_collision_free = -1
                else:
                    t1 = -np.ones_like(a, dtype=float)
                    t2 = np.ones_like(a, dtype=float)
                    #
                    ind = a != 0
                    t1[ind] = (-b[ind] - delta[ind] ** 0.5) / a[ind]
                    t2[ind] = (-b[ind] + delta[ind] ** 0.5) / a[ind]
                    # There is no collision at all if t1<=0 and t2>=1.
                    ind[np.logical_and(t1 <= 0, t2 >= 1)] = False
                    # t1 > 0 or t2 < 0 is initial collision.
                    t2[ind] = np.where(
                        np.logical_or(t1[ind] > 0, t2[ind] < 0), -1, t2[ind]
                    )
                    time_collision_free = min(
                        time_collision_free, round_down(t2).min()
                    )
                    is_collision |= ind.any()
        return is_collision, time_collision_free


class RRT:
    class Path(TypedDict):
        inds: np.ndarray  # Array of indexes of nodes in the path.
        poses: np.ndarray  # Array of positions in the path.
        cmds: np.ndarray  # Array of displacement inputs in the path.
        cmd_modes: np.ndarray  # Array of input modes in the path.
        value: float  # Value of the path..
        i: int  # iteration the path was found.

    def __init__(
        self,
        specs,
        collision,
        obstacle_contours,
        max_size=1000,
        tol_cmd=1e-2,
        goal_bias=0.07,
        tol_goal=1e-0,
    ) -> None:
        self._start = None
        self._goal = None
        self._mode_sequence = None
        #
        self._specs = specs
        self._collision = collision
        self._obstacle_contours = obstacle_contours
        self._WI = np.linalg.pinv(specs.W)
        self._WITZ = np.zeros(
            (specs.n_mode, specs.n_robot * 2, specs.n_robot * 2), dtype=float
        )
        for m in range(specs.n_mode):
            self._WITZ[m] = (
                self._WI[2 * m : 2 * m + 2, :].T
                @ self._WI[2 * m : 2 * m + 2, :]
            )
        self._WIT = self._WI.T @ self._WI
        self._T = np.zeros_like(self._WITZ)
        self._T[0] = np.eye(specs.n_robot * 2) + specs.B[0] @ specs.K[0]
        for m in range(specs.n_mode - 1):
            self._T[m + 1] = self._T[m] + specs.B[m + 1] @ specs.K[m + 1]
        # Tree properties.
        self._max_size = max_size
        lb, ub = map(np.array, list(zip(*specs.bounds)))
        self._lb = np.tile(lb, specs.n_robot)
        self._ub = np.tile(ub, specs.n_robot)
        # Resolutions and tolerances.
        self._tol_cmd = tol_cmd  # Tolerance for checking zero inputs.
        self._tol_pose = 1e-6  # Tolerance for checking repetition.
        self._goal_bias = goal_bias
        self._tol_goal = tol_goal
        #
        self._log = True
        self._reset_tree()
        #
        define_colors(self)
        self._colors = list(self._colors.keys())

    def _reset_tree(self):
        self._n_state = self._specs.n_robot * 2
        max_size = self._max_size
        self._N = 0  # Current number of nodes.
        # Position of a node at corresponding index.
        self._poses = np.zeros((max_size, self._n_state), dtype=np.float32)
        # Depth of a node at corresponding index.
        self._depths = np.zeros(max_size, dtype=int)
        # Index of parent node for a node at corresponding index.
        self._parents = np.ones(max_size, dtype=int) * -1
        # Displacement input and its mode at corresponding index.
        self._commands = []  # np.zeros((max_size * mult, 2), dtype=float)
        self._command_modes = []  # np.zeros(max_size * mult, dtype=int)
        # Path starting from parent and ending in current pose.
        self._tracks = []
        # Cost from parent node to node at corresponding index.
        self._costs = np.zeros(max_size, dtype=float)
        # Cost from start to node at correspondinf index.
        self._values = np.zeros(max_size, dtype=float)
        # List of childs of a node at corresponding index.
        self._childs = tuple([] for _ in range(max_size))
        # ltype of the node, 0: normal, 1: suboptimal path, 2: optimal.
        self._ltypes = np.zeros(max_size, dtype=int)
        #
        self._faiss = faiss.IndexFlatL2(self._n_state)
        self._not_goal_mask = np.ones(max_size, dtype=bool)
        self._not_expanded_to_goal_mask = np.ones(max_size, dtype=bool)
        #
        self._goal_inds = []
        self._goal_ind_values = {}
        self.best_value = float("inf")
        self._best_ind = None
        self._paths = []
        self._best_path_ind = None
        self.cmds = np.zeros((1, 3))
        # List of node arts.
        self._arts_info = [None]
        self._arts = {}
        # Define the mapping of ltype to (color, zorder)
        self._line_style_map = {
            0: ("deepskyblue", 4),  # Normal.
            1: ("violet", 6),  # Suboptimal.
            2: ("red", 10),  # Optimal.
        }

    def _set_mode_sequence(self, basic_mode_sequence=None):
        """
        Checks mode_sequence, if available.
        Otherwise returns the default sequence.
        """
        # Modify tumble_index.
        n_mode = self._specs.n_mode
        if basic_mode_sequence is None:
            basic_mode_sequence = [0] + list(range(1, n_mode))
        self._basic_mode_sequence = np.array(basic_mode_sequence)
        self._n_basic_mode_sequence = len(self._basic_mode_sequence)

    @staticmethod
    def _get_next_mode_sequence(modes):
        next_modes = np.roll([mode for mode in modes if mode], -1)
        next_mode_sequence = []
        cnt = 0
        for mode in modes:
            if mode:
                next_mode_sequence.append(next_modes[cnt])
                cnt += 1
            else:
                next_mode_sequence.append(0)
        return np.array(next_mode_sequence, dtype=int)

    def print_node(self, ind):
        ind = np.clip(ind, 0, self._N - 1).item(0)
        print(f"pose:{self._poses[ind]}")
        print(f"depth:{self._depths[ind]}")
        print(f"parent:{self._parents[ind]}")
        print(f"commands:{self._commands[ind]}")
        print(f"command_modes:{self._command_modes[ind]}")
        print(f"track:{self._tracks[ind]}")
        print(f"cost:{self._costs[ind]}")
        print(f"value:{self._values[ind]}")
        print(f"childs:{self._childs[ind]}")

    def _add_node(
        self,
        *,
        pose,
        depth=0,
        parent=-1,
        cmds=np.zeros((1, 2)),
        cmd_modes=np.zeros(1, dtype=int),
        track=None,
        cost=0,
        goal_reached=False,
    ):
        new_ind = self._N
        self._poses[new_ind] = pose
        self._depths[new_ind] = depth
        self._parents[new_ind] = parent
        self._commands.append(cmds)
        self._command_modes.append(cmd_modes)
        self._tracks.append(track)
        self._costs[new_ind] = cost
        self._values[new_ind] = cost + self._values[parent]
        if parent > -1:
            self._childs[parent].append(new_ind)
        if goal_reached:
            self._goal_inds.append(new_ind)
            self._goal_ind_values[new_ind] = self._values[new_ind]
            self._not_goal_mask[new_ind] = False
            self._not_expanded_to_goal_mask[new_ind] = False
        self._N += 1
        # Add to faiss index.
        self._faiss.add(np.atleast_2d(pose.astype(np.float32)))
        return new_ind

    def _sample(self):
        prob = np.random.random()
        if prob > self._goal_bias:
            # If random point is selected.
            return False, np.random.uniform(self._lb, self._ub).astype(
                np.float32
            )
        else:
            # If goal point is selected.
            return True, self._goal

    def _sample_collision_free(self):
        is_collision = True
        while is_collision:
            is_goal_selected, rnd = self._sample()
            is_collision = self._collision.is_collision(rnd)
        return is_goal_selected, rnd

    def _distance(self, arr):
        arr = np.atleast_2d(arr)
        return np.linalg.norm(arr, axis=1)

    def _cost(self, arr):
        arr = np.atleast_2d(arr)
        u = np.einsum("ij, hj->hi", self._WI, arr).reshape(
            -1, self._specs.n_mode, 2
        )
        return np.sum(
            np.linalg.norm(u, axis=-1) > self._tol_cmd, axis=1, dtype=float
        )

    def _get_mode(self, depth):
        return self._basic_mode_sequence[depth % self._n_basic_mode_sequence]

    def _get_depth_mode_sequence(self, depth_i, depth_f=None):
        depths = np.arange(depth_i, depth_i + self._n_basic_mode_sequence)
        mode_sequence = self._get_mode(depths)
        depths = depths + 1
        return depths, mode_sequence

    def _nearest_node(self, pose, goal_selected):
        if goal_selected:
            inds_selector = np.nonzero(
                self._not_expanded_to_goal_mask[: self._N]
            )[0]
        else:
            inds_selector = np.nonzero(self._not_goal_mask[: self._N])[0]
        inds_selector = faiss.IDSelectorArray(inds_selector)
        #
        distances, inds = self._faiss.search(
            pose.reshape(1, -1),
            1,
            params=faiss.SearchParametersIVF(sel=inds_selector),
        )
        nearest_ind = inds[0][0]
        # Check if the nearest_ind is previously expanded toward goal.
        if goal_selected and nearest_ind < 0:
            return None, None, None, None
        nearest_pose = self._poses[nearest_ind]
        if goal_selected:
            self._not_expanded_to_goal_mask[nearest_ind] = False
        depth_i = self._depths[nearest_ind]
        depths, mode_sequence = self._get_depth_mode_sequence(depth_i)
        return nearest_ind, nearest_pose, depths, mode_sequence

    def _dynamics(self, pose_i, cmds, mode_sequence):
        poses = [pose_i]
        for mode, cmd in zip(mode_sequence, cmds):
            poses.append(poses[-1] + self._specs.B[mode] @ cmd)
        poses = np.array(poses)
        return poses

    def _remove_zero_cmds(self, cmds, depths, mode_sequence):
        mask = np.linalg.norm(cmds, axis=1) > self._tol_cmd
        cmds = cmds[mask]
        depths = depths[mask]
        mode_sequence = mode_sequence[mask]
        return cmds, depths, mode_sequence

    def _steer(self, pose_i, pose_f, depths, modes):
        n_robot = self._specs.n_robot
        beta = self._specs.beta[:, modes]
        while True:
            WI = np.kron(np.linalg.pinv(beta), np.eye(2))
            cmds = (WI @ (pose_f - pose_i)).reshape(-1, 2)
            if len(cmds) > n_robot:
                inds_to_keep = np.argsort(np.linalg.norm(cmds, axis=1))[
                    -n_robot:
                ]
            else:
                inds_to_keep = np.where(
                    np.linalg.norm(cmds, axis=1) > self._tol_cmd
                )[0]
            if len(inds_to_keep) == len(cmds):
                break
            mask = np.zeros_like(modes, dtype=bool)
            mask[inds_to_keep] = True
            beta = beta[:, mask]
            modes = modes[mask]
            depths = depths[mask]
        #
        poses = self._dynamics(pose_i, cmds, modes)
        return cmds, poses, depths, modes

    def _within_pose_tol(self, dpose):
        return np.all(np.hypot(dpose[::2], dpose[1::2]) <= self._tol_pose)

    def _within_goal(self, pose):
        dpose = (self._goal - pose).reshape(-1, 2)
        distances = np.linalg.norm(dpose, axis=1)
        return (distances < self._tol_goal).all()

    def _remove_colliding(
        self,
        cmds,
        poses,
        depths,
        mode_sequence,
        is_collision,
        i_collision,
        time_collision,
    ):
        if is_collision:
            cmds[i_collision] = cmds[i_collision] * time_collision
            poses[i_collision + 1] = (
                time_collision * poses[i_collision + 1]
                + (1 - time_collision) * poses[i_collision]
            )
            if (time_collision <= 0) or (
                np.linalg.norm(cmds[i_collision]) <= self._tol_cmd
            ):
                i_collision -= 1
            cmds = cmds[: i_collision + 1]
            poses = poses[: i_collision + 2]
            depths = depths[: i_collision + 1]
            mode_sequence = mode_sequence[: i_collision + 1]
        return cmds, poses, depths, mode_sequence

    def _extend(
        self,
        ind_i,
        cmds,
        poses,
        depths,
        mode_sequence,
        goal_selected,
    ):
        new_ind = None
        within_goal = False
        #
        cost = len(cmds)
        if cost:
            dpose = poses[-1] - poses[0]
            if not self._within_pose_tol(dpose):
                # If end is not repeating start.
                within_goal = self._within_goal(poses[-1])
                # Add new node.
                new_ind = self._add_node(
                    pose=poses[-1],
                    depth=depths[-1].item(),
                    parent=ind_i,
                    cmds=cmds,
                    cmd_modes=mode_sequence,
                    track=poses,
                    cost=cost,
                    goal_reached=within_goal,
                )
        return new_ind, within_goal

    def _generate_path(self, ind):
        if self._log:
            msg = (
                f"Number of tree nodes {self._N}, "
                + f"Path value: {self._values[ind]: 10.2f}"
            )
            logger.debug(msg)
        inds = [ind]
        parent = self._parents[ind]
        while parent > 0:
            ind = parent
            inds.insert(0, ind)
            parent = self._parents[ind]
        inds.insert(0, parent)
        #
        poses = np.concatenate(
            [self._tracks[ind][:-1] for ind in inds[1:]]
            + [self._tracks[inds[-1]][[-1]]],
            axis=0,
        )
        cmds = np.concatenate(
            [self._commands[ind] for ind in inds[1:]], axis=0
        )
        cmd_modes = np.concatenate(
            [self._command_modes[ind] for ind in inds[1:]]
        )
        value = self._values[inds[-1]]
        #
        path = self.Path(
            inds=inds,
            poses=poses,
            cmds=cmds,
            cmd_modes=cmd_modes,
            value=value,
            i=self._N,
        )
        self._paths.append(path)

    def _set_cmds(self):
        path = self._paths[self._best_path_ind]
        cmds = path["cmds"]
        modes = path["cmd_modes"]
        cmds, _, modes = self._remove_zero_cmds(cmds, modes, modes)
        self.cmds = np.hstack((cmds, modes[:, None]))

    def _set_ltype(self, inds, ltype=0):
        self._ltypes[inds] = ltype

    def _add_new_path(self, ind):
        ltype = 1
        inds_to_draw = []
        if self._values[ind] < self.best_value * 0.99:
            # New best path is found.
            if self._best_path_ind is not None:
                # Set ltype of previous best path to suboptimal.
                self._set_ltype(self._paths[self._best_path_ind]["inds"], 1)
                inds_to_draw += self._paths[self._best_path_ind]["inds"][1:]
            # Set ltype of the best path to optimal and update info.
            ltype = 2
            self._best_ind = ind
            self._best_path_ind = len(self._paths)
            self.best_value = self._values[ind]
        # Generate path and set suboptimal path ltype.
        self._generate_path(ind)
        self._set_ltype(self._paths[-1]["inds"], ltype)
        inds_to_draw += self._paths[-1]["inds"][1:]
        if ltype == 2:
            self._set_cmds()
        return inds_to_draw

    def _update_paths_all(self, new_ind, goal_reached):
        # Check for updated paths.
        past_values = np.array(
            [self._goal_ind_values.get(ind) for ind in self._goal_inds]
        )
        inds_to_update = np.where(
            self._values[self._goal_inds] < past_values * 0.99
        )[0]
        # Add new_ind to update list if necessary.
        if goal_reached:
            inds_to_update = inds_to_update.tolist()
        # Update or add paths.
        path_added = False
        inds_to_draw = []
        for ind in inds_to_update:
            ind = self._goal_inds[ind]
            inds_to_draw += self._add_new_path(ind)
            path_added = True
        if goal_reached:
            inds_to_draw += self._add_new_path(new_ind)
            path_added = True
        # Reset best path ltype if needed.
        if path_added:
            self._set_ltype(self._paths[self._best_path_ind]["inds"], 2)
            inds_to_draw += self._paths[self._best_path_ind]["inds"][1:]
        return inds_to_draw

    def _update_paths_best(self, new_ind, goal_reached):
        inds_to_draw = []
        if len(self._goal_inds):
            ind_best = self._values[self._goal_inds].argmin()
            ind_best = self._goal_inds[ind_best]
            if self._values[ind_best] < self.best_value * 0.99:
                # A better path has been discovered.
                inds_to_draw += self._add_new_path(ind_best)
        return inds_to_draw

    def _update_paths(self, new_ind, goal_reached):
        return self._update_paths_best(new_ind, goal_reached)

    def _set_arts_info(self, inds):
        inds = np.unique(inds).tolist()
        tracks = [self._tracks[ind] for ind in inds]
        ltypes = self._ltypes[inds]
        self._arts_info.append(list(zip(inds, tracks, ltypes)))

    def _accurate_tumbling(self, cmd):
        """This function gets a desired rotation and returns a sequence
        of two pure steps of rotations that produce the desired movement
        in rotation mode."""

        def F(phi, r1, r2, cmd):
            f1 = r1 * np.cos(phi[0]) + r2 * np.cos(phi[1]) - cmd[0]
            f2 = r1 * np.sin(phi[0]) + r2 * np.sin(phi[1]) - cmd[1]
            return np.array([f1, f2])

        #
        tumbling_length = self._specs.tumbling_length
        r = np.linalg.norm(cmd)
        if r > 1.0:
            # Practially we cannot execute such small displacements.
            # Divide r into even number of tumblings.
            r1 = np.ceil(0.5 * r / tumbling_length) * (2 * tumbling_length)
            r1 -= tumbling_length
            r2 = tumbling_length
            # Calculate angles so the two complete tumbling motions
            # be equivalent of original tumbling requested.
            sol = root(F, (0.0, 0.0), args=(r1, r2, cmd))
            phi_val = sol.x
            u_possible = np.zeros((2, 3))
            u_possible[0, 0] = r1 * np.cos(phi_val[0])
            u_possible[0, 1] = r1 * np.sin(phi_val[0])
            u_possible[1, 0] = r2 * np.cos(phi_val[1])
            u_possible[1, 1] = r2 * np.sin(phi_val[1])
        else:
            u_possible = np.zeros((2, 3))
        return u_possible

    def post_process(self, cmds, ang=0):
        """Post processes the command and adds intermediate steps."""
        ang = np.deg2rad(ang)
        # Get next mode sequence.
        next_modes = self._get_next_mode_sequence(cmds[:, -1])
        # Adjust mode_change parameters.
        mode_change = np.array([np.cos(ang), np.sin(ang), 0.0])
        mode_change_remainder = np.zeros(3)
        mcmds = []
        # Do mode change if first nonzero mode of cmds and mode sequence
        # does not match.
        mode = np.nonzero(self._basic_mode_sequence)[0][0]
        mode = self._basic_mode_sequence[mode]
        next_mode = np.nonzero(cmds[:, -1])[0]
        next_mode = int(cmds[next_mode[0], -1]) if len(next_mode) else 0
        if next_mode and next_mode != mode:
            mode_rel_length = self._specs.mode_rel_length[mode, next_mode]
            cmd_mode_change = mode_rel_length * mode_change
            mode_change_remainder += cmd_mode_change
            mode_change *= -1
            cmd_mode_change[-1] = -next_mode
            mcmds.append(cmd_mode_change)
        #
        for cmd, next_mode in zip(cmds, next_modes):
            mode = int(cmd[-1])
            if mode == 0:
                # Tumbling, must be divided to integer number of tumbles.
                cmd_tumbling = self._accurate_tumbling(
                    cmd - mode_change_remainder
                )
                mcmds.extend(cmd_tumbling)
                mode_change_remainder = np.zeros(3)
            else:
                # Pivot-walking, needs mode change.
                mode_rel_length = self._specs.mode_rel_length[mode, next_mode]
                if (
                    np.linalg.norm(
                        mode_rel_length * mode_change + mode_change_remainder
                    )
                    > self._specs.tumbling_length
                ):
                    # If next mode_change_remainder gets large reverse.
                    mode_change *= -1
                cmd_mode_change = mode_rel_length * mode_change
                mode_change_remainder += cmd_mode_change
                mode_change *= -1
                cmd_mode_change[-1] = -next_mode
                mcmds.append(cmd)
                mcmds.append(cmd_mode_change)
        # Compensate the remainder if necessary.
        if np.linalg.norm(mode_change_remainder) > 1.0:
            cmd_tumbling = self._accurate_tumbling(-mode_change_remainder)
            mcmds.extend(cmd_tumbling)
        return np.vstack(mcmds)

    def plan(
        self,
        start,
        goal,
        basic_mode_sequence=None,
        fig_name=None,
        anim_name=None,
        anim_online=False,
        plot=True,
        log=True,
    ):
        # Reset existing tree.
        self._log = log
        self._reset_tree()
        self._set_mode_sequence(basic_mode_sequence)
        # Add initial node.
        self._start = np.array(start, dtype=np.float32).squeeze()
        self._goal = np.array(goal, dtype=np.float32).squeeze()
        assert not self._collision.is_collision(self._start)
        assert not self._collision.is_collision(self._goal)
        _ = self._add_node(
            pose=self._start,
        )
        if plot:
            fig, axes, cid = self._set_plot_online()
        i = 1
        toggle = True
        while i < self._max_size:
            # Draw a collision free sample from state space.
            goal_selected, rnd = self._sample_collision_free()
            # Find nearest node
            (
                nearest_ind,
                nearest_pose,
                depths,
                mode_sequence,
            ) = self._nearest_node(rnd, goal_selected)
            new_ind = None
            if nearest_ind is not None:
                # Calculate path from nearest to rnd.
                cmds, poses, depths, mode_sequence = self._steer(
                    nearest_pose, rnd, depths, mode_sequence
                )
                # Check path for collision and remove colliding parts.
                (
                    is_collision,
                    i_collision,
                    time_collision,
                ) = self._collision.is_collision_lines(poses[:-1], poses[1:])
                #
                cmds, poses, depths, mode_sequence = self._remove_colliding(
                    cmds,
                    poses,
                    depths,
                    mode_sequence,
                    is_collision,
                    i_collision,
                    time_collision,
                )
                # Extend the tree.
                new_ind, goal_reached = self._extend(
                    nearest_ind,
                    cmds,
                    poses,
                    depths,
                    mode_sequence,
                    goal_selected,
                )
                inds_to_draw = [new_ind]
            # Draw the node.
            if new_ind is not None:
                toggle = True
                # Generate and update paths.
                inds_to_draw += self._update_paths(new_ind, goal_reached)
                # Draw Nodes.
                if plot:
                    self._set_arts_info(inds_to_draw)
                    self._draw_nodes(i, axes)
                i += 1
            if self._log and (not i % 100) and toggle:
                logger.debug(f"iteration = {i:>6d}")
                toggle = False
            #
            if anim_online and plot:
                plt.pause(0.001)

    def _set_legends_online(self, ax, robot):
        fontsize = 8
        modes = range(self._specs.n_mode)
        colors = self._colors
        #
        legend_robot = lambda m, c: plt.plot(
            [], [], ls="", marker=m, mfc=c, mec="k", ms=6
        )[0]
        legend_line = lambda c, ls: plt.plot([], [], color=c, ls=ls)[0]
        # Add start and goal legends.
        handles = [legend_robot("s", "yellow")]
        labels = ["Start"]
        handles += [legend_robot("s", "lime")]
        labels += ["Goal"]
        # Add modes legends.
        # handles += [legend_line(colors[mode], "-") for mode in modes]
        """ handles += [
            legend_line("deepskyblue", self._styles[mode][1]) for mode in modes
        ]
        labels += [f"M {mode}" for mode in modes] """
        handles += [legend_line("deepskyblue", "-")]
        labels += ["Not reached"]
        # Add suboptimal and optimal legends.
        handles += [legend_line("violet", "-"), legend_line("red", "-")]
        labels += ["Suboptimal", "Optimal"]
        #
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="upper right",
            fontsize=fontsize,
            borderpad=0.2,
            labelspacing=0.01,
            handlelength=2.0,
            handletextpad=0.2,
            borderaxespad=0.2,
        ).set_zorder(10)

    def _draw_robot(self, ax, robot, pose, color):
        ax.plot(
            pose[0],
            pose[1],
            ls="",
            marker="s",
            mfc=color,
            mec="k",
            zorder=10.0,
        )

    def _set_subplot_online(self, ax, robot):
        ax.clear()
        # Boundaries.
        bx, by = self._specs.bounds
        ax.set_xlim(*bx)
        ax.set_ylim(*by)
        # Adjust axes ticks and labels.
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(direction="in", zorder=10)
        #
        ax.grid(zorder=0.0)
        ax.set_aspect("equal", adjustable="box")
        self._set_legends_online(ax, robot)
        # Draw obstacles.
        for i, cnt in enumerate(self._obstacle_contours):
            ax.add_patch(Polygon(cnt, ec="k", lw=1, fc="slategray", zorder=2))
        # Draw start and goal positions.
        self._draw_robot(
            ax, robot, self._start.reshape(-1, 2)[robot], "yellow"
        )
        self._draw_robot(ax, robot, self._goal.reshape(-1, 2)[robot], "lime")

    def _set_plot_online(self, fig=None, axes=None, cid=-1):
        fig = None
        if fig is None:
            n_robot = self._specs.n_robot
            n_col = 3
            n_row = np.ceil(n_robot / n_col).astype(int)
            fig, axes = plt.subplots(n_row, n_col, layout="compressed")
            axes = axes.ravel()
            # Remove unnecessary axes.
            for i in range(n_robot, n_row * n_col):
                fig.delaxes(axes[i])
            axes = axes[:n_robot]
        # Set up subplots.
        for i, ax in enumerate(axes):
            self._set_subplot_online(ax, i)
        # Escape key ends simulation.
        if cid != -1:
            fig.canvas.mpl_disconnect(cid)
        else:
            cid = fig.canvas.mpl_connect(
                "key_release_event",
                lambda event: [exit(0) if event.key == "escape" else None],
            )
        return fig, axes, cid

    def _draw_line(self, ax, robot, poses, ltype=0):
        color, zorder = self._line_style_map.get(ltype)
        (art,) = ax.plot(
            poses[:, 0], poses[:, 1], lw=1.0, color=color, zorder=zorder
        )
        return art

    def _edit_line(self, art, robot, poses, ltype=0):
        color, zorder = self._line_style_map.get(ltype)
        art.set(
            xdata=poses[:, 0],
            ydata=poses[:, 1],
            color=color,
            zorder=zorder,
        )

    def _draw_node(self, axes, ind, tracks, ltype):
        arts = self._arts.get(ind)
        if arts is None:
            # Draw line and save the artist.
            arts = []
            for i, ax in enumerate(axes):
                arts.append(
                    self._draw_line(ax, i, tracks[:, 2 * i : 2 * i + 2], ltype)
                )
            self._arts[ind] = arts
        else:
            # Edit line data and color.
            for i, art in enumerate(arts):
                self._edit_line(art, i, tracks[:, 2 * i : 2 * i + 2], ltype)

    def _draw_nodes(self, iteration, axes):
        for ind, tracks, ltype in self._arts_info[iteration]:
            self._draw_node(axes, ind, tracks, ltype)

    def _get_stop_req(self):
        stop_req = False
        in_str = input("Enter Y to stop: ").strip()
        if re.match("[Yy]", in_str):
            stop_req = True
        return stop_req


class RRTS(RRT):
    def __init__(
        self,
        specs,
        collision,
        obstacle_contours,
        max_size=1000,
        tol_cmd=1e-2,
        goal_bias=0.07,
        tol_goal=1e-0,
    ) -> None:
        super().__init__(
            specs,
            collision,
            obstacle_contours,
            max_size,
            tol_cmd,
            goal_bias,
            tol_goal,
        )
        self._ndim = 2 * self._specs.n_robot  # Space dimension.
        self._k_s = 2 * np.exp(1)
        self._eps = 0.99  # Improvement threshold.

    def _k_nearest_neighbor(self, ind):
        pose = self._poses[ind]
        inds_searchable = np.nonzero(self._not_goal_mask[: self._N])[0]
        inds_selector = faiss.IDSelectorArray(inds_searchable)
        k = int(self._k_s * np.log(self._N)) + 1
        n_searchable = len(inds_searchable)
        k = k if k < n_searchable else n_searchable
        distances, inds = self._faiss.search(
            pose.reshape(1, -1),
            k,
            params=faiss.SearchParametersIVF(sel=inds_selector),
        )
        near_inds = inds[0][1:]
        return near_inds

    def _edit_node(
        self, *, ind, pose, depth, parent, cmds, cmd_modes, track, cost
    ):
        self._poses[ind] = pose
        self._depths[ind] = depth
        previous_parent = self._parents[ind]
        self._parents[ind] = parent
        self._commands[ind] = cmds
        self._command_modes[ind] = cmd_modes
        self._tracks[ind] = track
        self._costs[ind] = cost
        self._values[ind] = cost + self._values[parent]
        if previous_parent > -1:
            self._childs[previous_parent].remove(ind)
        self._childs[parent].append(ind)

    def _sort_inds(self, inds, values, costs, distances):
        combined = np.array(
            list(zip(inds, values, costs, distances)),
            dtype=[
                ("ind", int),
                ("value", int),
                ("cost", int),
                ("distance", float),
            ],
        )
        sorted_inds = np.argsort(combined, order=["value", "cost", "distance"])
        return inds[sorted_inds], values[sorted_inds], costs[sorted_inds]

    def _get_all_childs(self, ind):
        childs = []
        for index in self._childs[ind]:
            childs.append(index)
            childs += self._get_all_childs(index)
        return childs

    def _propagate_cost_to_childs(self, ind, value_diff):
        # Get list of indexes of all childs down the tree.
        childs = self._get_all_childs(ind)
        # Add value diff to all childs.
        self._values[childs] += value_diff

    def _try_rewiring(self, ind_i, ind_f, rewiring_near_nodes=False):
        rewired = False
        not_goal = self._not_goal_mask[ind_f]
        value_past = self._values[ind_f]
        # Obtain mode sequence and depths.
        depth_i = self._depths[ind_i]
        depths, mode_sequence = self._get_depth_mode_sequence(depth_i)
        # Calculate path from nearest to rnd.
        pose_f = self._poses[ind_f] if not_goal else self._goal
        cmds, poses, depths, mode_sequence = self._steer(
            self._poses[ind_i], pose_f, depths, mode_sequence
        )
        cost = len(cmds)
        if cost < 1:
            return rewired
        # Check if final pose is within goal or near_pose if needed.
        if not_goal:
            if rewiring_near_nodes and (
                not self._within_pose_tol(poses[-1] - pose_f)
            ):
                return rewired
        elif not self._within_goal(poses[-1]):
            return rewired
        # Check if rewiring improves near_node value.
        value_new = self._values[ind_i] + cost
        if not (value_new < value_past * self._eps):
            return rewired
        # Check path for collision and remove colliding parts.
        if self._collision.is_collision_path(poses):
            return rewired
        # Rewire new node.
        rewired = True
        self._edit_node(
            ind=ind_f,
            pose=poses[-1],
            depth=depths[-1],
            parent=ind_i,
            cmds=cmds,
            cmd_modes=mode_sequence,
            track=poses,
            cost=cost,
        )
        if rewiring_near_nodes:
            # Propagate cost to childs.
            value_diff = self._values[ind_f] - value_past
            self._propagate_cost_to_childs(ind_f, value_diff)
        return rewired

    def _rewire_new_node(self, near_inds, new_ind):
        new_value = self._values[new_ind]
        new_pose = self._poses[new_ind]
        near_poses = self._poses[near_inds]
        dposes = new_pose - near_poses
        rewiring_costs = self._cost(dposes)
        rewired_values = self._values[near_inds] + rewiring_costs
        # Filter based on estimated value.
        candidate_inds = np.where(rewired_values < new_value * self._eps)
        dposes = dposes[candidate_inds]
        distances = self._distance(dposes)
        rewiring_costs = rewiring_costs[candidate_inds]
        rewired_values = rewired_values[candidate_inds]
        candidate_inds = near_inds[candidate_inds]
        # Sort based on the estimated value.
        candidate_inds, rewired_values, rewiring_costs = self._sort_inds(
            candidate_inds, rewired_values, rewiring_costs, distances
        )
        for ind in candidate_inds:
            rewired = self._try_rewiring(
                ind, new_ind, rewiring_near_nodes=False
            )
            if rewired:
                break

    def _rewire_near_nodes(self, new_ind, near_inds):
        # Remove indices with higher rewired_values.
        new_pose = self._poses[new_ind]
        near_values = self._values[near_inds]
        near_poses = self._poses[near_inds]
        dposes = new_pose - near_poses
        rewiring_costs = self._cost(dposes)
        rewired_values = self._values[new_ind] + rewiring_costs
        candidate_inds = np.where(rewired_values < near_values * self._eps)
        candidate_inds = near_inds[candidate_inds]
        # Try rewiring.
        rewired_inds = []
        for ind in candidate_inds:
            rewired = self._try_rewiring(
                new_ind, ind, rewiring_near_nodes=True
            )
            if rewired:
                rewired_inds.append(ind)
        return rewired_inds

    def plans(
        self,
        start,
        goal,
        basic_mode_sequence=None,
        fig_name=None,
        anim_name=None,
        anim_online=False,
        plot=True,
        log=True,
    ):
        # Reset existing tree.
        self._log = log
        self._reset_tree()
        self._set_mode_sequence(basic_mode_sequence)
        # Add initial node.
        self._start = np.array(start, dtype=np.float32).squeeze()
        self._goal = np.array(goal, dtype=np.float32).squeeze()
        assert not self._collision.is_collision(self._start)
        assert not self._collision.is_collision(self._goal)
        _ = self._add_node(
            pose=self._start,
        )
        if plot:
            fig, axes, cid = self._set_plot_online()
        i = 1
        toggle = True
        while i < self._max_size:
            # Draw a collision free sample from state space.
            goal_selected, rnd = self._sample_collision_free()
            # Find nearest node
            (
                nearest_ind,
                nearest_pose,
                depths,
                mode_sequence,
            ) = self._nearest_node(rnd, goal_selected)
            new_ind = None
            if nearest_ind is not None:
                # Calculate path from nearest to rnd.
                cmds, poses, depths, mode_sequence = self._steer(
                    nearest_pose, rnd, depths, mode_sequence
                )
                # Check path for collision and remove colliding parts.
                (
                    is_collision,
                    i_collision,
                    time_collision,
                ) = self._collision.is_collision_lines(poses[:-1], poses[1:])
                #
                cmds, poses, depths, mode_sequence = self._remove_colliding(
                    cmds,
                    poses,
                    depths,
                    mode_sequence,
                    is_collision,
                    i_collision,
                    time_collision,
                )
                # Extend the tree.
                new_ind, goal_reached = self._extend(
                    nearest_ind,
                    cmds,
                    poses,
                    depths,
                    mode_sequence,
                    goal_selected,
                )
                inds_to_draw = [new_ind]
            # Rewire and draw the node.
            if new_ind is not None:
                toggle = True
                # Find near inds.
                near_inds = self._k_nearest_neighbor(new_ind)
                # Rewire.
                self._rewire_new_node(near_inds, new_ind)
                inds_to_draw += self._rewire_near_nodes(new_ind, near_inds)
                # Generate and update paths.
                inds_to_draw += self._update_paths(new_ind, goal_reached)
                # Draw Nodes.
                if plot:
                    self._set_arts_info(inds_to_draw)
                    self._draw_nodes(i, axes)
                i += 1
            if self._log and (not i % 1000) and toggle:
                logger.debug(f"iteration = {i:>6d}")
                toggle = False
            #
            if anim_online and plot:
                plt.pause(0.001)
            """ if goal_reached:
                break """


def test_obstacle():
    from matplotlib.patches import Polygon

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
    #
    simulation = model.Simulation(specs)
    pos = [-20.0, 0.0, 0.0, 0.0, 20.0, 0.0]
    cmds = [
        [10, np.radians(90), 1],
    ]
    # Run simulation.
    simulation.simulate(cmds, pos)
    # Draw simulation results.
    fig, ax = simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    # Draw mesh contours.
    for cnt in mesh_contours:
        ax.add_patch(Polygon(cnt, ec="lime", fill=False))
    plt.show(block=False)


def test_collision():
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

    N = 100
    pose_i = np.array([0.0, 0.0, -40.0, 20.0, 20.0, 0.0], dtype=float)
    pose_f = np.array([0.0, 75.0, 75.0, 20.0, 20.0, 74.0], dtype=float)
    poses = np.linspace(pose_i, pose_f, N + 1)
    #
    print(f"Inc {collision._is_collision_inc(poses)}")
    print(
        f"Line intra {collision._is_collision_line_intrarobot(pose_i, pose_f)}"
    )
    print(f"Intra {collision._is_collision_intrarobot(pose_i)}")
    print(
        f"Boundary line {collision._is_collision_line_boundary(pose_i, pose_f)}"
    )
    print(
        f"Boundary line {collision._is_collision_line_boundary(pose_i, pose_f*2)}"
    )
    print(f"Boundary {collision._is_collision_boundary(pose_i)}")
    print(f"Boundary {collision._is_collision_boundary(pose_f * 10)}")
    #
    print(f"is_collision {collision.is_collision(pose_i)}")
    print(f"is_collision {collision.is_collision(pose_f * 10)}")
    print(f"is_collision_line {collision.is_collision_line(pose_i, pose_f)}")


def test_rrt3():
    np.random.seed(42)  # Keep for consistency, but can be removed.
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

    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        goal_bias=0.05,
        max_size=4000,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, [0, 1, 2], anim_online=False, plot=True)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    cmds = model.cartesian_to_polar(rrt.cmds)
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        anim_length=1100,
        boundary=True,
        last_section=True,
    )
    plt.show()


def test_rrt4():
    np.random.seed(42)  # Keep for consistency, but can be removed.
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo4()
    specs_mod = model.SwarmSpecs.robo4()
    specs_mod.set_space(clearance=10)
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
    collision_mod = Collision(mesh, specs_mod)
    pose = np.array([50, 40, 50, 15, 50, -15, 50, -40], dtype=float)
    _ = collision.is_collision(pose)
    #
    N = 100
    pose_i = np.array([50, 40, 50, 15, 50, -15, 50, -40], dtype=float)
    pose_f = np.array([-50, 40.0, -50, 15, -50, -15, -50, -40], dtype=float)
    poses = np.linspace(pose_i, pose_f, N + 1)

    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        goal_bias=0.09,
        max_size=4000,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    # Process the command.
    cmds = rrt.cmds
    cmds = rrt.post_process(rrt.cmds, ang=240)  # Add mode change.
    cmds = model.cartesian_to_polar(cmds)  # Convert to polar.
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    poses, cmds = simulation.simulate(cmds, pose_i)
    logger.debug(
        f"is_collision_path: {collision_mod.is_collision_path(np.vstack(poses[:,0]))}"
    )
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        anim_length=1100,
        boundary=True,
        last_section=True,
    )
    plt.show()


def test_errt4(tol_cmd=15.0, goal_bias=0.09):
    np.random.seed(42)  # Keep for consistency, but can be removed.
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo(4)
    specs.set_space(rcoil=90)
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
    # pose_i = np.array([-43,40, -43,18, -45,-10, -36,-45], dtype=float)
    pose_f = np.array([-40, 45, -40, 15, -40, -15, -40, -45], dtype=float)
    #
    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=20000,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    """ # Process the command.
    cmds = rrt.cmds
    cmds = rrt.post_process(rrt.cmds, ang=10)  # Add mode change.
    cmds = model.cartesian_to_polar(cmds)  # Convert to polar.
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    poses, cmds = simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    ) """
    """ anim = simulation.simanimation(
        anim_length=1100,
        boundary=True,
        last_section=True,
    ) """
    # plt.show()


def test_rrt5():
    np.random.seed(42)  # Keep for consistency, but can be removed.
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

    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        goal_bias=0.07,
        max_size=50000,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    cmds = model.cartesian_to_polar(rrt.cmds)
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        anim_length=1100,
        boundary=True,
        last_section=True,
    )
    plt.show()


def test_rrt10_big():
    np.random.seed(42)  # Keep for consistency, but can be removed.
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
    #
    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        goal_bias=0.05,
        max_size=25000,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, np.arange(11), anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    cmds = model.cartesian_to_polar(rrt.cmds)
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        vel=30,
        anim_length=1100,
        boundary=True,
        last_section=True,
    )
    plt.show()


def test_rrt10(tol_cmd=0.01, goal_bias=0.12, max_size=200000):
    np.random.seed(42)  # Keep for consistency, but can be removed.
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
    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=max_size,
    )
    self = rrt
    start_time = time.time()
    rrt.plans(pose_i, pose_f, np.arange(11), anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    cmds = model.cartesian_to_polar(rrt.cmds)
    # Run simulation.
    simulation = model.Simulation(specs)
    # Simulate the system
    simulation.simulate(cmds, pose_i)
    # Draw simulation results.
    simulation.simplot(
        step=10,
        plot_length=1000,
        boundary=True,
        last_section=False,
    )
    anim = simulation.simanimation(
        vel=30,
        anim_length=1100,
        boundary=True,
        last_section=True,
    )
    plt.show()


########## test section ################################################
if __name__ == "__main__":
    # Create a file handler and set its level
    file_handler = logging.FileHandler("logfile.log", mode="w")
    file_handler.setLevel(logging.DEBUG)  # Writes logs to file.
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    # test_obstacle()
    # test_collision()
    # test_rrt4()
    plt.show()
