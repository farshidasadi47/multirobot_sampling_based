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
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerTuple
from scipy.optimize import root
import faiss
from matplotlib import animation

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
    """
    Find repetitive rows in a 2D array withing a given tolerance.

    Parameters
    ----------
    data: array_like
        A 2D array containing the data of interest.
    tol: float
        Tolerance for determining the repetitive rows.

    Returns
    -------
    groups: list of list
      A list where each sublist contains indices of repeated rows.
    """

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
    This class creates a mesh representation of the workspace and
    obstacles based on given dimensions and the external contours that
    define the obstacle outlines. This class uses triangle library for
    triangulation.

    Each mesh is represented as a tuple containing:
        - A 2D array of obstacle contour vertices.
        - A 2D array where each row represents a triangle, defined
            by indices referencing the vertices array.
        - A boolean indicating whether the obstacle is convex.

    Parameters
    ----------
    specs : object
        A SwarmSpecs object from model.py module containing robots
        specifications, SwarmSpecs also includes workspace limits.
    obstacle_contours : list of arrays, optional
        A list of 2D contours representing Cartesian coordinates of
        obstacle outlines (default is an empty list).
    with_boundary : bool, optional
        If True, it includes a meshes for workspace boundary (default is
        False).
    with_coil : bool, optional
        If True, it includes a mesh for magnetic coil (default is False)
        .

    Attributes
    ----------
    _specs : SwarmSpecs object
        The robots and workspace boundary specifications.
    _with_boundary : bool
        Flag indicating whether the boundary mesh is included.
    _with_coil : bool
        Flag indicating whether the coil mesh is included.
    _contours : list
        List of obstacle contours approximated to have less vertices and
        convex shape if possible without losing too much detail.

    Methods
    -------
    get_obstacle_mesh()
        Returns the triangulated obstacle mesh.
    get_cartesian_obstacle_contours()
        Returns the obstacle contours in Cartesian coordinates.
    _approximate_with_convex(contours, eps=0.1)
        Approximates given contours with their convex hulls if too much
        details is not getting lost based on a tolerance value.
    _get_mesh_obstacles(contours)
        Triangulates obstacles and returns the mesh representation.
    _get_space_mesh()
        Generates and returns the workspace boundary mesh.
    _get_coil_mesh()
        Approximates a circular coil with a polygon and returns its mesh
        representation.
    """

    def __init__(
        self, specs, obstacle_contours=[], with_boundary=False, with_coil=False
    ):
        """
        Class constructor.

        Parameters
        ----------
        specs : object
            A SwarmSpecs object from model.py module containing robots
            specifications, SwarmSpecs also includes workspace limits.
        obstacle_contours : list of arrays, optional
            A list of 2D contours representing Cartesian coordinates of
            obstacle outlines (default is an empty list).
        with_boundary : bool, optional
            If True, it includes a meshes for workspace boundary
            (default is False).
        with_coil : bool, optional
            If True, it includes a mesh for magnetic coil (default is
            False).
        """
        self._specs = specs
        self._with_boundary = with_boundary
        self._with_coil = with_coil
        eps = 0.1  # Convex approximation tolerance.
        self._contours = self._approximate_with_convex(obstacle_contours, eps)

    def get_obstacle_mesh(self):
        """
        Returns
        -------
        The list of triangulated obstacle mesh.
        """
        return self._get_mesh_obstacles(self._contours)

    def get_cartesian_obstacle_contours(self):
        """
        Returns
        -------
        The list of obstacle contours in Cartesian coordinates.
        """
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
        Triangulates obstacles and returns list of mesh and list of
        mech_contour.

        Each mesh is represented as a tuple containing:
            - A 2D array of obstacle contour vertices.
            - A 2D array where each row represents a triangle, defined
                by indices referencing the vertices array.
            - A boolean indicating whether the obstacle is convex.

        Parameters
        ----------
        contours: list of arrays
            List of 2D arrays of contours outlining obstacles.

        Returns
        -------
        mesh: list of tuples
            List of mesh tuples for each object.
        mech_contours: list of arrays
            List of contours of all triangles in the returned mesh for
            all objects.
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
        """
        Calculates and gives space boundaries mesh representation.
        See _get_mesh_obstacles docstring for further explanation.

        Returns
        -------
        mesh: list of mesh tuples
        mesh_contours: list of arrays
        """
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
        Calculates and gives magnetic coil mesh representation.
        See _get_mesh_obstacles docstring for further explanation.

        Returns
        -------
        mesh: list of mesh tuples
        mesh_contours: list of arrays
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
    """
    A class for handling collision detection with other robots,
    environment obstacles, and environment boundaries. This class uses
    the robot specifications and workspace boundaries in SwarmSpecs
    object instance along with mesh produced by Obstacles object
    instanceand uses python-fcl library for collision detection.

    Check python-fcl documentation for how to use it:
        https://github.com/BerkeleyAutomation/python-fcl

    Parameters
    ----------
    mesh : list of tuples
        A list of obstacle mesh representations. Each tuple contains:
            - A 2D array of obstacle contour vertices.
            - A 2D array where each row defines a triangle by
                referencing vertex indices.
            - A boolean indicating whether the obstacle is convex.
    specs : SwarmSpecs instance
        An object containing specifications of robot and such as
        obstacle clearance, minimum distance, and workspace boundaries.
    with_coil : bool, optional
        If True, it considers collision checks with magnetic coitl
        (default is True).

    Attributes
    ----------
    T0 : fcl.Transform
        Default transformation for collision checks, see python-fcl doc.
    mesh : list of tuples
        The current obstacle mesh representation.
    rrob : float
        The robot radius for obstacle clearance check.
    rmin : float
        Half of the minimum allowed distance between robots, adjusted
        slightly to allow contact.
    dmin : float
        Minimum allowed distance between robots.
    dmin2 : float
        Square of `dmin`, used for efficiency in collision checks.
    robot_pairs : list of tuples
        A list defining which robots should be checked for intrarobot
        collisions.
    _specs : object
        The provided robot specifications SwarmSpecs instance.
    ball0 : fcl.Sphere
        Represents the robots for collision detection with obstacles.
    obstacles : list of fcl.CollisionObject
        A list of obstacle collision objects according to python-fcl.
    cmanager : fcl.DynamicAABBTreeCollisionManager
        Collision manager for handling obstacle collisions.
    col_req : fcl.CollisionRequest
        Collision request object that holds collision results.
    dis_req : fcl.DistanceRequest
        Distance request object that holds minimum distance results.
    _with_coil : bool
        Indicates whether coil boundary collision checks are enabled.

    Methods
    -------
    update_obstacles(mesh)
        Builds and stores obstacle objects from list of obstacle mesh.
    is_collision(pose)
        Checks if a given robot position is in collision.
    is_collision_line(pose_i, pose_f)
        Checks for collisions along a line segment between two robot
        positions.
    is_collision_lines(poses_i, poses_f)
        Checks for collisions along multiple line segments.
    is_collision_path(poses)
        Checks if any part of a given path results in a collision.
    """

    def __init__(self, mesh, specs, with_coil=True):
        """
        Class constructor.

        Parameters
        ----------
        mesh: list of mesh tuples.
            List of obstacle mesh calculated from Obstacle instance.
        specs: SwarmSpecs
            Specification of robots and boundary of workspace.
        with_coil: bool
            If True, magnetic coil is considered in collision detection
             (default is True).
        """
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
        """
        Updates the mesh and build obstacle fcl objects.

        Parameters
        ----------
        mech: list of mesh tuples
            Obstacle mesh list from Obstacles object.
        """
        self.mesh = mesh
        self.obstacles = self._build_obstacles(mesh)

    def _build_obstacles(self, mesh):
        """
        Builds fcl obstacle objects from given mesh of obstacles.

        Parameters
        ----------
        mesh: list of mesh tuples
            Mesh tuples of obstacles from Obstacle object.

        Returns
        -------
        obstacle: list of fcl obstacle objects
            List of obstacle objects used by python-fcl.
        """
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
        """
        Buils fcl collision manager, collision request, and distance
        request.

        Returns
        -------
        fcl collision manager
        fcl collision request
        fcl distance request
        """
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
        """
        Tells is a given position is in collision.

        Parameters
        ----------
        pose: array
            1D array of robots positions.

        Returns
        -------
        bool
            True if there is collision, else False.
        """
        return (
            self._is_collision_boundary(pose)
            or self._is_collision_intrarobot(pose)
            or self._is_collision(pose.reshape(-1, 2))[0]
        )

    def is_collision_line(self, pose_i, pose_f):
        """
        Checks for collision and time of it along the path between two
        robot positions.

        Parameters
        ----------
        pose_i : array-like
            1D array of initial positions of the robots.
        pose_f : array-like
            1D array of final positions of the robots.

        Returns
        -------
        is_collision : bool
            True if a collision is detected, False otherwise.
        time_collision : float
            The earliest time at which a collision occurs, with 0
            indicating an immediate collision and 1 meaning no collision
            occurs along the path.
        """
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
        """
        This method evaluates whether any collision occurs while moving
        from a set of initial positions (poses_i) to corresponding final
        positions (poses_f). It iterates through each pair of start and
        end points, stopping early if a collision is detected.

        Parameters
        ----------
        poses_i : array-like
            A 2D array of initial positions.
        poses_f : array-like
            A 2D array of final positions, corresponding to poses_i.

        Returns
        -------
        is_collision : bool
            True if any collision is detected, False otherwise.
        i_collision : int or None
            The index of the first collision detected, or None if no
            collision occurs.
        time_collision : float
            The earliest time at which a collision occurs in i_collision
            section of path, with 0 indicating an immediate collision
            and 1 meaning no collision occurs along any path.
        """
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
        """
        Checks if a given path results in a collision. It excludes start
        and end point from collision check.

        Parameters
        ----------
        poses : array-like
            A 2D array of positions representing the path.

        Returns
        -------
        bool
            True if collision occurs along the path, False otherwise.
        """
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
        Checks if any robot pairs are colliding with each other.

        Parameters
        ----------
        pose : array-like
            1D arrays of robots positions.

        Returns
        -------
        bool
            True if any robots are in collision, False otherwise.
        """
        is_collision = False
        pose = np.reshape(pose, (-1, 2))
        dpose = np.diff(pose[self._specs.robot_pairs], axis=1).squeeze()
        D2 = np.einsum("ij,ij->i", dpose, dpose)
        is_collision = (D2 < self.dmin2).any()
        return is_collision

    def _is_collision_boundary(self, pose):
        """
        Checks if any robot collides with work workspace boundary.

        Parameters
        ----------
        pose : array-like
            1D arrays of robots positions.

        Returns
        -------
        bool
            True if any robots are in collision, False otherwise.
        """
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
        """
        Checks if a given list of robots positions collides with
        obstacles.

        Parameters
        ----------
        poses : array-like
            A 2D array where each row represents a robot system
            positions.

        Returns
        -------
        tuple
            (bool, int) :
            - True if a collision is detected, False otherwise.
            - Index of the last collision-free row of positions.
        """
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
        """
        Incrementally checks for collisions along array of positions.

        Parameters
        ----------
        poses : array-like
            2D array of sequence of robots positions.

        Returns
        -------
        tuple
            (bool, int) :
            - True if a collision occurs, False otherwise.
            - Index of the last collision-free of positions sequence.
        """
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
        """
        Checks for collisions with obstacles along a linear path between
        two given robots positions.

        Parameters
        ----------
        pose_i : array-like
            1D array of robots initial positions.
        pose_f : array-like
            1D array of robots final positions.

        Returns
        -------
        tuple
            (bool, float) :
            - True if a collision occurs, False otherwise.
            - Earliest time of collision, rounded down.
        """
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
        Checks for collisions between robot pairs along a linear path
        between two given robots positions.

        Parameters
        ----------
        pose_i : array-like
            1D array of robots initial positions.
        pose_f : array-like
            1D array of robots final positions.

        Returns
        -------
        tuple
            (bool, float) :
            - True if a collision occurs, False otherwise.
            - Earliest time of collision, rounded down.
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
        """
        Checks for collisions with workspace boundary along a linear
        path between two given robots positions.

        Parameters
        ----------
        pose_i : array-like
            1D array of robots initial positions.
        pose_f : array-like
            1D array of robots final positions.

        Returns
        -------
        tuple
            (bool, float) :
            - True if a collision occurs, False otherwise.
            - Earliest time of collision, rounded down.
        """
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
    """
    Adapter Rapidly-exploring Random Tree (Adapted RRT) algorithm for
    path planning for a class of heterogeneous robotic systems.

    Attributes
    ----------
    Path : TypedDict
        A dictionary-like object that stores details of a path.
    _specs : SwarmSpecs object
        Contains robot specifications and workspace boundaries.
    _collision : Collision object
        A collision-checking module.
    _obstacle_contours : list
        A list of obstacle boundary contours.
    _max_size : int
        Maximum number of nodes allowed in the tree.
    _max_iter : int
        Maximum iterations allowed for RRT search.
    _tol_cmd : float
        Tolerance value for removing zero pseudo displacements.
    _goal_bias : float
        Probability of sampling the goal directly.
    _tol_goal : float
        Tolerance for considering a node has reached the goal.
    _start : np.ndarray
        The starting position of the robots.
    _goal : np.ndarray or None
        The goal position of the robots.
    _mode_sequence : np.ndarray or None
        The basic mode sequence of the tree.
    _WI : np.ndarray
        Pseudo inverse of robot controllability matrix.
    _WIT : np.ndarray
        The transpose of _WI used for least square control solution.
    _lb : np.ndarray
        The lower bounds of the robot's configuration space.
    _ub : np.ndarray
        The upper bounds of the robot's configuration space.
    _tol_pose : float
        The tolerance for avoiding duplicate points.
    _tol_goal : float
        The tolerance for determining if the robot has reached the
        goal.
    _log : bool
        A flag for showing the log info during the RRT expansion.
    _n_state : int
        The state dimension of the robotic systems (twice number of
        robots).
    _poses : np.ndarray
        A 2D array where each row represents the position of a tree
        node at that row's index.
    _depths : np.ndarray
        2D array where each row represents a tree node's depths. By
        depth it actually means the index of next immediate mode in
        the basic mode sequence that will be used for expanding from
        the node.
    _parents : np.ndarray
        1D array where each elemnt is the index of parent node.
    _commands : list
        A list of 2D arrays, where each array is the commands that
        takes the node's parent to the node.
    _command_modes : list
        A list of arrays representing the modes of _commands.
    _tracks : list
        A list of 2D arrays representing the path from the node's
        parent to the node using the corresponding _commands and
        _command_modes.
    _costs : np.ndarray
        A array where each element is the cost of going node's
        parent to the node.
    _values : np.ndarray
        1D array where each element is the cost of reaching the node
        at that index.
    _childs : tuple of lists
        Each list is the list of children's of the corresponding
        node.
    _ltypes : np.ndarray
        The type of the immediate path toward the corresponding node
        (normal, suboptimal, or optimal), used for visualization.
    _faiss : faiss.IndexFlatL2
        A Faiss index for fast nearest-neighbor search in the tree.
    _not_goal_mask : np.ndarray
        A boolean mask that is True when the node is not goal.
    _not_expanded_to_goal_mask : np.ndarray
        A boolean mask that is True if the node is not yet selected
        for expansion toward goal.
    _goal_inds : list
        A list of indices of nodes that reached the goal.
    _goal_ind_values : dict
        A dictionary mapping goal node indices to their path costs.
    best_value : float
        The best path value found so far.
    _best_ind : int
        The index of the best goal node found.
    _paths : list
        A list of valid paths found during the planning process.
    _best_path_ind : int
        The index of the best path found in the list of paths.
    cmds : np.ndarray
        The array of commands for the best path found.
    _arts_info : list
        A list of information about the graphical representations of
        nodes.
    _arts : dict
        A dictionary storing matplotlib artist of the corresponding
        node. Keys are node indices.
    _line_style_map : dict
        A mapping of node line styles (normal, suboptimal, or
        optimal).

    Methods
    -------
    _reset_tree():
        Resets the tree structure used in RRT.
    _set_mode_sequence(basic_mode_sequence=None)
        Sets the basic mode sequence for motion planning.
    _get_next_mode_sequence(modes)
        Generates the next mode sequence.
    print_node(ind)
        Prints information about a specific node in the tree.
    _add_node(
        pose, depth=0, parent=-1, cmds, cmd_modes, track, cost,
        goal_reached)
        Adds a new node to the tree and updates relevant attributes.
    _sample()
        Samples a random position in the environment.
    _sample_collision_free()
        Samples a random collision-free position.
    _distance(arr)
        Computes the Euclidean norm of a given displacement array.
    _cost(arr):
        Computes the movement cost based on a displacement array.
    _get_mode(depth)
        Returns the motion mode for a given depth.
    _get_depth_mode_sequence(depth_i, depth_f=None)
        Computes depth indices and mode sequences for a given range.
    _nearest_node(pose, goal_selected)
        Finds the nearest node in the tree to a given point.
    _dynamics(pose_i, cmds, mode_sequence)
        Simulates the system dynamics to compute next states.
    _remove_zero_cmds(cmds, depths, mode_sequence)
        Filters out zero displacement commands from a motion sequence.
    _steer(pose_i, pose_f, depths, modes)
        Generates a sequence of commands and poses from an initial to a
        final pose.
    _within_pose_tol(dpose)
        Checks if a displacement is within tolerance limits.
    _within_goal(pose)
        Checks if a given pose is within the goal region.
    _remove_colliding(
        cmds, poses, depths, mode_sequence, is_collision, i_collision,
        time_collision)
        Removes sections of a path that are in collision.
    _extend(ind_i, cmds, poses, depths, mode_sequence, goal_selected)
        Extends the tree by adding new nodes and checking goal
        conditions.
    _generate_path(ind)
        Generates a path from the root node to a given node index.
    _set_cmds()
        Updates best path pseudo displacement if any path is found.
    _set_ltype(inds, ltype=0)
        Sets the label type for nodes (normal, suboptimal, optimal).
    _add_new_path(ind)
        Adds a new feasible path to the stored list of paths.
    _update_paths_all(new_ind, goal_reached)
        Updates all stored paths when a new node is added.
    _update_paths_best(new_ind, goal_reached)
        Updates only the best path in the stored paths.
    _update_paths(new_ind, goal_reached)
        Calls the best path update function when a new node is added.
    _set_arts_info(inds)
        Updates information related to visualization artifacts.
    _accurate_tumbling(cmd)
        Converts given tumbling into two experimentally executable
        tumbling such that the mode of robots remains the same after
        executing them.
    post_process(cmds, ang=0)
        Post-processes the command by adding mode changes and modifying
        tumblings to be executable experimentally.
    plan(
        start, goal, basic_mode_sequence=None, fig_name=None,
        anim_name=None, anim_online=False, plot=False, log=True,
        lazy=False)
        Executes the adapted RRT algorithm to find a path.
    _set_legends_online(ax, robot)
        Adds legends to the online visualization.
    _draw_robot(ax, robot, pose, color)
        Draws a robot at a given pose on the plot.
    _set_subplot_online(ax, robot)
        Sets up subplots for visualization.
    _set_plot_online(fig=None, axes=None, cid=-1)
        Initializes an interactive plot for real-time visualization.
    _draw_line(ax, robot, poses, ltype=0)
        Draws a line representing a path segment.
    _edit_line(art, robot, poses, ltype=0)
        Updates an existing line with new data.
    _draw_node(axes, ind, tracks, ltype)
        Draws a tree node in the visualization.
    _draw_nodes(iteration, axes)
        Draws all nodes in the tree for a given iteration.
    _get_stop_req()
        Checks user input for stopping the planning process.
    """

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
        max_iter=500000,
        tol_cmd=1e-2,
        goal_bias=0.07,
        tol_goal=1e-0,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        specs : SwarmSpecs object
            Specifications of robots and workspace boundary.
        collision : Collision object
            Collision checker.
        obstacle_contours : list of arrays
            List of 2D arrays of obstacle outline contours.
        max_size : int, optional, default=1000
            The maximum number of adapted RRT nodes.
        max_iter : int, optional, default=500000
            The maximum number of iterations to run the adapted RRT.
        tol_cmd : float, optional, default=1e-2
            The tolerance for removing zero displacements.
        goal_bias : float, optional, default=0.07
            The probability of sampling the goal state for tree
            expansion.
        tol_goal : float, optional, default=1e-0
            The tolerance for determining if the goal has been reached.
        """
        self._start = None
        self._goal = None
        self._mode_sequence = None
        #
        self._specs = specs
        self._collision = collision
        self._obstacle_contours = obstacle_contours
        self._WI = np.linalg.pinv(specs.W)
        # Tree properties.
        self._max_size = max_size
        self._max_iter = max_iter
        lb, ub = map(np.array, list(zip(*specs.bounds)))
        self._lb = np.tile(lb, specs.n_robot)
        self._ub = np.tile(ub, specs.n_robot)
        # Resolutions and tolerances.
        self._tol_cmd = tol_cmd  # Tolerance for checking zero inputs.
        self._tol_pose = 1e-6  # Tolerance for checking repetition.
        self._goal_bias = goal_bias
        self._tol_goal = tol_goal
        self._eps = 0.1  # Improvement threshold.
        #
        self._log = True
        self._reset_tree()
        #
        define_colors(self)
        self._colors = list(self._colors.keys())

    def _reset_tree(self):
        """
        Resets and reinitializes the tree.
        """
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
            0: ("dodgerblue", 4),  # Normal.
            1: ("lime", 6),  # Suboptimal.
            2: ("red", 10),  # Optimal.
        }

    def _set_mode_sequence(self, basic_mode_sequence=None):
        """
        Sets the basic mode sequence for the planner.

        Parameters
        ----------
        basic_mode_sequence : list of int, optional
            A predefined sequence of mode indices that includes all
            modes of the system without repetition. If None, the default
            sequence [0, 1, ..., n_mode-1] is used.
        """
        # Modify tumble_index.
        n_mode = self._specs.n_mode
        if basic_mode_sequence is None:
            basic_mode_sequence = [0] + list(range(1, n_mode))
        self._basic_mode_sequence = np.array(basic_mode_sequence)
        self._n_basic_mode_sequence = len(self._basic_mode_sequence)

    @staticmethod
    def _get_next_mode_sequence(modes):
        """
        Generates the next mode sequence based on cycling the given mode
        sequence. The next mode for zero modes is considered zero.

        Parameters
        ----------
        modes : list or array-like of int
            A mode sequence.

        Returns
        -------
        np.ndarray
            Next mode sequence.
        """
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
        """
        Prints the details of a node in the tree.

        Parameters
        ----------
        ind : int
            Index of the node to print. The index is clipped _max_size.
        """
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
        """
        Adds a new node to the tree.

        Parameters
        ----------
        pose : array-like
            The position of the new node.
        depth : int, optional (default=0)
            Depth of the node in the tree. Depth is the index of next
            mode in the basic mode sequence.
        parent : int, optional (default=-1)
            Index of the parent node. If -1, the node has no parent.
        cmds : ndarray, optional (default=zeros((1, 2)))
            Array of commands that move the parent to this node.
        cmd_modes : ndarray, optional (default=zeros(1, dtype=int))
            Array indicating the command modes corresponding to cmds.
        track : object, optional
            The path positions from parent node to this node.
        cost : float, optional (default=0)
            Cost from parent node to this node.
        goal_reached : bool, optional (default=False)
            True if this node has reached the goal position.

        Returns
        -------
        int
            The index of the newly added node.

        Notes
        -----
        - Updates general node related tree attributes.
        - If the goal is reached, updates goal related tree attributes.
        """
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
        """
        Samples a random point in the search space or selects the goal
        with a probability defined by _goal_bias.

        Returns
        -------
        tuple (bool, ndarray)
            - A boolean indicating whether the goal was selected.
            - A 2D sampled position.
        """
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
        """
        Samples a random collision-free point in the search space.

        Returns
        -------
        tuple (bool, ndarray)
            - A boolean indicating whether the goal is sampled.
            - A collision free sampled robot position.
        """
        is_collision = True
        while is_collision:
            is_goal_selected, rnd = self._sample()
            is_collision = self._collision.is_collision(rnd)
        return is_goal_selected, rnd

    def _distance(self, arr):
        """
        Computes the Euclidean norm (L2 norm) for rows of a given array.
        This is used to calculate distance between pose_1 and pose_2
        by feedeing [arr] where arr is pose_2 - pose_1

        Parameters
        ----------
        arr : ndarray
            2D array where each row is a dpose:= pose_1 - pose_2 vector.

        Returns
        -------
        ndarray
            Array of Euclidean distances for each row of arr.
        """
        arr = np.atleast_2d(arr)
        return np.linalg.norm(arr, axis=1)

    def _cost(self, arr):
        """
        Computes the cost estimate for rows of arrays of displacements.
        This estimates the number of steps that it takes to go from
        pose_1 to pose_2 based on least square solution that is
        thresholded by _cmd_tol to remove zero length steps.

        Parameters
        ----------
        arr : ndarray
            2D array where each row is a dpose:= pose_1 - pose_2 vector.

        Returns
        -------
        ndarray
            Array of costs for each row of arr.
        """
        arr = np.atleast_2d(arr)
        u = np.einsum("ij, hj->hi", self._WI, arr).reshape(
            -1, self._specs.n_mode, 2
        )
        return np.sum(
            np.linalg.norm(u, axis=-1) > self._tol_cmd, axis=1, dtype=float
        )

    def _get_mode(self, depth):
        """
        Gives the mode corresponding to a given depth.

        Parameters
        ----------
        depth : int
            The depth for which to determine the mode.

        Returns
        -------
        int
            The mode corresponding to the given depth, determined
            cyclically.
        """
        return self._basic_mode_sequence[depth % self._n_basic_mode_sequence]

    def _get_depth_mode_sequence(self, depth_i, depth_f=None):
        """
        Generates a sequence of depths and corresponding modes.

        Parameters
        ----------
        depth_i : int
            The initial depth value.
        depth_f : int, optional
            The final depth value (not used explicitly in the function).

        Returns
        -------
        tuple of ndarray
            - depths : ndarray
                Array of depth values starting from `depth_i`.
            - mode_sequence : ndarray
                Corresponding mode sequence for the depth values.
        """
        depths = np.arange(depth_i, depth_i + self._n_basic_mode_sequence)
        mode_sequence = self._get_mode(depths)
        depths = depths + 1
        return depths, mode_sequence

    def _nearest_node(self, pose, goal_selected):
        """
        Finds the nearest node in the tree to a given pose.

        Parameters
        ----------
        pose : ndarray
            The given position to find the nearest node to.
        goal_selected : bool
            True if pose is the goal position.

        Returns
        -------
        tuple
            - nearest_ind : int or None
                Index of the nearest node.
            - nearest_pose : ndarray or None
                The position of the nearest node.
            - depths : ndarray or None
                Depth values for the mode sequence.
            - mode_sequence : ndarray or None
                Mode sequence corresponding to depths.

        Notes
        -----
        If searching nearest node to the goal position (goal_selected
        is True), but no node is left in the tree that hasn't been
        previously selectiod fo expansion toward goal, then it returns a
        tuple of None. This is to avoid choosing the same nearest node
        repeatedly when expanding toward goal position.
        """
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
        """
        Computes the sequence of poses given an initial pose, commands,
        and mode sequence.

        Parameters
        ----------
        pose_i : ndarray
            Initial position of robots.
        cmds : ndarray
            2D array of commands, each row is a pseudo displacement to
            be applied sequentially.
        mode_sequence : ndarray
            Sequence of modes corresponding to each row of command.

        Returns
        -------
        ndarray
            Array of positions generated by applying the commands
            sequentially.
        """
        poses = [pose_i]
        for mode, cmd in zip(mode_sequence, cmds):
            poses.append(poses[-1] + self._specs.B[mode] @ cmd)
        poses = np.array(poses)
        return poses

    def _remove_zero_cmds(self, cmds, depths, mode_sequence):
        """
        Removes zero commands from the command sequence.

        Parameters
        ----------
        cmds : ndarray
            Array of pseudo displacements.
        depths : ndarray
            Depths corresponding to each row of cmds.
        mode_sequence : ndarray
            Mode sequence corresponding to each row of cmds.

        Returns
        -------
        tuple of ndarrays
            - cmds : ndarray
                Filtered array of nonzero commands.
            - depths : ndarray
                Corresponding depths of the nonzero commands.
            - mode_sequence : ndarray
                Mode sequence of the nonzero commands.
        """
        mask = np.linalg.norm(cmds, axis=1) > self._tol_cmd
        cmds = cmds[mask]
        depths = depths[mask]
        mode_sequence = mode_sequence[mask]
        return cmds, depths, mode_sequence

    def _steer(self, pose_i, pose_f, depths, modes):
        """
        Implements sparse least square algorithm to find a sparse
        sequence of displacement that is taking pose_i to pose_f using
        the provided mode sequence.

        Parameters
        ----------
        pose_i : ndarray
            Initial position of robots.
        pose_f : ndarray
            Final position of robots.
        depths : ndarray
            Depth values corresponding to the mode sequence.
        modes : ndarray
            Mode sequence to use.

        Returns
        -------
        tuple of ndarrays
            - cmds : ndarray
                Array of pseudo displacements.
            - poses : ndarray
                Sequence of posisions from pose_i to pose_f using cmds.
            - depths : ndarray
                Sequence of depths corresponding to cmds rows.
            - modes : ndarray
                Mode sequence correspondint to cmds.

        Notes
        -----
        - cmds are removed if their length is smaller than _tol_cmd.
        """
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
        """
        Checks if the dpose is within the _tol_pose.

        Parameters
        ----------
        dpose : ndarray
            1D array representing position difference dpose.

        Returns
        -------
        bool
            True if within tolerance, else False.
        """
        return np.all(np.hypot(dpose[::2], dpose[1::2]) <= self._tol_pose)

    def _within_goal(self, pose):
        """
        Determines if a position is within _tol_goal of goal position.

        Parameters
        ----------
        pose : ndarray
            The robots position.

        Returns
        -------
        bool
            True if within tolerance, else False.
        """
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
        """
        Removes colliding parts of a path.

        Parameters
        ----------
        cmds : ndarray
            Array of path pseudo displacement sequence.
        poses : ndarray
            Array of path positions.
        depths : ndarray
            Aray of path depths.
        mode_sequence : ndarray
            Mode sequence of the path.
        is_collision : bool
            Indicates whether a collision was detected.
        i_collision : int
            Index of path section where first collision happened.
        time_collision : float
            Time fraction at which the collision detected in i_collision
            section.

        Returns
        -------
        tuple of ndarrays
            - cmds : ndarray
                Cleaned out cmds.
            - poses : ndarray
                Correspondinf cleaned out array of path positions.
            - depths : ndarray
                Corresponding cleaned out depths.
            - mode_sequence : ndarray
                Corresponding cleaned out mode sequence.
        """
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
        """
        Extends the tree by adding a given new node, if the position
        does not already exist in the tree.

        Parameters
        ----------
        ind_i : int
            Index of the parent node in the tree.
        cmds : ndarray
            Array of pseudo displacements taking parent node to new
            node.
        poses : ndarray
            Array of positions along path from parent node to new node.
        depths : ndarray
            Sequence of depths along path from parent node to new node.
        mode_sequence : ndarray
            The mode sequence used to take parent node to new node.
        goal_selected : bool
            Wheter this was result to extension toward goal position.

        Returns
        -------
        tuple
            - new_ind (int or None):
                Index of the newly added node, or None if not added.
            - within_goal (bool):
                True if the new node has reached goal position.
        """
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
        """
        Gives the path from the start node to the specified node. It is
        intended to be used with indexes that reached goal position.

        Parameters
        ----------
        ind : int
            Index of the node to give the path to.

        Notes
        -----
        - The path is stored in _paths attribute.
        - Logs the number of nodes and path value if logging is enabled.
        """
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
        """
        Updates the pseudo displacement and mode sequence for the best
        path and stores it in cmds instance variable.
        """
        path = self._paths[self._best_path_ind]
        cmds = path["cmds"]
        modes = path["cmd_modes"]
        cmds, _, modes = self._remove_zero_cmds(cmds, modes, modes)
        self.cmds = np.hstack((cmds, modes[:, None]))

    def _set_ltype(self, inds, ltype=0):
        """
        Updates the label type for the given node indices. The line type
        is used for the RRT visualization.

        Parameters
        ----------
        inds : array-like
            Indices of nodes to update.
        ltype : int, optional
            Label type (default is 0).
        """
        self._ltypes[inds] = ltype

    def _add_new_path(self, ind):
        """
        Adds a new path to the path list and updates the best path if
        applicable.

        Parameters
        ----------
        ind : int
            Index of the last node of newly added path.

        Returns
        -------
        list of int
            Indices of nodes to redraw.
        """
        ltype = 1
        inds_to_draw = []
        if self._values[ind] < self.best_value - self._eps:
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
        """
        Updates all goal-reaching paths and checks for improvements.
        Not used in the implementation due to massive computations.

        Parameters
        ----------
        new_ind : int
            Index of the newly added node.
        goal_reached : bool
            Whether the goal was reached by the new node.

        Returns
        -------
        list of int
            Indices of nodes to draw or redraw.
        """
        # Check for updated paths.
        past_values = np.array(
            [self._goal_ind_values.get(ind) for ind in self._goal_inds]
        )
        inds_to_update = np.where(
            self._values[self._goal_inds] < past_values - self._eps
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
        """
        Updates only the best path if a better one is found.

        Parameters
        ----------
        new_ind : int
            Index of the newly added node, not used.
        goal_reached : bool
            Whether the goal was reached by the new node, not used.

        Returns
        -------
        list of int
            Indices of nodes to redraw.
        """
        inds_to_draw = []
        if len(self._goal_inds):
            ind_best = self._values[self._goal_inds].argmin()
            ind_best = self._goal_inds[ind_best]
            if self._values[ind_best] < self.best_value - self._eps:
                # A better path has been discovered.
                inds_to_draw += self._add_new_path(ind_best)
        return inds_to_draw

    def _update_paths(self, new_ind, goal_reached):
        """
        Updates paths using the _update_best_path.

        Parameters
        ----------
        new_ind : int
            Index of the newly added node.
        goal_reached : bool
            Whether the goal was reached by the new node.

        Returns
        -------
        list of int
            Indices of nodes to redraw.

        Notes
        -----
        - Can be overridden to use _update_paths_all if needed.
        """
        return self._update_paths_best(new_ind, goal_reached)

    def _set_arts_info(self, inds):
        """
        Stores matplotlib artists related to nodes.

        Parameters
        ----------
        inds : array-like
            Indices of nodes to store information for.

        Notes
        -----
        - Collects and stores node indices, tracks, and label types.
        - Ensures unique indices before storing.
        """
        inds = np.unique(inds).tolist()
        tracks = [self._tracks[ind] for ind in inds]
        ltypes = self._ltypes[inds]
        self._arts_info.append(list(zip(inds, tracks, ltypes)))

    def _accurate_tumbling(self, cmd):
        """
        Arbitrary tumblings cannot be directly performed since tumbling
        each tumbling results in discrete displacements. This method
        converts a given tumbling into two seperate pieces of complete
        tumblings. The total number of tumblings is forced to be even so
        the robots modes stays the same after performing real tumblings.

        Parameters
        ----------
        cmd : array-like, shape (2,)
            The desired tumbling.

        Returns
        -------
        u_possible : ndarray, shape (2, 3)
            A sequence of two tumbling pseudo displacements with total
            even number of tumbles.
        """

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
        """
        Post-processes commands by adding required mode changed and
        modifying tumblings.

        Parameters
        ----------
        cmds : ndarray, shape (N, 3)
            Array of pseudo displacements in different modes.
        ang : float, optional
            Angle that the mode change should be performed along with.

        Returns
        -------
        ndarray, shape (M, 3)
            The modified pseudo displacement sequence.
        """
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
        plot=False,
        log=True,
        lazy=False,
    ):
        """
        Plans a path from a start position to a goal position using a
        Adapter RRT approach.

        Parameters
        ----------
        start : array-like
            The robots start position.
        goal : array-like
            The robots goal position.
        basic_mode_sequence : array-like, optional
            The sequence of movement modes to follow (default is None).
        fig_name : str, optional
            Filename for saving the tree figure (default is None).
        anim_name : str, optional
            Filename for saving the tree animation (default is None).
        anim_online : bool, optional
            Whether to animate the planning process in real-time
            (default is False).
        plot : bool, optional
            Whether to plot the search tree during execution
            (default is False).
        log : bool, optional
            Whether to log progress messages (default is True).
        lazy : bool, optional
            Whether to stop immediately when the goal is reached
            (default is False).
        """
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
        ranim = None
        if plot or anim_name:
            fig, axes, cid = self._set_plot_online()
        i = 1
        n_nodes = 1
        toggle = True
        while n_nodes < self._max_size and i < self._max_iter:
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
                # Update indices to draw.
                if plot or anim_name:
                    self._set_arts_info(inds_to_draw)
                # Draw Nodes.
                if plot:
                    self._draw_nodes(n_nodes, axes)
                n_nodes += 1
            i += 1
            if self._log and (not n_nodes % 1000) and toggle:
                logger.debug(f"# nodes = {n_nodes:> 6d}, iteration = {i:>7d}")
                toggle = False
            #
            if anim_online and plot:
                plt.pause(0.001)
            # Stop if goal reached andlazy evaluation requested.
            if goal_reached and lazy:
                break
        if self._log:
            logger.debug(f"# nodes = {n_nodes:> 6d}, iteration = {i:>7d}")
        #
        if anim_name:
            ranim = self._animate(fig, axes, i, file_name=anim_name)
        return ranim

    def _set_legends_online(self, ax, robot):
        """
        Sets up the tree plot legends for different movement types and
        robot states.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to add the legend.
        robot : int
            The index of the robot.
        """
        fontsize = 8
        modes = range(self._specs.n_mode)
        colors = self._colors
        #
        legend_robot = lambda m, c: plt.plot(
            [], [], ls="", marker=m, mfc=c, mec="k", ms=6
        )[0]
        legend_line = lambda c, ls: plt.plot([], [], color=c, ls=ls)[0]
        # Add start and goal legends.
        handles = [legend_robot("o", "yellow")]
        labels = ["Start"]
        handles += [legend_robot("o", "red")]
        labels += ["Goal"]
        # Add modes legends.
        # handles += [legend_line(colors[mode], "-") for mode in modes]
        """ handles += [
            legend_line("deepskyblue", self._styles[mode][1]) for mode in modes
        ]
        labels += [f"M {mode}" for mode in modes] """
        handles += [legend_line(self._line_style_map[0][0], "-")]
        labels += ["Not reached"]
        # Add suboptimal and optimal legends.
        handles += [
            legend_line(self._line_style_map[1][0], "-"),
            legend_line(self._line_style_map[2][0], "-"),
        ]
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
        """
        Draws the robot at a given position on the tree plot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the robot.
        robot : int
            The robot index.
        pose : array-like, shape (2,)
            The (x, y) position of the robot.
        color : str
            The color used for the robot marker.
        """
        ax.plot(
            pose[0],
            pose[1],
            ls="",
            marker="o",
            mfc=color,
            mec="k",
            zorder=20.0,
        )

    def _set_subplot_online(self, ax, robot):
        """
        Configures a subplot for visualizing the tree and workspace.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The subplot axes to configure.
        robot : int
            The robot index.
        """
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
        # ax.grid(zorder=0.0)
        ax.grid(False)
        ax.set_aspect("equal", adjustable="box")
        self._set_legends_online(ax, robot)
        # Set subplot title
        if robot == 0:
            ax.set_title("Robot 0, leader")
        else:
            ax.set_title(f"Robot {robot}")
        # Draw obstacles.
        for i, cnt in enumerate(self._obstacle_contours):
            ax.add_patch(Polygon(cnt, ec="k", lw=1, fc="slategray", zorder=2))
        # Draw start and goal positions.
        self._draw_robot(
            ax, robot, self._start.reshape(-1, 2)[robot], "yellow"
        )
        self._draw_robot(ax, robot, self._goal.reshape(-1, 2)[robot], "red")

    def _set_plot_online(self, fig=None, axes=None, cid=-1):
        """
        Initializes and configures the plotting environment for
        real-time visualization of tree,.

        Parameters
        ----------
        fig : matplotlib.figure.Figure, optional
            The figure object for plotting (default is None).
        axes : array-like of matplotlib.axes.Axes, optional
            The subplot axes (default is None).
        cid : int, optional
            Connection ID for keyboard event handling (default is -1).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The updated figure object.
        axes : array-like of matplotlib.axes.Axes
            The updated axes for subplots.
        cid : int
            The connection ID for event handling.
        """
        fig = None
        if fig is None:
            n_robot = self._specs.n_robot
            n_col = 3
            n_row = np.ceil(n_robot / n_col).astype(int)
            fig, axes = plt.subplots(n_row, n_col, constrained_layout=True)
            fig.set_size_inches(5 * n_col, 4 * n_row)
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
        """
        Draws a line representing a planned path for requested robot.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes on which to plot the line.
        robot : int
            The robot index.
        poses : ndarray, shape (N, 2)
            The sequence of (x, y) positions defining the path.
        ltype : int, optional
            The line type (default is 0,  see _ltypes attribute).

        Returns
        -------
        art : matplotlib.lines.Line2D
            The drawn line matplotlib artist.
        """
        color, zorder = self._line_style_map.get(ltype)
        (art,) = ax.plot(
            poses[:, 0], poses[:, 1], lw=1.0, color=color, zorder=zorder
        )
        return art

    def _edit_line(self, art, robot, poses, ltype=0):
        """
        Updates an existing path drawing for requested robot.

        Parameters
        ----------
        art : matplotlib.lines.Line2D
            The existing line artist to update.
        robot : int
            The robot index.
        poses : ndarray, shape (N, 2)
            The updated sequence of (x, y) path positions.
        ltype : int, optional
            The line type (default is 0).
        """
        color, zorder = self._line_style_map.get(ltype)
        art.set(
            xdata=poses[:, 0],
            ydata=poses[:, 1],
            color=color,
            zorder=zorder,
        )

    def _draw_node(self, axes, ind, tracks, ltype):
        """
        Draws or updates a tree node representing.

        Parameters
        ----------
        axes : array-like of matplotlib.axes.Axes
            The set of axes for drawing the node.
        ind : int
            The index of the node.
        tracks : ndarray
            The carray of path from parent node to current index.
        ltype : int
            The line type for the node's path.

        Notes
        -----
        - If the node exists, it updates the path instead of redrawing.
        """
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
        """
        Draws all nodes for a given iteration of the planning process.

        Parameters
        ----------
        iteration : int
            The current iteration index.
        axes : array-like of matplotlib.axes.Axes
            The set of axes for plotting.
        """
        for ind, tracks, ltype in self._arts_info[iteration]:
            self._draw_node(axes, ind, tracks, ltype)

    def _get_stop_req(self):
        """
        Checks for user input to stop execution.

        Returns
        -------
        stop_req : bool
            True if the user enters 'Y' or 'y', otherwise False.
        """
        stop_req = False
        in_str = input("Enter Y to stop: ").strip()
        if re.match("[Yy]", in_str):
            stop_req = True
        return stop_req

    def _animate(self, fig, axes, iterations, file_name=None):
        # Clear arts.
        self._arts = {}
        # Clear and set up screen.
        for i, ax in enumerate(axes):
            ax.clear()
            self._set_subplot_online(ax, i)
        dpi = 300
        fig.set_dpi(dpi)
        # Set up video configs.
        fps = 60
        interval = 1000 // fps
        fps = 1000 / interval
        # Animating
        frames = list(range(1, self._N)) + [self._N - 1] * (int(fps) * 3)
        anim = animation.FuncAnimation(
            fig,
            self._draw_nodes,
            fargs=(axes,),
            interval=interval,
            frames=frames,
        )
        # Saving animation.
        if file_name:
            file_name = os.path.join(os.getcwd(), f"{file_name}.mp4")
            anim.save(
                file_name,
                fps=fps,
                writer="ffmpeg",
                codec="libx264rgb",
                extra_args=["-crf", "0", "-preset", "slow", "-b:v", "10M"],
            )
        plt.show(block=False)
        plt.pause(0.01)
        return anim


class RRTS(RRT):
    """
    Adapted Assymptotically Optimal Rapidly-exploring Random Tree
    (Adapted RRT*) motion planner for class of heterogeneous robot
    systems. This class extends the Adapted RRT algorithm with
    additional features such as rewiring and cost propagation, improving
    path quality over time.

    Attributes
    ----------
        _ndim (int): Dimensionality of the state space (2 * number of
        robots).
        _k_s (float): Scaling factor for k-nearest neighbor selection.

    Methods
    -------
    _k_nearest_neighbor(ind)
        Finds the k-nearest neighbors of a node at the given index.
    _edit_node(ind, pose, depth, parent, cmds, cmd_modes, track, cost)
        Updates an existing node's properties.
    _sort_inds(inds, values, costs, distances)
        Sorts a list of nodes based on values, costs, and distances.
    _get_all_childs(ind)
        Retrieves all child nodes of a given node.
    _propagate_cost_to_childs(ind, value_diff)
        Propagates value updates to all child nodes.
    _try_rewiring(ind_i, ind_f, rewiring_near_nodes=False)
        Attempts to rewire a node for cost improvement.
    _rewire_new_node(near_inds, new_ind)
        Rewires the new node if improving its path value.
    _rewire_near_nodes(new_ind, near_inds)
        Rewires near nodes if improving its path value.
    plans(start, goal, basic_mode_sequence=None, fig_name=None,
          anim_name=None, anim_online=False, plot=False, log=True, lazy=False)
        Generates a motion plan using Adapted RRT*.
    """

    def __init__(
        self,
        specs,
        collision,
        obstacle_contours,
        max_size=1000,
        max_iter=500000,
        tol_cmd=1e-2,
        goal_bias=0.07,
        tol_goal=1e-0,
    ) -> None:
        """
        Class constructor.

        Parameters
        ----------
        specs : SwarmSpecs object
            Specifications of robots and workspace boundaries.
        collision : Collision object
            Collision checking module.
        obstacle_contours : List
            List of obstacle outline contours.
        max_size : int, optional
            Maximum number of nodes in the tree (default is 1000).
        max_iter : int, optional
            Maximum iterations for the planner (default is 500000).
        tol_cmd : float, optional (default is 1e-2).
             Tolerance for removing zero length pseudo displacements.
        goal_bias : float, optional
            Probability of sampling goal position (default is 0.07).
        tol_goal : float, optional
            Tolerance for considering a node has reached the goal
            (default is 1.0).
        """
        super().__init__(
            specs,
            collision,
            obstacle_contours,
            max_size,
            max_iter,
            tol_cmd,
            goal_bias,
            tol_goal,
        )
        self._ndim = 2 * self._specs.n_robot  # Space dimension.
        self._k_s = 2 * np.exp(1)

    def _k_nearest_neighbor(self, ind):
        """
        Finds the k-nearest neighbors of a given node.

        Parameters
        ----------
        ind : int
            Index of the node in the tree.

        Returns
        -------
        np.ndarray
            Indices of the nearest neighbors.
        """
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
        """
        Updates an existing node in the tree.

        Parameters
        ----------
        ind : int
            Index of the node.
        pose : np.ndarray
            Updated position of the node.
        depth : int
            Updated depth of the node in the tree.
        parent : int
            Updated index of the parent node.
        cmds : list
            Updated pseudo displacement leading to this node.
        cmd_modes : list
            Updated corresponding mode sequence of cmds.
        track : list
            Updated path positions from parent node to this node.
        cost : float
            Updated cost of reaching this node from its parent.
        """
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
        """
        Sorts indices based on given values, costs, and distances in
        ascending order.

        Parameters
        ----------
        inds : np.ndarray
            Array of node indices.
        values : np.ndarray
            Array of node values (e.g., path cost).
        costs : np.ndarray
            Array of immediate costs from each node's parent.
        distances : np.ndarray
            Array of given distances related to nodes.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            Sorted indices, values, and costs, ordered first by value,
            then cost, and finally distance.
        """
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
        """
        Retrieves all child nodes of a given node recursively.

        Parameters
        ----------
        ind : int
            Index of the node of interest.

        Returns
        -------
        list
            List of child node indices.
        """
        childs = []
        for index in self._childs[ind]:
            childs.append(index)
            childs += self._get_all_childs(index)
        return childs

    def _propagate_cost_to_childs(self, ind, value_diff):
        """
        Propagates cost updates to all child nodes.

        Parameters
        ----------
        ind : int
            Index of the node.
        value_diff : float
            Value difference to be propagated.
        """
        # Get list of indexes of all childs down the tree.
        childs = self._get_all_childs(ind)
        # Add value diff to all childs.
        self._values[childs] += value_diff

    def _try_rewiring(self, ind_i, ind_f, rewiring_near_nodes=False):
        """
        Tries to rewire a node at ind_f to the node at ind_i if the path
        from node in ind_i to the node in ind_i is collision free and
        improving value of the node in ind_f.

        Parameters
        ----------
        ind_i : int
            Index of potential parent node.
        ind_f : int
            Index of the node to be rewired if possible.
        rewiring_near_nodes: bool (default is False)
            True if ind_f is index of a node from near neighbhors.
            False if it is index of new node.
        """
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
        if not (value_new < value_past - self._eps):
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
        """
        Attempts to rewire the new node from its neighbhor nodes if it
        improves value of new node. It does this by sorting potential
        rewirings by their value estimates and stops at first successful
        rewire.

        Parameters
        ----------
        near_inds : np.ndarray
            Indices of nearby nodes that could be potential new parents.
        new_ind : int
            Index of the new node
        """
        new_value = self._values[new_ind]
        new_pose = self._poses[new_ind]
        near_poses = self._poses[near_inds]
        dposes = new_pose - near_poses
        rewiring_costs = self._cost(dposes)
        rewired_values = self._values[near_inds] + rewiring_costs
        # Filter based on estimated value.
        candidate_inds = np.nonzero(rewired_values < new_value - self._eps)
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
        """
        Attempts to rewire near nodes from new node to improve their
        path value. It uses rewired value estimates to weed out
        potentially non improving candidates.

        Parameters
        ----------
        new_ind : int
            Index of the new node.
        near_inds : np.ndarray
            Indices of near nodes that might benefit from rewiring.

        Returns
        -------
        List[int]
            Indices of nodes that were successfully rewired.
        """
        # Remove indices with higher rewired_values.
        new_pose = self._poses[new_ind]
        near_values = self._values[near_inds]
        near_poses = self._poses[near_inds]
        dposes = new_pose - near_poses
        rewiring_costs = self._cost(dposes)
        rewired_values = self._values[new_ind] + rewiring_costs
        candidate_inds = np.nonzero(rewired_values < near_values - self._eps)
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
        plot=False,
        log=True,
        lazy=False,
    ):
        """
        Plans a path from the start position to the goal using the
        Adapted RRT* algorithm.

        Parameters
        ----------
        start : array-like
            Start position of robots.
        goal : array-like
            Goal position of robots.
        basic_mode_sequence : optional
            A basic mode sequence for path planning.
        fig_name : str, optional
            Name of the figure to save the final plot, if provided.
        anim_name : str, optional
            Name of the animation file, if provided.
        anim_online : bool, default=False
            If True, updates animation online during planning.
        plot : bool, default=False
            If True, visualizes tree the planning process.
        log : bool, default=True
            If True, logs the planning process iterations.
        lazy : bool, default=False
            If True, terminates the search immediately after reaching
            the goal.

        Returns
        -------
        None
            The function modifies internal state variables and stores
            the planned path.
        """

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
        ranim = None
        if plot or anim_name:
            fig, axes, cid = self._set_plot_online()
        i = 1
        n_nodes = 1
        toggle = True
        while n_nodes < self._max_size and i < self._max_iter:
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
                # Update indices to draw.
                if plot or anim_name:
                    self._set_arts_info(inds_to_draw)
                # Draw Nodes.
                if plot:
                    self._set_arts_info(inds_to_draw)
                    self._draw_nodes(n_nodes, axes)
                n_nodes += 1
            i += 1
            if self._log and (not n_nodes % 1000) and toggle:
                logger.debug(f"# nodes = {n_nodes:> 6d}, iteration = {i:>7d}")
                toggle = False
            #
            if anim_online and plot:
                plt.pause(0.001)
            # Stop if goal reached andlazy evaluation requested.
            if goal_reached and lazy:
                break
        if self._log:
            logger.debug(f"# nodes = {n_nodes:> 6d}, iteration = {i:>7d}")
        #
        if anim_name:
            ranim = self._animate(fig, axes, i, file_name=anim_name)
        return ranim


def test_obstacle():
    """
    Tests obstacle representation and simulation for a group of robots.

    This function sets up a heterogeneous robotic system, defines
    obstacles in the environment, and runs a simulation with specified
    commands to visualize the system movements

    Parameters
    ----------
    None

    Returns
    -------
    None
        This function runs a simulation and visualizes the results.
    """
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
        ax.add_patch(Polygon(cnt, ec="g", fill=False))


def test_collision():
    """
    Tests various collision detection methods of Collision class.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function prints out the results of multiple collision checks.
    """
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


def test_rrt3(tol_cmd=17.0, goal_bias=0.01):
    """
    Tests Adapted RRT* planning algorithm on a 3 robot system.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function logs the runtime of the planning process and
        displays simulation results.
    """
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
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=1200,
    )
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


def test_errt4(tol_cmd=15.0, goal_bias=0.09):
    """
    Tests Adapted RRT* planning algorithm on a 4 robot system.
    It uses specifications of the experimental robots.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function logs the runtime of the planning process and
        displays simulation results.
    """
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
    start_time = time.time()
    rrt.plans(pose_i, pose_f, [0, 1, 2, 3, 4], anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    # Process the command.
    cmds = rrt.cmds
    # cmds = rrt.post_process(rrt.cmds, ang=10)  # Add mode change.
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
    )
    anim = simulation.simanimation(
        anim_length=1100,
        boundary=True,
        last_section=True,
    )


def test_rrt5():
    """
    Tests Adapted RRT* planning algorithm on a 5 robot system.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function logs the runtime of the planning process and
        displays simulation results.
    """
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


def test_rrt10(tol_cmd=0.01, goal_bias=0.04, max_size=100000):
    """
    Tests Adapted RRT* planning algorithm on a 10 robot system.

    Parameters
    ----------
    None

    Returns
    -------
    None
        The function logs the runtime of the planning process and
        displays simulation results.
    """
    np.random.seed(42)  # Keep for consistency, but can be removed.
    # Build specs of robots and obstacles.
    specs = model.SwarmSpecs.robo10()
    # Obstacle contours.
    obstacle_contours = [
        np.array([[-45, 5], [-45, -200], [-55, -200], [-55, 5]], dtype=float),
        np.array([[55, 200], [55, -5], [45, -5], [45, 200]], dtype=float),
    ]
    # Build obstacle mesh.
    obstacles = Obstacles(specs, obstacle_contours)
    obstacle_contours = obstacles.get_cartesian_obstacle_contours()
    mesh, mesh_contours = obstacles.get_obstacle_mesh()
    # Add obstacles to specification for visualizations.
    specs.set_obstacles(obstacle_contours=obstacle_contours)
    # Set up collision detection.
    collision = Collision(mesh, specs, with_coil=False)
    # Start and goal positions
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
    # Set up Adapted RRT* instance.
    rrt = RRTS(
        specs,
        collision,
        obstacle_contours,
        tol_cmd=tol_cmd,
        goal_bias=goal_bias,
        max_size=max_size,
    )
    # Search for path plan.
    start_time = time.time()
    rrt.plans(pose_i, pose_f, np.arange(10), anim_online=False, plot=False)
    end_time = time.time()
    runtime = end_time - start_time
    logger.debug(f"The runtime of the test() function is {runtime} seconds")
    # Convert pseudo displacements to polar coordinate.
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
    # Animate the simulation results.
    anim = simulation.simanimation(
        vel=30,
        anim_length=1100,
        boundary=True,
        last_section=True,
    )


########## test section ################################################
if __name__ == "__main__":
    # Create a file handler and set its level
    file_handler = logging.FileHandler("logfile.log", mode="w")
    file_handler.setLevel(logging.DEBUG)  # Writes logs to file.
    file_handler.setFormatter(log_formatter)
    logger.addHandler(file_handler)
    # test_obstacle()
    # test_collision()
    # test_errt4()
    # test_rrt10()
    plt.show()
