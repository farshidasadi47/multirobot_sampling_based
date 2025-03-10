# %%
########################################################################
# This module is responsible for classes and methods that publish
# and subscribe to the arduino for closed loop planning  and control.
# For camera communication, data pipeline is used.
# Camera should be installed and working for using this class.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import sys
import os
import time
import re
import csv
from itertools import groupby
from pprint import pprint
import traceback as tback

import numpy as np
import cv2
import skvideo.io
from cv_bridge import CvBridge, CvBridgeError

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from geometry_msgs.msg import Point32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Image, CompressedImage
from std_srvs.srv import Empty
from turtlesim.action import RotateAbsolute

try:
    from multirobot_sampling_based import closedloop, model, localization
except ModuleNotFoundError:
    import closedloop, model, localization

# Change this to your projects path.
package_name = r"multirobot_sampling_based"
package_dir1 = r"/ros_ws/src/"
package_dir2 = os.path.join(os.getenv("HOME", ""), "ros_ws", "src")
# Check if the directories exist and select the first one that does
if os.path.isdir(package_dir1):
    package_dir = package_dir1
elif os.path.isdir(package_dir2):
    package_dir = package_dir2
else:
    raise FileNotFoundError("Package directory was not found.")
ros_dir = os.path.join(package_dir, package_name, package_name)


########## Definiitons #################################################
class NodeTemplate(Node):
    """Parent class for all nodes used in the project."""

    def __init__(self, name):
        if rclpy.ok() is not True:
            rclpy.init(args=sys.argv)
        super().__init__(name)
        self.counter = 0
        # QoS profile
        self.qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.pubs_msgs = dict()
        self.subs_msgs = dict()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True:
            rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__, exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True

    def _add_publisher(self, msg_type, topic: str, reliable=False):
        qos_profile = 1 if reliable else self.qos_profile
        self.create_publisher(msg_type, topic, qos_profile)

    def _add_subscriber(self, msg_type, topic: str, callback, reliable=False):
        qos_profile = 1 if reliable else self.qos_profile
        self.create_subscription(msg_type, topic, callback, qos_profile)

    def _construct_pubsub_msgs(self):
        """Constructs a dictionaries containing pubs and subs msgs."""
        # Get all publishers and subscribers.
        self.pubs_dict = {pub.topic_name: pub for pub in self.publishers}
        self.subs_dict = {sub.topic_name: sub for sub in self.subscriptions}
        # Publishers
        for k, v in self.pubs_dict.items():
            self.pubs_msgs[k] = v.msg_type()  # v is a publisher.
        # Subscribers
        for k, v in self.subs_dict.items():
            self.subs_msgs[k] = v.msg_type()  # v is a subscriber.

    def _add_service_server(self, srv_type, srv_name: str, callback):
        """Adds an action server and puts it into a dictionary."""
        self.create_service(srv_type, srv_name, callback)

    def _add_action_server(self, action_type, action_name, callback):
        ActionServer(self, action_type, action_name, callback)


class MultiThreadedExecutorTemplate(rclpy.executors.MultiThreadedExecutor):
    """Parent class for all multithreaded executors in the project."""

    def __init__(self):
        if rclpy.ok() is not True:
            rclpy.init(args=sys.argv)
        super().__init__()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True:
            rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__, exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True


class SingleThreadedExecutorTemplate(rclpy.executors.SingleThreadedExecutor):
    """Parent class for all singlethreaded executors in the project."""

    def __init__(self):
        if rclpy.ok() is not True:
            rclpy.init(args=sys.argv)
        super().__init__()

    def __enter__(self):
        """To be able to use the class in with statement."""
        return self

    def __exit__(self, exc_type, exc, traceback):
        """To be able to use it in with statement."""
        # Shutdown rclpy.
        if rclpy.ok() is True:
            rclpy.shutdown()
        # Catch exceptions
        if exc_type is not KeyboardInterrupt:
            print(type(exc).__name__, exc.args)
        if exc_type is KeyboardInterrupt:
            print("Interrupted by user.")
        return True


class GetVideo(NodeTemplate):
    """Main node for getting camera frame."""

    def __init__(self, rate=50, name="getvideo"):
        super().__init__(name)
        self.camera = localization.Localization()
        _, self.frame, _ = self.camera.get_frame(True, True)
        self.br = CvBridge()
        self.current = 0
        self.past = time.time()
        # Add publishers.
        self._add_publisher(Image, "/camera", reliable=False)
        self._add_publisher(Float32MultiArray, "/state_fb", reliable=False)
        # Add timer
        self.timer = self.create_timer(1 / rate, self._timer_callback)
        self.timer_cam = self.create_timer(1 / 25, self._timer_cam_callback)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()

    def _timer_callback(self):
        """Orders a state_fb publication."""
        ret, self.frame, state_fb = self.camera.get_frame(True, True)
        if ret:
            self.pubs_msgs["/state_fb"].data = state_fb.tolist()
            self.pubs_dict["/state_fb"].publish(self.pubs_msgs["/state_fb"])
            self.current = time.time()
            elapsed = round((self.current - self.past) * 1000)
            self.past = self.current
            #
            msg = (
                f"{self.current%1e3:+08.3f}|{elapsed:03d}|{self.counter:06d}|"
            )
            none = "None"
            for i, k in enumerate(self.camera._hsv_ranges):
                v = state_fb[3 * i : 3 * i + 3]
                if 999 in v:
                    msg += f"'{k:1s}': {none:>21s},"
                else:
                    msg += (
                        f"'{k:1s}': {v[0]:+06.1f},{v[1]:+06.1f},"
                        f"{np.rad2deg(v[-1]):+07.2f}|"
                    )
            print(msg)
        self.counter = (self.counter + 1) % 1000000

    def _timer_cam_callback(self):
        self.pubs_dict["/camera"].publish(self.br.cv2_to_imgmsg(self.frame))


class ShowVideo(NodeTemplate):
    """Main node for showing cmera frames live."""

    def __init__(self, rate=25, name="showvideo"):
        super().__init__(name)

        def nothing(x):
            pass

        self.window = cv2.namedWindow("workspace", cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar("Header", "workspace", 0, 1, nothing)
        cv2.createTrackbar("Path annotation", "workspace", 0, 1, nothing)
        cv2.setTrackbarPos("Header", "workspace", 0)
        cv2.setTrackbarPos("Path annotation", "workspace", 0)
        # Some global variable
        self.br = CvBridge()
        self.dir = None
        self.header = False
        self.annotation = False
        self._p2mm = 0.5475215  # Copy from localization module.
        self._center = None
        self.csv_file = None
        self.csv_writer = None
        self.video_writer = None
        self.recording = False
        self.current = 0
        self.rate = rate
        self.past = time.time()
        model.define_colors(self)
        self._colors = list(self._colors.values())
        # Add subscribers.
        self._add_subscriber(
            Float32MultiArray, "/state_fb", self._state_fb_cb, reliable=False
        )
        self._add_subscriber(Image, "/camera", self._camera_cb, reliable=False)
        self._add_subscriber(
            Float32MultiArray, "/logs", self._logs_cb, reliable=False
        )
        # Add timer.
        self.timer = self.create_timer(1 / rate, self._timer_callback)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()
        self.subs_msgs["/logs"].data = [0.0]

    def _state_fb_cb(self, msg):
        self.subs_msgs["/state_fb"].data = msg.data

    def _camera_cb(self, msg):
        self.subs_msgs["/camera"] = msg

    def _logs_cb(self, msg):
        self.subs_msgs["/logs"].data = msg.data

    def get_subs_values(self):
        """Return the current value of subscribers as an array."""
        try:
            frame = self.br.imgmsg_to_cv2(self.subs_msgs["/camera"])
        except CvBridgeError:
            frame = np.zeros((200, 200, 3), dtype=np.uint8)
        logs = self.subs_msgs["/logs"].data
        h, w = frame.shape[:2]
        self._center = (int(w / 2), int(h / 2))
        return frame, logs

    def _cartesian2pixel(self, point):
        """Does opposite of _pixel2cartesian."""
        pixel = np.zeros(2, dtype=int)
        pixel[0] = int(point[0] / self._p2mm) + self._center[0]
        pixel[1] = self._center[1] - int(point[1] / self._p2mm)
        return pixel

    def _add_logs_to_frame(self, img, logs):
        if logs[0] and self.annotation and logs[3] > -1:
            logs = np.array(logs)
            input_cmd = logs[1:4]
            theta = logs[4]
            alpha = logs[5]
            mode = logs[6]
            x, xi, xg, error, shape = logs[7:].reshape(5, -1, 2)
            # Draw shape or expected path.
            if np.any(shape != 999):
                shape = shape.ravel()
                shape = shape[shape != 999].reshape(-1, 2).astype(int)
                for i1, i2 in shape:
                    p1 = self._cartesian2pixel(xg[i1])
                    p2 = self._cartesian2pixel(xg[i2])
                    cv2.line(img, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                if np.all(xi != 999) and np.all(xg != 999):
                    for idx, (pt1, pt2) in enumerate(zip(xi, xg)):
                        p1 = self._cartesian2pixel(pt1)
                        p2 = self._cartesian2pixel(pt2)
                        cv2.arrowedLine(
                            img,
                            p1,
                            p2,
                            self._colors[idx],
                            1,
                            cv2.LINE_AA,
                            tipLength=0.05,
                        )

    def _add_header(self, img, logs):
        if self.header:
            w = img.shape[1]
            attic = np.ones((40, w, 3), dtype=np.uint8) * 192
            img = np.concatenate((attic, img), axis=0)
            if logs[0] and self.annotation:
                mode = int(logs[3])
                if mode < 0:
                    text = f"Changing to mode {-mode:1d}"
                elif mode < 1:
                    text = "Tumbling, mode 0"
                else:
                    text = f"Pivot walking in mode {mode:1d}"
                if mode < 999:
                    img = cv2.putText(
                        img,
                        text,
                        [10, 28],
                        cv2.FONT_HERSHEY_COMPLEX,
                        1,
                        (0, 255, 0),
                        1,
                        cv2.LINE_AA,
                    )
        return img

    def _update_recording_path(self):
        parent_dir = os.path.join(ros_dir, "results")
        index = 1
        result_dir = os.path.join(parent_dir, f"{index}")
        while os.path.exists(result_dir):
            if len(os.listdir(result_dir)) < 1:
                break
            index += 1
            result_dir = os.path.join(parent_dir, f"{index}")
        return result_dir

    def _set_writers(self, frame_size, frame_m_size, logs):
        # Calculate len of positions.
        width = len(logs[7:]) // 5 - 1
        self.counter = 1
        os.makedirs(self.dir, exist_ok=True)
        csv_path = os.path.join(self.dir, "logs.csv")
        self.csv_file = open(csv_path, "w")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(
            [
                "counter",
                "INPUT_CMD",
                *[""] * 2,
                "theta",
                "alpha",
                "mode",
                "X",
                *[""] * width,
                "XI",
                *[""] * width,
                "XG",
                *[""] * width,
                "error",
                *[""] * width,
                "SHAPE",
            ]
        )
        video_path = os.path.join(self.dir, "logs.mp4")
        self.video_writer = skvideo.io.FFmpegWriter(
            video_path,
            inputdict={"-r": f"{self.rate}"},
            outputdict={
                "-vcodec": "libx264",
                "-crf": "9",
                "-tune": "film",
                "-r": f"{self.rate}",
            },
        )
        video_m_path = os.path.join(self.dir, "logs_m.mp4")
        self.video_m_writer = skvideo.io.FFmpegWriter(
            video_m_path,
            inputdict={"-r": f"{self.rate}"},
            outputdict={
                "-vcodec": "libx264",
                "-crf": "9",
                "-tune": "film",
                "-r": f"{self.rate}",
            },
        )

    def _write_logs_and_video(self, img, img_m, logs):
        self.csv_writer.writerow(logs)
        self.video_writer.writeFrame(img[:, :, ::-1])
        self.video_m_writer.writeFrame(img_m[:, :, ::-1])

    def _record_video_n_data(self, img, img_m, logs):
        if logs[0] > 0.5:
            # Recording requested, set writers if they are not set.
            if not self.recording:
                self.recording = True
                self.dir = self._update_recording_path()
                self._set_writers(img.shape[:2], img_m.shape[:2], logs)
        else:
            # Stop ongoing recording and release resources.
            if self.recording:
                self.recording = False
                self.csv_writer = None
                self.csv_file.close()
                self.csv_file = None
                self.video_writer.close()
                self.video_writer = None
                self.video_m_writer.close()
                self.video_m_writer = None
        #
        if self.recording and (logs[0] < 16777215):
            self._write_logs_and_video(img, img_m, logs)

    def _timer_callback(self):
        img, logs = self.get_subs_values()
        img_m = img.copy()
        self._add_logs_to_frame(img_m, logs)
        img_m = self._add_header(img_m, logs)
        self._record_video_n_data(img, img_m, logs)
        cv2.imshow("workspace", img_m)
        cv2.waitKey(1)
        if logs[0] == 0:
            self.header = cv2.getTrackbarPos("Header", "workspace")
        self.annotation = cv2.getTrackbarPos("Path annotation", "workspace")
        # Timing and stats.
        self.current = time.time()
        elapsed = round((self.current - self.past) * 1000)
        self.past = self.current
        msg = (
            f"vid: {self.current%1e3:+08.3f}|{elapsed:03d}|{self.counter:06d}|"
        )
        msg += ",".join(f"{i:+07.2f}" for i in logs[:1])
        self.counter = (self.counter + 1) % 1000000
        print(msg, self.recording)


class ControlNode(NodeTemplate):
    """Contains all control related service servers."""

    def __init__(self, control: closedloop.Controller, rate=50):
        super().__init__("controlservice")
        self.control = control
        self.cmd_mode = "idle"
        self.rate = self.create_rate(rate)
        # Add publishers and subscribers.
        self._add_publisher(Point32, "/arduino_field_cmd")
        self._add_publisher(Float32MultiArray, "/logs")
        self._add_subscriber(
            Point32, "/arduino_field_fb", self._arduino_field_fb_cb
        )
        self._add_subscriber(Float32MultiArray, "/state_fb", self._state_fb_cb)
        # Custruct publisher and subscription instance variables.
        self._construct_pubsub_msgs()
        # Service and action servers
        self._add_service_server(Empty, "/set_idle", self._set_idle_server_cb)
        # Open loops with information from camera.
        self._add_action_server(
            RotateAbsolute, "/pivot_walking", self._pivot_walking_server_cb
        )
        self._add_action_server(
            RotateAbsolute, "/mode_change", self._mode_change_server_fb
        )
        self._add_action_server(
            RotateAbsolute, "/tumbling", self._tumbling_server_fb
        )
        # Closed loops
        self._add_action_server(
            RotateAbsolute, "/cart_pivot", self._cart_pivot_server_cb
        )
        self._add_action_server(
            RotateAbsolute,
            "/calibration_pivot",
            self._calibration_pivot_server_cb,
        )
        self._add_action_server(
            RotateAbsolute,
            "/calibration_tumble",
            self._calibration_tumble_server_cb,
        )
        self._add_action_server(
            RotateAbsolute, "/closed_lines", self._closed_lines_server_cb
        )
        self._add_action_server(
            RotateAbsolute, "/closed_plan", self._closed_plan_server_cb
        )

    def _arduino_field_fb_cb(self, msg):
        """Call back for /arduino_field_fb."""
        # Reads arduino field feedback.
        self.subs_msgs["/arduino_field_fb"].x = msg.x
        self.subs_msgs["/arduino_field_fb"].y = msg.y
        self.subs_msgs["/arduino_field_fb"].z = msg.z

    def _state_fb_cb(self, msg):
        self.subs_msgs["/state_fb"].data = msg.data

    def publish_field(self, field):
        """Publishes all given messages."""
        # Update all values to be published.
        self.pubs_msgs["/arduino_field_cmd"].x = field[0]
        self.pubs_msgs["/arduino_field_cmd"].y = field[1]
        self.pubs_msgs["/arduino_field_cmd"].z = field[2]
        # Publish topics.
        self.pubs_dict["/arduino_field_cmd"].publish(
            self.pubs_msgs["/arduino_field_cmd"]
        )

    def publish_logs(
        self,
        *,
        record=None,
        polar_cmd=None,
        state=None,
        initial=None,
        goal=None,
        error=None,
        shape=None,
    ):
        n_robot = self.control.specs.n_robot
        record = record if record is not None else 0
        polar_cmd = polar_cmd if polar_cmd is not None else [999] * 3
        state = (
            state
            if state is not None
            else ([999] * 2 * n_robot, 999, 999, 999)
        )
        initial = initial if initial is not None else [999] * 2 * n_robot
        goal = goal if goal is not None else [999] * 2 * n_robot
        error = error if error is not None else [999] * 2 * n_robot
        shape = shape if shape is not None else [999] * 2 * n_robot
        msg = [
            record,  # Record if > 0
            *polar_cmd,  # r, phi , cmd mode
            state[1],  # theta
            state[2],  # alpha
            state[3],  # mode
            *state[0],  # {x_i, y_i, ...}
            *initial,  # {xi_i, yi_i, ...}
            *goal,  # {xg_i, yg_i, ...}
            *error,  # {ex_i, ey_i, ...}
            *shape,  # {idx_i, idy_i, ...}
        ]
        msg = list(map(float, msg))
        self.pubs_msgs["/logs"].data = msg
        self.pubs_dict["/logs"].publish(self.pubs_msgs["/logs"])

    def get_subs_values(self):
        """Return the current value of subscribers as an array."""
        field_fb = [
            self.subs_msgs["/arduino_field_fb"].x,
            self.subs_msgs["/arduino_field_fb"].y,
            self.subs_msgs["/arduino_field_fb"].z,
        ]
        feedback = self.subs_msgs["/state_fb"].data
        return field_fb, feedback

    def print_stats(self, field, field_fb, state, state_fb, cnt):
        msg = (
            f"{time.time()%1e3:+8.3f}|{cnt:05d}|"
            + "".join(f"{elem:+06.1f}" for elem in field[:2])
            + "|"
            + "".join(f"{elem:+06.1f}" for elem in field_fb[:2])
            + "|"
            + "".join(f"{elem:+06.1f}" for elem in state_fb[0])
            + "|"
            + f"{np.rad2deg(state_fb[1]):+06.1f}"
            + "|"
            + "".join(f"{elem:+6.1f}" for elem in state[0])
            + "|"
            + f"{np.rad2deg(state[1]):+06.1f},{np.rad2deg(state[2]):+06.1f}| "
            + f"{state[3]:01d}"
        )
        print(msg)

    # Service and action servar callbacks
    def _set_idle_server_cb(self, request, response):
        """
        Sets field command in idle condition.
        """
        print("*" * 72)
        regex = r"([+-]?\d+\.?\d*(, *| +)){2}([+-]?\d+\.?\d* *)"
        field = [0.0, 0.0, 0.0]
        self.cmd_mode = "idle"
        self.rate.sleep()
        while True:
            try:
                print("Enter body angles deg and power: theta, alpha, %power.")
                print('Enter "q" for quitting.')
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print('Exitted "set_idle".')
                    break
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    raise ValueError
                # Parse user input.
                body = list(map(float, re.findall(r"[+-]?\d+\.?\d*", in_str)))
                field = self.control.body2magnet(np.deg2rad(body[:2])) + [
                    body[2]
                ]
                msg1 = (
                    f"body[theta, alpha, %power] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in body)
                    + "], "
                )
                msg2 = (
                    f"field[theta, alpha, %power] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in field)
                    + "]"
                )
                print(msg1 + msg2)
                self.control.reset_state(
                    theta=np.deg2rad(body[0]), alpha=np.deg2rad(body[1])
                )
            except:
                print("Ooops! values ignored. Enter values like the template.")
            self.publish_field(field)
        self.publish_field([0.0] * 3)
        print("*" * 72)
        return response

    def _pivot_walking_server_cb(self, goal_handle):
        """
        This service calls pivot_walking function.
        """
        # Ignore the call if we are in the middle of another process.
        cnt = 0
        field = [0.0] * 3
        print("*" * 72)
        regex = r"([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)"
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get the input.
                print("Enter params: num_steps, starting_theta (in degrees)")
                print('Enter "q" for quitting.')
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print('Exitted "service server".')
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                i_cmd = list(map(float, re.findall(r"[+-]?\d+\.?\d*", in_str)))
                msg = (
                    f"[n_steps, starting_theta] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in i_cmd)
                    + "]"
                )
                print(msg)
                # Get current position
                _, feedback = self.get_subs_values()
                self.publish_field(field)
                xi, theta_i = self.control.process_robots(feedback)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.rate.sleep()
                # Change command mode and execute.
                i_cmd[1] = np.deg2rad(i_cmd[1])
                iterator = self.control.pivot_walking_walk(
                    i_cmd, line_up=False
                )
                self.rate.sleep()
                # Execute main controllr.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                print(msg_i)
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.cart2pol(xf[:2] - xi[:2])
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0] * 3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*" * 72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _mode_change_server_fb(self, goal_handle):
        """
        This service calls mode_changing function and performs one
        mode change.
        """
        print("*" * 72)
        cnt = 0
        field = [0.0] * 3
        regex = r"([+-]?\d+(, *| +)){1}([+-]?\d+\.?\d* *)"
        self.rate.sleep()
        # Ignore the call if we are in the middle of another process.
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                xi, theta_i = state_fb
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.control.reset_state(pos=xi, theta=theta_i)
                self.rate.sleep()
                # Get the input.
                print("Enter params: phi (degrees), mode.")
                print('Enter "q" for quitting.')
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    raise ValueError('Exitted "service server".')
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    raise ValueError("Invalid input. Exitted service request.")
                # Parse user input and reset mode, theta.
                params = list(
                    map(float, re.findall(r"[+-]?\d+\.?\d*", in_str))
                )
                str_msg = (
                    f"[phi, mode] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in params)
                    + "]"
                )
                print(str_msg)
                self.publish_field(field)
                self.rate.sleep()
                # Change command mode and execute.
                iterator = self.control.mode_changing(
                    [0.0, np.deg2rad(params[0]), params[1]]
                )
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    # state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                print(msg_i)
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.cart2pol(xf[:2] - xi[:2])
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0] * 3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*" * 72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _tumbling_server_fb(self, goal_handle):
        """
        This service calls tumbling function and performs tumbling.
        """
        print("*" * 72)
        cnt = 0
        field = [0.0] * 3
        regex = r"([+-]?\d+(, *| +)){1}([+-]?\d+\.?\d* *)"
        self.rate.sleep()
        # Check if we are not in the middle of another service that
        # calls data pipeline, If we are, ignore this service call.
        if self.cmd_mode == "idle":
            self.cmd_mode = "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                xi, theta_i = state_fb
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                self.control.reset_state(pos=xi, theta=theta_i)
                self.rate.sleep()
                # Get the input.
                print("Enter params: n_tumble, phi.")
                print('Enter "q" for quitting.')
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    raise ValueError('Exitted "service server".')
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    raise ValueError("Invalid input. Exitted service request.")
                # Parse user input and reset mode, theta.
                params = list(
                    map(float, re.findall(r"[+-]?\d+\.?\d*", in_str))
                )
                str_msg = (
                    f"[n_tumble, phi] = ["
                    + ",".join(f"{elem:+07.2f}" for elem in params)
                    + "]"
                )
                print(str_msg)
                params[0] = params[0] * self.control.specs.tumbling_length
                params[1] = np.deg2rad(params[1])
                self.publish_field(field)
                self.rate.sleep()
                # Get controller.
                iterator = self.control.tumbling(params)
                self.rate.sleep()
                # Execute main controller and update pipeline.
                for field in iterator:
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # Get the first robot found.
                    field_fb, feedback = self.get_subs_values()
                    if abs(state[2]) < 0.01:
                        state_fb = self.control.process_robots(feedback)
                    self.publish_field(field)
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                print(msg_i)
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                polar_dist = self.control.cart2pol(xf[:2] - xi[:2])
                polar_dist[1] = np.rad2deg(polar_dist[1])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                # Set commands to zero.
                self.publish_field([0.0] * 3)
                self.rate.sleep()
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        print("*" * 72)
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _cart_pivot_server_cb(self, goal_handle):
        """Executes closed loop pivot alking for one robot."""
        cnt = 0
        print("*" * 72)
        regex = r"([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)"
        field = [0.0, 0.0, 0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                print("Enter final position: [xf, yf].")
                print('Enter "q" for quitting.')
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print('Exitted "set_idle".')
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                xf = list(map(float, re.findall(r"[+-]?\d+\.?\d*", in_str)))
                xf *= self.control.specs.n_robot
                # Get current position
                _, feedback = self.get_subs_values()
                xi, theta_i = self.control.process_robots(feedback)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                msg_f = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                print(msg_f)
                # Reset state
                self.control.reset_state(pos=xi, theta=theta_i)
                self.publish_field(field)
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.closed_pivot_walking_cartesian(xf)
                self.rate.sleep()
                for field in iterator:
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    if field is None:
                        field = iterator.send(state_fb)
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg_i)
                print(msg_f)
                print(msg)
                polar_dist = self.control.cart2pol(xf[:2] - xi[:2])
                print(f"r:{polar_dist[0]:+07.2f},phi: {polar_dist[1]:+07.2f}")
                self.publish_field([0.0] * 3)
                self.rate.sleep()
            except StopIteration:
                pass
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _calibration_pivot_server_cb(self, goal_handle):
        cnt = 0
        print("*" * 72)
        regex = r"([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)"
        field = [0.0, 0.0, 0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                print("Enter final position: [r, n_sections].")
                print('Enter "q" for quitting.')
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print('Exitted "set_idle".')
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                params = list(
                    map(float, re.findall(r"[+-]?\d+\.?\d*", in_str))
                )
                # Get current position
                _, feedback = self.get_subs_values()
                xi, theta_i = self.control.process_robots(feedback)
                print("=" * 79)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                print(f"r: {params[0]:+07.2f}, n_sections:{params[1]:+07.2f}")
                # Reset state
                self.control.reset_state(pos=xi, theta=theta_i)
                self.publish_field(field)
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.pivot_calibration(*params)
                print("=" * 79)
                self.rate.sleep()
                # for field in iterator:
                while True:
                    field = next(iterator)
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    state_fb = self.control.process_robots(feedback)
                    if field is None:
                        field = iterator.send(state_fb)
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
            except StopIteration as e:
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                self.publish_field([0.0] * 3)
                self.rate.sleep()
                print("=" * 72)
                print(e.value)
                pass
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _calibration_tumble_server_cb(self, goal_handle):
        cnt = 0
        print("*" * 72)
        regex = r"([+-]?\d+\.?\d*(, *| +)){1}([+-]?\d+\.?\d* *)"
        field = [0.0, 0.0, 0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                print("Enter final position: [r, n_sections].")
                print('Enter "q" for quitting.')
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print('Exitted "set_idle".')
                    raise ValueError
                # Check if user input matches the template.
                if re.fullmatch(regex, in_str) is None:
                    print("Invalid input. Exitted service request.")
                    raise ValueError
                # Parse user input.
                params = list(
                    map(float, re.findall(r"[+-]?\d+\.?\d*", in_str))
                )
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                xi, theta_i = state_fb
                print("=" * 79)
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                print(f"r: {params[0]:+07.2f}, n_sections:{params[1]:+07.2f}")
                # Reset state
                self.control.reset_state(pos=xi, theta=theta_i)
                self.publish_field(field)
                state = self.control.get_state()[:4]
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.tumble_calibration(*params)
                print("=" * 79)
                self.rate.sleep()
                # for field in iterator:
                while True:
                    field = next(iterator)
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    if abs(state[2]) < 0.01:
                        state_fb = self.control.process_robots(feedback)
                    if field is None:
                        field = iterator.send(state_fb)
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    self.print_stats(field, field_fb, state, state_fb, cnt)
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
            except StopIteration as e:
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback)
                self.print_stats(field, field_fb, state, state_fb, cnt)
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback)
                msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf) + "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(msg)
                self.publish_field([0.0] * 3)
                self.rate.sleep()
                print("=" * 72)
                print(e.value)
                pass
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _closed_lines_server_cb(self, goal_handle):
        cnt = 1
        print("*" * 72)
        polar_cmds = np.array(
            [
                [70, np.pi / 2, 1],
                [70, -3 * np.pi / 4, 1],
                [10, -np.pi / 4, -2],
                [50, -np.pi / 2, 2],
                [50, np.pi / 4, 2],
                [50, np.pi / 4, 0],
                [10, np.pi / 4, -1],
            ]
        )
        polar_cmds = np.array(
            [
                [4 * self.control.specs.tumbling_length, np.pi, 0],
                [20, np.pi / 2, 1],
            ]
        )
        shape = [0, 1, 1, 2, 999, 999, 999, 999, 999, 999]
        shape = shape[: 2 * self.control.specs.n_robot]
        field = [0.0, 0.0, 0.0]
        if polar_cmds[polar_cmds[:, 2] > 0, 2].size:
            mode_s = polar_cmds[polar_cmds[:, 2] > 0, 2][0]
        else:
            mode_s = 1
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(
                    feedback, any_robot=False
                )
                xi, theta_i = state_fb
                msg_i = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg_i += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg_i)
                user_input = input("Enter phi offset in degrees:\n")
                if re.match("[+-]?\d+(.\d+)?", user_input):
                    polar_cmds[:, 1] += np.deg2rad(float(user_input))
                user_input = input('Enter "y" if you want to save data.\n')
                save = 1 if re.match("y|Y", user_input) else -1
                # Calculate ultimate goal
                XG = self.control.get_cartesian_goal(xi, mode_s, polar_cmds)[
                    -1
                ]
                # Reset state
                self.control.reset_state(pos=xi, theta=theta_i, mode=mode_s)
                self.publish_field(field)
                state = self.control.get_state()[:4]
                self.rate.sleep()
                # Execute closed loop control.
                iterator = self.control.closed_lines(polar_cmds, average=False)
                print("=" * 79)
                self.rate.sleep()
                # for field in iterator:
                while True:
                    from_control = next(iterator)
                    # Get current position
                    field_fb, feedback = self.get_subs_values()
                    if abs(state[2]) < 0.1:
                        state_fb = self.control.process_robots(feedback, False)
                    if from_control is None:
                        from_control = iterator.send(state_fb)
                    field, input_cmd, xi, xg, _ = from_control
                    field.append(self.control.power)
                    state = self.control.get_state()[:4]
                    # self.print_stats(field, field_fb, state, state_fb,cnt)
                    self.publish_logs(
                        record=cnt * save,
                        polar_cmd=input_cmd,
                        state=state,
                        initial=xi,
                        goal=xg,
                        error=xg - state[0],
                    )
                    self.publish_field(field)
                    cnt += 1
                    self.rate.sleep()
            except StopIteration as e:
                # Get current position
                field_fb, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(feedback, False)
                self.publish_logs(
                    record=cnt * save,
                    state=state,
                    initial=xi,
                    goal=XG,
                    error=XG - state[0],
                    shape=shape,
                )
                self.rate.sleep()
                _, feedback = self.get_subs_values()
                xf, theta_f = self.control.process_robots(feedback, False)
                self.publish_field(field)
                msg = f"xg: [" + ",".join(f"{i:+07.2f}" for i in xg)
                msg += "]\n"
                msg += f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf)
                msg += "]"
                msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}"
                print(e.value[0])
                print(msg)
                print("=" * 72)
                time.sleep(10.0)
                pass
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_logs(record=0)
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        self.rate.sleep()
        return result

    def _closed_plan_server_cb(self, goal_handle):
        cnt = 1
        os.system("clear")
        print("*" * 72)
        n_robot = self.control.specs.n_robot
        x_dim = n_robot * 2 - 1
        regex_num = r"[+-]?\d+\.?\d*((, *| +)[+-]?\d+\.?\d*){%d} *" % x_dim
        regex_ltr = r"\w((, *| +)\w)* *"
        regex_ltr = r"([\w\d)((, *| +)(\w\d))* *"  # A0, 11, ...
        chars = "".join(char for char in self.control.specs.chars)
        regex_ltr = r"[%s]\d((, *| +)[%s]\d)* *" % (chars, chars)
        field = [0.0, 0.0, 0.0]
        self.rate.sleep()
        if self.cmd_mode == "idle":
            self.cmd_mode == "busy"
            try:
                # Get current position
                _, feedback = self.get_subs_values()
                state_fb = self.control.process_robots(
                    feedback, any_robot=False
                )
                xi, theta_i = state_fb
                msg = f"xi: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                msg += f", theta_i: {np.rad2deg(theta_i):+07.2f}"
                print(msg)
                # Get initial locations.
                print(f"Enter goal for {n_robot} robots as x_i, y_i, ... OR")
                print("Anything else, but q, prompts to use camra feedback OR")
                print('Enter "q" for quitting.')
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print("Exitted action call.")
                    raise ValueError
                if re.fullmatch(regex_num, in_str):
                    xi = list(map(float, re.split(r", *| +", in_str)))
                    xi = np.array(xi, dtype=float)
                    msg = f"XI: [" + ",".join(f"{i:+07.2f}" for i in xi) + "]"
                    print(msg)
                # Get final positions.
                print("Enter final positions by coordinate or letter:")
                print(f"Enter goal for {n_robot} robots as x_i, y_i, ... OR")
                print(f"Enter list of prespecified letters with steps from")
                msg = ",".join(
                    k + str(v["steps"])
                    for k, v in self.control.specs.chars.items()
                )
                print(msg + " OR")
                print("Example: A2, B1")
                print('Enter "q" for quitting.')
                # Read user input.
                in_str = input("Enter values: ").strip()
                # Check if user requests quitting.
                if re.match("q", in_str) is not None:
                    print("Exitted action call.")
                    raise ValueError
                elif re.fullmatch(regex_num, in_str):
                    # Process xg.
                    in_xg = list(map(float, re.split(r", *| +", in_str)))
                    in_xg = np.array(in_xg, dtype=float)
                    if len(in_xg) != 2 * self.control.specs.n_robot:
                        raise ValueError("Wrong input.")
                    # Get shape.
                    in_str = input("Enter shape indices: ").strip()
                    regex_cr = r"((\d(, *| +)){2})*(\d(, *| +)\d)"
                    if re.fullmatch(regex_cr, in_str) is not None:
                        shape = list(map(int, re.split(r", *| +", in_str)))
                        shape = np.array(shape, dtype=int)
                        if np.any(shape >= self.control.specs.n_robot):
                            raise ValueError("Wrong input.")
                        shape = shape[: 2 * self.control.specs.n_robot]
                        shape_append = (
                            self.control.specs.n_robot - len(shape) // 2
                        )
                        shape = np.append(shape, [999] * shape_append * 2)
                    else:
                        shape = None
                    goals = [(in_xg, shape)]
                elif re.fullmatch(regex_ltr, in_str):
                    # Get goal and shapes from predefined values.
                    in_chars = [
                        c for c, _ in groupby(re.split(r", *| +", in_str))
                    ]
                    goals_raw = [
                        self.control.specs.get_letter(c[0]) for c in in_chars
                    ]
                    goals = [(g[0], g[1]) for g in goals_raw]
                else:
                    print('Invalid input. Ignored "action request".')
                    raise ValueError
                print("Goals and shapes:")
                pprint(goals)
                print("-" * 79)
                # Get the scenario.
                print("Scenarios:")
                pprint(self.control.specs.scenarios)
                user_input = input("Enter scenario number: \n").strip()
                if re.fullmatch(r"\d+", user_input):
                    scenario = int(user_input)
                    self.control.specs.set_obstacle_scenario(scenario)
                else:
                    scenario = 0
                print(f"Selected scenario: {scenario}")
                print("-" * 79)
                # Get params.
                params = {}
                # Get tol_cmd.
                user_input = input("Enter tol cmd.\n")
                if re.fullmatch(r"[+]?\d+(\.\d*)?|\d*\.\d+", user_input):
                    params["tol_cmd"] = float(user_input)
                # Get goal_bias.
                user_input = input("Enter goal bias.\n")
                if re.fullmatch(r"[+]?\d+(\.\d*)?|\d*\.\d+", user_input):
                    params["goal_bias"] = float(user_input)
                # Get max_size.
                user_input = input("Enter max size of tree.\n")
                if re.fullmatch(r"[+]?\d+(\.\d*)?", user_input):
                    params["max_size"] = int(float(user_input))
                # Get max_iterations.
                user_input = input("Enter max iteration of rrt.\n")
                if re.fullmatch(r"[+]?\d+(\.\d*)?", user_input):
                    params["max_iter"] = int(float(user_input))
                # Get initial feasibility tolerance.
                user_input = input("Enter initial feasibility tolerance.\n")
                if re.fullmatch(r"[+]?\d+(\.\d*)?|\d*\.\d+", user_input):
                    params["tol_goal"] = float(user_input)
                # Get ang_mode_change.
                user_input = input("Enter angle of mode change: \n")
                if re.fullmatch(r"[+-]?\d+(\.\d*)?", user_input):
                    params["ang_mode_change"] = float(user_input)
                print("Params:")
                pprint(params)
                print("-" * 79)
                # Get other user inputs.
                user_input = input('Enter "y" if you want to save data.\n')
                save = 1 if re.match("y|Y", user_input) else -1
                # Reset state
                self.control.reset_state(pos=xi, theta=theta_i)
                self.publish_field(field)
                state = self.control.get_state()[:4]
                self.rate.sleep()
                # Execute closed loop control.
                n_goal = len(goals)
                for idx, goal in enumerate(goals):
                    print("*" * 72)
                    print(f"Running plan {idx+1:02d} of {n_goal:02d}.")
                    XG, shape = goal
                    print(shape)
                    print(f"XG: [" + ",".join(f"{i:+07.2f}" for i in XG) + "]")
                    # Build planner iterator
                    print("=" * 72)
                    iterator = self.control.plan(xi, XG, **params)
                    self.rate.sleep()
                    try:
                        while True:
                            from_control = next(iterator)
                            # Get current position
                            field_fb, feedback = self.get_subs_values()
                            if abs(state[2]) < 5.0:
                                state_fb = self.control.process_robots(
                                    feedback, False
                                )
                            # Send feedback if requested.
                            if from_control is None:
                                from_control = iterator.send(state_fb)
                            field, input_cmd, xi, xg, pause_rec = from_control
                            # Check if pause_record requested.
                            if pause_rec:
                                rec_cnt = 16777215
                            else:
                                rec_cnt = cnt
                                cnt += 1
                            field.append(self.control.power)
                            state = self.control.get_state()[:4]
                            error = xg - state[0]
                            self.publish_logs(
                                record=rec_cnt * save,
                                polar_cmd=input_cmd,
                                state=state,
                                initial=xi,
                                goal=xg,
                                error=error,
                            )
                            self.publish_field(field)
                            self.rate.sleep()
                    except StopIteration as e:
                        p_msg, finished = e.value
                        # Get current position
                        field_fb, feedback = self.get_subs_values()
                        state_fb = self.control.process_robots(feedback, False)
                        ERROR = XG - state[0]
                        self.publish_logs(
                            record=cnt * save,
                            state=state,
                            initial=xi,
                            goal=XG,
                            error=ERROR,
                            shape=shape if finished else None,
                        )
                        self.rate.sleep()
                        _, feedback = self.get_subs_values()
                        xf, theta_f = self.control.process_robots(
                            feedback, False
                        )
                        self.publish_field(field)
                        msg = f"xf: [" + ",".join(f"{i:+07.2f}" for i in xf)
                        msg += "]\n"
                        msg += f"xg: [" + ",".join(f"{i:+07.2f}" for i in xg)
                        msg += f", theta_f: {np.rad2deg(theta_f):+07.2f}\n"
                        msg += f"XG: [" + ",".join(f"{i:+07.2f}" for i in XG)
                        msg += "]\n"
                        msg += f"e: [" + ",".join(f"{i:+07.2f}" for i in error)
                        msg += "]\n"
                        msg += f"E: [" + ",".join(f"{i:+07.2f}" for i in ERROR)
                        msg += "]\n"
                        print(p_msg)
                        if finished:
                            print(msg)
                            time.sleep(10.0)
                        pass
                    except RuntimeError as exc:
                        print(type(exc).__name__, exc.args)
                        pass
                # Set everything to neutral.
                self.publish_logs(record=0)
                self.publish_field([0.0] * 3)
                self.rate.sleep()
            except KeyboardInterrupt:
                pass
            except Exception as exc:
                print(type(exc).__name__, exc.args)
                tback.print_exc()
                self.publish_logs(record=0)
                self.publish_field([0.0] * 3)
                self.rate.sleep()
                pass
        else:
            print("Not in idle mode. Current server call is ignored.")
        # Neutral magnetic field.
        goal_handle.succeed()
        result = RotateAbsolute.Result()
        self.publish_logs(record=0)
        self.control.specs.set_obstacle_scenario(0)
        self.publish_field([0.0] * 3)
        self.cmd_mode = "idle"
        print("*" * 72)
        self.rate.sleep()
        return result


class MainExecutor(MultiThreadedExecutorTemplate):
    """Main executor for arduino comunications."""

    def __init__(self, rate=50, n_robot=3):
        super().__init__()
        specs = model.SwarmSpecs.robo(n_robot)
        control = closedloop.Controller(specs)
        # Add nodes.
        self.add_node(ControlNode(control, rate))
        print("*" * 72 + "\nMain executor is initialized.\n" + "*" * 72)


class GetVideoExecutor(MultiThreadedExecutorTemplate):
    """Main executor for getting camera frames and feedback."""

    def __init__(self, rate=50):
        super().__init__()
        self.add_node(GetVideo(rate))
        print("*" * 72 + "\nGetVideo node is initialized.\n" + "*" * 72)


class ShowVideoExecutor(SingleThreadedExecutorTemplate):
    def __init__(self, rate=25):
        super().__init__()
        self.add_node(ShowVideo(rate))
        print("*" * 72 + "\nShowVideo  node is initialized.\n" + "*" * 72)


def main(n_robot=3):
    print("*" * 72)
    regex = r"[3-5]"
    while True:
        # Read user input.
        in_str = input("Enter number of robots from {3,4,5}: ").strip()
        # Check if user input matches the template.
        if re.fullmatch(regex, in_str) is None:
            print("Invalid value enterred!!! Value Ignored.")
            continue
        # Parse user input.
        n_robot = int(in_str)
        print(f"Swarm planner is called for {n_robot:1d} robots.")
        break
    with MainExecutor(50, n_robot) as executor:
        executor.spin()


def get_video():
    with GetVideoExecutor(51) as executor:
        executor.spin()


def show_video():
    with ShowVideoExecutor(20) as video:
        video.spin()


########## Test section ################################################
if __name__ == "__main__":
    # get_video()
    pass
