# %%
########################################################################
# This files hold classes and functions that takes video frames of
# experimental tests and converts them to progression images for paper.
# Author: Farshid Asadi, farshidasadi47@yahoo.com
########## Libraries ###################################################
import os
import sys
import string

import cv2

import numpy as np
import pandas as pd

np.set_printoptions(precision=4, suppress=True)
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors as mc
from matplotlib import rc
from matplotlib.colors import to_rgba
from matplotlib.legend_handler import HandlerTuple

plt.rcParams["figure.figsize"] = [7.2, 8.0]
plt.rcParams.update({"font.size": 11})
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.family"] = "serif"
# plt.rcParams["font.serif"] = "Computer Modern Roman"
# plt.rcParams["text.usetex"] = True
mpl.rcParams["hatch.linewidth"] = 0.5


try:
    from multirobot_sampling_based.model import define_colors
except ModuleNotFoundError:
    # Add parent directory and import modules.
    sys.path.append(os.path.abspath(".."))
    from model import define_colors


########## Functions ###################################################
class ProcessVideo:
    """
    This class gets video of swarm movement and converts it to snapshots
    of each step.
    """

    def __init__(
        self,
        path_list,
        length=30,
        stride=0,
        skip_mode_change=False,
        combine_similar_sections=True,
    ):
        """
        ----------
        Parameters
        ----------
        path_list: List denoting file paths
               path_list= [video_path, csv_path]
        """
        # From camera calibration, localization module.
        self._p2mm = 0.5711465
        self._mm2p = 1 / self._p2mm
        self._fps = 20
        define_colors(self)
        self._colors = list(self._colors.keys())
        # Replace colors if you want.
        self._colors[self._colors.index("b")] = "deepskyblue"
        self.vid_path = path_list[0]
        self.csv_path = path_list[1]
        self.csv_data, self.sections = self._process_csv(
            self.csv_path, skip_mode_change
        )
        # Combine consequtive sections with same mode.
        if combine_similar_sections:
            self.sections = self._combine_consequtives(self.sections)
        # Densify sections.
        if stride > 0:
            self.dense_sections = self._populate_sections_stride(
                self.csv_data, self.sections, stride
            )
        elif stride == 0:
            self.dense_sections = self._populate_sections(
                self.csv_data, self.sections, length
            )
        else:
            self._dense_sections = self.sections
        # Process video
        self.frames = self._process_video(self.dense_sections, self.vid_path)
        self.blends = self._blend_frames(self.frames, self.dense_sections)

    def light(self, c, factor=0.0):
        """Brightens colors."""
        if (c == "k") and (factor != 0):
            # Since the transformation does not work on black.
            rgb = np.array([[[140, 140, 140]]], dtype=np.uint8)
        else:
            rgb = (np.array([[to_rgba(c)[:3]]]) * 255).astype(np.uint8)
        hls = cv2.cvtColor(rgb, cv2.COLOR_RGB2HLS_FULL)
        hls[0, 0, 1] = max(0, min(255, (1 + factor) * hls[0, 0, 1]))
        rgb = (
            np.squeeze(cv2.cvtColor(hls, cv2.COLOR_HLS2RGB_FULL)).astype(float)
            / 255
        )
        return rgb.tolist()

    def _itemize(self, data_row):
        """
        Gets csv_data and returns a tuple of:
        input_cmd (3), mode, X, XI, XG, E, shape
        """
        # Headers
        """
        0  : counter,
        1-3: input_cmd,
        4  : theta,
        5  : alpha,
        6  : mode,
        7- : X, XI, XG, ERROR, SHAPE
        """
        input_cmd = data_row[1:4]
        mode = data_row[6]
        X_SHAPE = np.reshape(data_row[7:], (5, -1))
        X = X_SHAPE[0]
        XI = X_SHAPE[1]
        XG = X_SHAPE[2]
        E = X_SHAPE[3]
        SHAPE = X_SHAPE[4]
        return (input_cmd, mode, X, XI, XG, E, SHAPE)

    def _process_csv(self, csv_path, skip_mode_change=True):
        csv_data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
        csv_data[:, 0] = np.arange(len(csv_data))  # Reindex data.
        # Find indexes where command changed.
        diff_data = np.zeros((csv_data.shape[0], 3), dtype=float)
        diff_data[0, 0] = 1.0
        diff_data[1:, :] = np.diff(csv_data[:, [1, 2, 3]], axis=0)
        change_idx = np.where(np.any(diff_data, axis=1))[0]
        change_idx = np.append(change_idx, len(csv_data) - 1)
        # Collect start and end of each section.
        sections = []
        for i in range(len(change_idx) - 1):
            # Iterate in change_idx in pair.
            i_s = change_idx[i]
            i_e = change_idx[i + 1] - 1
            mode_section = csv_data[i_s, 3]
            # Exclude mode changes if requested.
            if (mode_section < 0) and skip_mode_change:
                continue
            section = []
            for ind in (i_s, i_e):
                data = csv_data[ind]
                data[0] = ind
                section.append(data)
            sections.append(np.array(section, dtype=float))
        return csv_data, sections

    def _combine_consequtives(self, sections):
        """
        Combine consequtive displacements of the same mode.
        """
        sections_n = []
        section = sections[0][:-1].tolist()
        for i in range(1, len(sections)):
            mode_p = sections[i - 1][0][3]
            mode = sections[i][0][3]
            if mode_p == mode:
                # Consequtive sections with same mode.
                section.extend(sections[i][:-1])
            else:
                # Consequtive sections do not have same mode.
                section.append(sections[i - 1][-1])
                sections_n.append(np.array(section, dtype=float))
                section = sections[i][:-1].tolist()
        section.append(sections[i][-1])
        sections_n.append(np.array(section, dtype=float))
        return sections_n

    def _populate_sections_stride(self, csv_data, sections, stride=3):
        dense_sections = []
        for section in sections:
            mode_section = section[0, 3].astype(int)
            if mode_section == 999 or mode_section < 0:
                # Staying still or mode change.
                # No need to populating the points.
                dense_sections.append(section)
                continue
            # Populate by given stride.
            dense_section = []
            for i in range(len(section) - 1):
                # Iterate pair by pair.
                r_sect = section[i, 1]
                i_s = section[i, 0].astype(int)
                i_e = section[i + 1, 0].astype(int)
                alpha = csv_data[i_s : i_e + 1, 5]
                # Determine indexes where both pivot was unlifted.
                idxs0 = np.diff(np.sign(abs(alpha)))
                idxs0 = np.where(idxs0 > 0)[0] + i_s
                idxs = np.abs(np.diff(np.sign(alpha)))
                idxs = np.where(idxs > 1)[0] + i_s
                idxs = np.sort(np.concatenate((idxs, idxs0)))
                idxs[0] = i_s  # To avoid showing inplace rotations.
                # Get necessary indexes.
                divs = len(idxs) / stride
                if (divs - int(divs)) > 0.4:
                    max_cnt = int(divs) + 1
                    indexes = np.arange(max_cnt) * stride
                    indexes = idxs[indexes]
                    indexes = np.append(indexes, (indexes[-1] + i_e) // 2)
                else:
                    max_cnt = int(divs)
                    indexes = np.arange(max_cnt) * stride
                    indexes = idxs[indexes]
                # Get detailed data
                for ind in indexes[:-1]:
                    data = csv_data[ind]
                    data[0] = ind
                    dense_section.append(data)
            # Add end point.
            data = csv_data[i_e]
            data[0] = i_e
            dense_section.append(data)
            dense_sections.append(np.array(dense_section, dtype=float))
        return dense_sections

    def _populate_sections(self, csv_data, sections, length=30):
        """
        Populates data between each sequence section.
        """
        dense_sections = []
        for section in sections:
            mode_section = section[0, 3]
            if mode_section == 999 or mode_section < 0:
                # Staying still or mode change.
                # No need to populating the points.
                dense_sections.append(section)
                continue
            # Populate by given length.
            dense_section = []
            for i in range(len(section) - 1):
                # Iterate pair by pair.
                r_sect = section[i, 1]
                i_s = section[i, 0].astype(int)
                i_e = section[i + 1, 0].astype(int)
                # Calculate number of steps needed.
                n_step = np.ceil(r_sect / length).astype(int)
                step = np.ceil((i_e - i_s) / n_step).astype(int)
                for ind in range(i_s, i_e, step):
                    data = csv_data[ind]
                    data[0] = ind
                    dense_section.append(data)
            # Add end point.
            data = csv_data[i_e]
            data[0] = i_e
            dense_section.append(data)
            dense_sections.append(np.array(dense_section, dtype=float))
        return dense_sections

    def _cart2pixel(self, point):
        """Converts cartesian coordinte to pixel coordinate (x, y)."""
        pixel = np.zeros(2, dtype=int)
        pixel[0] = int(point[0] / self._p2mm) + self._center[0]
        pixel[1] = self._center[1] - int(point[1] / self._p2mm)
        return pixel

    def _find_center(self, frame):
        """
        Finds center of coordinate system based white border.
        """
        masked = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        _, masked = cv2.threshold(masked, 250, 255, cv2.THRESH_BINARY)
        # Find the external contours in the masked image
        contours, hierarchy = cv2.findContours(
            masked, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        # Filter biggest external contour.
        external_areas = np.array(
            [
                cv2.contourArea(contours[idx]) if elem > -1 else 0
                for idx, elem in enumerate(hierarchy[0, :, 2])
            ]
        )

        cnt = contours[np.argmax(external_areas)]
        x, y, w, h = cv2.boundingRect(cnt)
        xc = int(x + w / 2)
        yc = int(y + h / 2)
        # Frame parameters and center.
        f_height, f_width = frame.shape[:2]
        center = (xc, yc)
        return f_width, f_height, center, (x, y, w, h)

    def _process_video(self, sections, vid_path):
        """
        Gets frames for each section.
        """
        crop_flag = True
        cap = cv2.VideoCapture(vid_path)
        frames = []
        # Find center and limits
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        _, frame = cap.read()
        frame = frame[:, :, ::-1]
        self.f_width, self.f_height, self._center, bound = self._find_center(
            frame
        )
        x, y, w, h = bound
        for section in sections:
            frame_sec = []
            for line in section:
                idx = line[0].astype(int)
                # Set frame index and store frame.
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                _, frame = cap.read()
                if crop_flag:
                    frame = frame[y : y + h, x : x + w]
                frame_sec.append(frame[:, :, ::-1])
            frames.append(frame_sec)
        if crop_flag:
            self.f_height, self.f_width = frame.shape[:-1]
            self._center = (int(self.f_width / 2), int(self.f_height / 2))
        return frames

    def _blend_frames(self, frames, sections):
        """
        Creates blended snapshot of each section of movement.
        """
        h, w = self.f_height, self.f_width  # Size of mask.
        radii = 17
        blends = []
        for frame_sec, section in zip(frames, sections):
            alpha = np.ones(len(frame_sec)) * 0.4
            alpha[0] = 0.6
            alpha = np.linspace(0.3, 0.9, len(frame_sec))
            alpha[-1] = 1.0
            master = frame_sec[-1].copy()
            for i, (frame, data) in enumerate(zip(frame_sec, section)):
                input_cmd, mode, X, XI, XG, E, SHAPE = self._itemize(data)
                al = alpha[i]
                # Make mask based on position of robots.
                mask = np.zeros((h, w), dtype=np.uint8)
                for pos in np.reshape(X, (-1, 2)):
                    cent = self._cart2pixel(pos)
                    mask = cv2.circle(mask, cent, radii, 1, cv2.FILLED)
                weighted = cv2.addWeighted(frame, al, master, 1 - al, 0)
                master[mask > 0] = weighted[mask > 0]
            blends.append(master)
        return blends

    def _simplot_set(self, ax, title=None, fontsize=None):
        """Sets the plot configuration."""
        if title is not None:
            ax.set_title(title, fontsize=fontsize, pad=12)
            ax.set_xlabel("x axis", fontsize=fontsize)
            ax.set_ylabel("y axis", fontsize=fontsize)
        ax.axis("off")
        return ax

    def _plot_legends(
        self, ax, robots, light, path=True, shape=False, obstacle=True
    ):
        # Add robot legends.
        if path:
            ls = self._styles[0][1]
        else:
            ls = "none"
        legends = lambda ls, c, m, ms, mfc, mec, l: plt.plot(
            [],
            [],
            linestyle=ls,
            linewidth=3.5,
            color=c,
            marker=m,
            markersize=ms,
            markerfacecolor=mfc,
            markeredgecolor=mec,
            label=l,
        )[0]
        # Robots and path.
        handles = [
            legends(
                ls,
                self.light(self._colors[robot]),
                self._markers[robot],
                22,
                self.light(self._colors[robot], light),
                # self.light(self._colors[robot],light),
                "k",
                f"Robot {robot}",
            )
            for robot in robots
        ]
        labels = [f"R{robot}" for robot in robots]
        handlelength = 1.5
        # Obstacle.
        if obstacle:
            handles += [
                plt.Rectangle(
                    [0, 0],
                    width=6,
                    height=3,
                    linestyle="-",
                    linewidth=1.0,
                    edgecolor=to_rgba("k", 1.0),
                    facecolor="w",
                )
            ]
            labels += [f"Obs"]
        # Shape
        if shape:
            handlelength = 1.5
            ls = self._styles[0][1]
            handles += [
                legends(
                    ls, "orange", "o", 20, "orange", "orange", f"Target shape"
                )
            ]
            labels += [f"Desired"]
        ax.legend(
            handles=handles,
            labels=labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            fontsize=40,
            framealpha=0.8,
            facecolor="w",
            handlelength=handlelength,
            handletextpad=0.15,
            labelspacing=0.01,
            borderpad=0.15,
            borderaxespad=0.1,
            loc="best",
        )
        return ax

    def _plot_markers(self, ax, robots, pixels, light, size=16):
        """
        Places markers on path based on given length.
        """
        pixels = np.atleast_2d(pixels)
        if len(pixels) > 1:
            pixels = pixels[:-1]
        for robot in robots:
            ax.scatter(
                pixels[:, 2 * robot],
                pixels[:, 2 * robot + 1],
                marker=self._markers[robot],
                color=self.light(self._colors[robot], light),
                edgecolors="k",
                s=size**2,
                zorder=3,
            )
        return ax

    def _plot_scalebar(self, ax, pos=(70, -80), scale=20, unit="mm"):
        """
        Draws a scale bar on the plot.
         20 mm
        |-----|
        """
        pos_t = np.array(pos, dtype=float)
        pos_t[1] += 10
        pix_t = self._cart2pixel(pos_t)
        pos_l = np.array(pos, dtype=float)
        pos_l[0] -= scale / 2
        pos_r = np.array(pos, dtype=float)
        pos_r[0] += scale / 2
        pix_l = self._cart2pixel(pos_l)
        pix_r = self._cart2pixel(pos_r)
        # Text.
        ax.text(
            pix_t[0],
            pix_t[1],
            f"{scale:02d} {unit}",
            fontsize=50,
            ha="center",
            va="center",
        )
        # Scale bar body.
        ax.annotate(
            "",
            xy=(
                (pix_r[0] + 0.5) / self.f_width,
                (self.f_height - pix_r[1] - 0.5) / self.f_height,
            ),
            xycoords="axes fraction",
            xytext=(
                (pix_l[0] + 0.5) / self.f_width,
                (self.f_height - pix_l[1] - 0.5) / self.f_height,
            ),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc",
                color="k",
                linewidth=7.0,
                joinstyle="miter",
                capstyle="butt",
            ),
        )
        # Scale bar heads.
        ax.annotate(
            "",
            xy=(
                (pix_r[0] + 0.5) / self.f_width,
                (self.f_height - pix_r[1] - 1.0) / self.f_height,
            ),
            xycoords="axes fraction",
            xytext=(
                (pix_l[0] - 0.5) / self.f_width,
                (self.f_height - pix_l[1] - 1.0) / self.f_height,
            ),
            textcoords="axes fraction",
            arrowprops=dict(
                arrowstyle="|-|, widthA=0.5, widthB=.5",
                connectionstyle="arc",
                color="k",
                linewidth=4.0,
                joinstyle="miter",
                capstyle="butt",
            ),
        )
        return ax

    def _plot_timestamp(self, ax, timestamp, pos=(-100, 75)):
        """Draws time stamp on the plot."""
        pix = self._cart2pixel(pos)
        ax.text(
            pix[0],
            pix[1],
            f"{int(timestamp)} s",
            fontsize=50,
            ha="left",
            va="center",
        )
        return ax

    def _plot_section(
        self, ax, robots, pixels, light=0.6, ls=None, lw=7.0, last_section=True
    ):
        ls = self._styles[1][1] if ls is None else ls
        # Adjust pixels based on last_section.
        if last_section:
            l_pixels = pixels[-2:]
            pixels = None if len(pixels) < 3 else pixels[:-1]
            # Draw last section arrow.
            for robot in robots:
                # Draw line.
                ax.annotate(
                    "",
                    xy=(
                        (l_pixels[-1, 2 * robot] + 0.5) / self.f_width,
                        (self.f_height - l_pixels[-1, 2 * robot + 1] - 0.5)
                        / self.f_height,
                    ),
                    xycoords="axes fraction",
                    xytext=(
                        (l_pixels[-2, 2 * robot] + 0.5) / self.f_width,
                        (self.f_height - l_pixels[-2, 2 * robot + 1] - 0.5)
                        / self.f_height,
                    ),
                    textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-",
                        connectionstyle="arc3",
                        shrinkB=10,
                        color=self.light(self._colors[robot]),
                        linewidth=lw,
                        linestyle=ls,
                    ),
                )
                # Calculate start and end of arrow in axes fraction.
                XY = np.zeros((2, 2), dtype=float)
                XY[:, 0] = (l_pixels[:, 2 * robot] + 0.5) / self.f_width
                XY[:, 1] = (
                    self.f_height - l_pixels[:, 2 * robot + 1] - 0.5
                ) / self.f_height
                # Adjust arrow head starting point.
                dXY = dXY = np.diff(XY, axis=0)
                dXY = dXY / np.linalg.norm(dXY)
                XY[0] = XY[1]
                XY[1] = XY[0] + 0.01 * dXY
                # XY[0] = XY[1] - 0.01 * dXY
                # Draw arrow head.
                ax.annotate(
                    "",
                    xy=XY[1],  # End of arrow.
                    xycoords="axes fraction",
                    xytext=XY[0],  # Start of arrow.
                    textcoords="axes fraction",
                    arrowprops=dict(
                        arrowstyle="-|>,head_length=1.25,head_width=1.0",
                        connectionstyle="arc3",
                        shrinkB=0,
                        facecolor=self.light(self._colors[robot], light),
                        edgecolor=self.light("k", 0),
                        linewidth=2.0,
                    ),
                    zorder=5,
                )
        # Draw pixel path.
        if pixels is not None:
            for robot in robots:
                ax.plot(
                    pixels[:, 2 * robot],
                    pixels[:, 2 * robot + 1],
                    color=self.light(self._colors[robot]),
                    linewidth=lw,
                    linestyle=ls,
                )

    def _plot_shape(self, ax, section):
        input_cmd, mode, X, XI, XG, E, SHAPE = self._itemize(section)
        SHAPE = np.reshape(SHAPE[SHAPE < 999], (-1, 2)).astype(int)
        pixes = []
        for point in np.reshape(XG, (-1, 2)):
            pixes.extend(self._cart2pixel(point))
        pixes = np.reshape(pixes, (-1, 2)).astype(int)
        for inds in SHAPE:
            ax.plot(
                pixes[inds, 0],
                pixes[inds, 1],
                ls=self._styles[2][1],
                lw=6.0,
                c="orange",
                marker="o",
                markersize=20,
                zorder=1,
            )

    def _section_info(self, section):
        # Calculate corresponding pixels..
        section = np.atleast_2d(section)
        pixels = []
        ind_s = section[0][0]
        ind_f = section[-1][0]
        cmd_mode = section[-1][3]
        for sect in section:
            input_cmd, mode, X, XI, XG, E, SHAPE = self._itemize(sect)
            pixes = []
            for point in np.reshape(X, (-1, 2)):
                pixes.extend(self._cart2pixel(point))
            pixels.append(pixes)
        pixels = np.array(pixels, dtype=int)
        return ind_s, ind_f, cmd_mode, pixels

    def plot_single(
        self,
        blend,
        section,
        step,
        light,
        title=None,
        legend=False,
        obstacle=False,
        scalebar=False,
        timestamp=False,
    ):
        """
        Plots single section of movements.
        """
        letter = string.ascii_letters[step]
        fig, ax = plt.subplots(layout="constrained")
        # Plot the blended frame.
        ax.imshow(blend)
        # Calculate corresponding pixels.
        ind_s, ind_f, cmd_mode, pixels = self._section_info(section)
        # Plot sections.
        robots = np.arange(int(len(pixels[0]) / 2))
        self._plot_section(ax, robots, pixels, light=light)
        # Plot markers.
        ax = self._plot_markers(ax, robots, pixels, light)
        # Plot scalebar.
        if scalebar:
            ax = self._plot_scalebar(ax)
        if timestamp:
            ax = self._plot_timestamp(ax, ind_f / self._fps)
        if legend and (step < 1):
            # Plot legends.
            ax = self._plot_legends(ax, robots, light, obstacle=obstacle)
        # Set plot border and title
        if title == -1:
            mode = int(cmd_mode)
            move_name = "Walking" if mode else "Tumbling"
            title = f"({letter}): {move_name} in Mode {mode:1d}."
        ax = self._simplot_set(ax, title, 60)
        return fig, ax

    def _save_plot(self, fig, name):
        """Overwrites files if they already exist."""
        fig_name = os.path.join(os.getcwd(), f"{name}.pdf")
        fig.savefig(fig_name, bbox_inches="tight", pad_inches=0.05)

    def plot_transition(
        self,
        name=None,
        title=None,
        light=0.6,
        legend=False,
        obstacle=False,
        scalebar=False,
        timestamp=False,
    ):
        FIG_AX = []
        sections = self.sections
        blends = self.blends
        for i, (blend, section) in enumerate(zip(blends, sections)):
            input_cmd, mode, X, XI, XG, E, SHAPE = self._itemize(section[-1])
            if np.any(input_cmd == 999):
                continue
            fig, ax = self.plot_single(
                blend,
                section,
                i,
                light,
                title=title,
                legend=legend,
                obstacle=obstacle,
                scalebar=scalebar,
                timestamp=timestamp,
            )
            scalebar &= False
            # Add frame to whole plot.
            fig.patch.set_edgecolor((0, 0, 0, 1.0))
            fig.patch.set_linewidth(2)
            if name is not None:
                self._save_plot(fig, f"{name}_{i}")
            FIG_AX.append((fig, ax))
        return FIG_AX

    def plot_shape(
        self,
        name=None,
        title=None,
        light=0.6,
        legend=True,
        desired=False,
        initial=False,
        scalebar=False,
        timestamp=-1,
    ):
        """
        Draws final pattern of the robots.
        """
        fig, ax = plt.subplots(layout="constrained")
        # Plot the blended frame.
        if initial:
            blend = self.frames[0][0]
            section = [self.sections[0][0]]
            desired = False
        else:
            blend = self.blends[-1]
            section = self.sections[-1]
        ax.imshow(blend)
        # Calculate corresponding pixels.
        ind_s, ind_f, cmd_mode, pixels = self._section_info(section[-1])
        robots = np.arange(len(pixels[-1]) // 2)
        # Plot markers.
        self._plot_markers(ax, robots, pixels, light, size=20)
        # Plot shapes.
        if desired:
            self._plot_shape(ax, section[-1])
        # Plot scalebar.
        if scalebar:
            ax = self._plot_scalebar(ax)
        if timestamp >= 0:
            ax = self._plot_timestamp(ax, timestamp)
        # Plot legends.
        if legend:
            ax = self._plot_legends(
                ax, robots, light, path=False, shape=desired
            )
        ax = self._simplot_set(ax, title, 60)
        # Add frame to whole plot.
        fig.patch.set_edgecolor((0, 0, 0, 1.0))
        fig.patch.set_linewidth(2)
        if name is not None:
            self._save_plot(fig, f"{name}")
        return [(fig, ax)]

    def _plot_path_section(
        self, ax, robots, section, light, last_section=False
    ):
        # Calculate pixels.
        ind_s, ind_f, cmd_mode, pixels = self._section_info(section)
        # Calculate corresponding pixels.
        self._plot_section(
            ax, robots, pixels, light, ls="-", lw=4, last_section=last_section
        )
        # Plot markers.
        ax = self._plot_markers(ax, robots, pixels, light, 8)

    def plot_path(
        self,
        robots,
        name=None,
        title=None,
        light=0.6,
        legend=False,
        obstacle=False,
        scalebar=False,
        timestamp=False,
    ):
        # Filter final stand stills.
        sections = list(filter(lambda item: item[0, 1] < 998, self.sections))
        # Remocve last zero command section if exists.
        if sections[-1][0, 1] < 1:
            sections = sections[:-1]
        #
        FIG_AX = []
        fig, ax = plt.subplots(layout="constrained")
        # Blend initial and final frame.
        radii = 17
        frame = self.frames[0][0]
        master = self.frames[-1][-1]
        data = sections[0][0]
        alpha = 0.5
        input_cmd, mode, X, XI, XG, E, SHAPE = self._itemize(data)
        # Make mask based on position of robots.
        mask = np.zeros((self.f_height, self.f_width), dtype=np.uint8)
        for pos in np.reshape(X, (-1, 2)):
            cent = self._cart2pixel(pos)
            mask = cv2.circle(mask, cent, radii, 1, cv2.FILLED)
        weighted = cv2.addWeighted(frame, alpha, master, 1 - alpha, 0)
        master[mask > 0] = weighted[mask > 0]
        # Plot the blended frame.
        ax.imshow(master)
        # Plot paths for the all sections.
        for section in sections[:-1]:
            self._plot_path_section(ax, robots, section, light, False)
        # Plot last section.
        self._plot_path_section(ax, robots, sections[-1], light, True)
        # Get info.
        ind_s, ind_f, cmd_mode, pixels = self._section_info(sections[-1])
        # Plot scalebar.
        if scalebar:
            ax = self._plot_scalebar(ax)
        if timestamp:
            ax = self._plot_timestamp(ax, ind_f / self._fps)
        if legend:
            # Plot legends.
            ax = self._plot_legends(ax, robots, light, obstacle=obstacle)
        #
        ax = self._simplot_set(ax, title)
        # Add frame to whole plot.
        fig.patch.set_edgecolor((0, 0, 0, 1.0))
        fig.patch.set_linewidth(2)
        # Save.
        if name is not None:
            self._save_plot(fig, f"{name}")
        return fig, ax


def example_1_old():
    save = False
    light = 0.6
    FIG_AX = []
    # Q to S3S2
    file_dir = os.path.join(os.getcwd(), "passage")
    path_lists = [
        os.path.join(file_dir, "logs.mp4"),
        os.path.join(file_dir, "logs.csv"),
    ]
    process = ProcessVideo(
        path_lists,
        length=30,
        skip_mode_change=True,
        combine_similar_sections=True,
    )
    # All movement snapshots.
    name = None  # "example_1_snapshot"
    title = -1
    FIG_AX = process.plot_transition(
        name=name, title=title, light=light, scalebar=True, timestamp=True
    )
    # Start and end.
    name = None  # "example_start"
    title = None
    FIG_AX += process.plot_shape(
        name, title, light, legend=False, initial=True, scalebar=True
    )
    name = None  # "example_1_desired"
    FIG_AX += process.plot_shape(
        name,
        title,
        light,
        legend=True,
        desired=True,
        scalebar=True,
        timestamp=208,
    )


def example_1():
    light = 0.6
    FIG_AX = []
    # Q to S3S2
    file_dir = os.path.join(os.getcwd(), "passage")
    path_lists = [
        os.path.join(file_dir, "logs.mp4"),
        os.path.join(file_dir, "logs.csv"),
    ]
    process = ProcessVideo(
        path_lists,
        length=30,
        skip_mode_change=False,
        combine_similar_sections=True,
    )
    # Robots 0, 2.
    robots = [0, 2]
    name = None  # "example_1_02"
    title = None
    FIG_AX += process.plot_path(
        robots,
        name,
        title,
        light,
        legend=True,
        obstacle=True,
        scalebar=True,
        timestamp=True,
    )
    # Robots 1, 3.
    robots = [1, 3]
    name = None  # "example_1_13"
    FIG_AX += process.plot_path(robots, name, title, light)


########## test section ################################################
if __name__ == "__main__":
    example_1()
    plt.show()
