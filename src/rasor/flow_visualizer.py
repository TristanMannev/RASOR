from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

import cv2
import matplotlib as mpl  # type: ignore[import-untyped]
import numpy as np
import scipy
import torch
import tqdm  # noqa: F401
from cellpose import models
from skimage.measure import find_contours

if TYPE_CHECKING:
    from collections.abc import Sequence

_hsv = mpl.colormaps["hsv"]
_colors = [tuple(int(c * 255) for c in x[:3]) for x in map(_hsv, [i / 200 for i in range(200)])]


def _convert_to_uint8(frame: np.ndarray) -> np.ndarray:
    """Conversion from a floating or integer frame to uint8 displayable image.

    Args:
        frame (np.ndarray): Video frame 2D (If 3D video, the caller should reduce the depth dimension)
            Integer (resp. Floating) frames are expected to be normalized in [0, 255] (resp. [0.0, 1.0])
            If this is not the case, a simple normalization is performed before conversion.
            Shape: (H, W, C), dtype: float | int

    Returns:
        np.ndarray: Displayable frame in uint8
            Shape: (H, W, C), dtype: uint8

    """
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max() > 1.0:
            warnings.warn(
                "Floating videos are expected to be normalized in [0, 1].\n"
                "The simple normalization implemented here is suboptimal.",
                stacklevel=2,
            )
            frame = frame / frame.max()
        frame = (frame * 255).round()
    elif frame.max() > 255:
        warnings.warn(
            "Integer videos are expected to be normalized in [0, 255].\n"
            "The simple normalization implemented here is suboptimal.",
            stacklevel=2,
        )

        frame = (frame * (255 / frame.max())).round()

    return frame.astype(np.uint8)


# def display_lifetime(tracks: Collection[byotrack.Track]) -> None:
#     """Display the lifetime of tracks.

#     Active tracks are in white. (Tracks on x-axis, Frames on y-axis)

#     Args:
#         tracks (Collection[byotrack.Track]): Tracks

#     """
#     start = min(track.start for track in tracks)
#     tracks_tensor = byotrack.Track.tensorize(tracks)

#     mask = (~torch.isnan(tracks_tensor).any(dim=2, keepdim=True)).numpy() * 255
#     mask = np.concatenate([mask] * 3, axis=2)

#     fig = plt.figure(figsize=(24, 16), dpi=100)
#     plt.xlabel("Track id")
#     plt.ylabel("Frame")
#     plt.imshow(mask)
#     # Hacky axe relabelling
#     fig.axes[0].set_yticks(fig.axes[0].get_yticks()[1:-1])
#     fig.axes[0].set_yticklabels(fig.axes[0].get_yticks().astype(np.int32) + start)
#     plt.show()


def temporal_projection(
    tracks_tensor: torch.Tensor,
    colors: Sequence[tuple[int, int, int]] = ((255, 255, 255),),
    background: np.ndarray | None = None,
    *,
    color_by_time=False,
) -> np.ndarray:
    """Project all given tracks into a single image.

    A track is displayed as the line of its consecutive positions. Track's undefined positions
    are not displayed.

    3d tracks and images are not supported

    Args:
        tracks_tensor (torch.Tensor): Tracks tensor (See `byotrack.Track.tensorize`)
            Positions of each track at each frame. (NaN if not defined).
            Shape: (T, N, 2)
        colors (Sequence[tuple[int, int, int]]): Color sequence. When we color each track independently,
            each track is associated with a color (ith track with ith color, it loops if more tracks than colors).
            For time coloring, each frame has its own color (same across all tracks).
            Default: ((255, 255, 255),) (Everything is white)
        background (np.ndarray | None): Optional frame to display behind the tracks.
            If not given, we set the background as 0, with the smallest size to contain all tracks.
            Default: None
            Shape: (H, W[, C])
        color_by_time (bool): If set to True, each frame has its own color. Otherwise, each track has its own color.
            Color by time is much slower (cv2 do not allow complex multilines draw)
            Default: False

    Returns:
        np.ndarray: Projection image
            Shape: (H, W, 3), dtype: np.uint8
    """
    is_defined = ~torch.isnan(tracks_tensor).any(dim=-1)
    frames = torch.arange(len(tracks_tensor))

    if background is not None:
        if background.ndim == 2:
            background = background[..., None]
        image = _convert_to_uint8(background)
        if image.shape[2] == 1:
            image = np.concatenate([image] * 3, axis=-1)
        offset = torch.zeros(2)
    else:
        offset = torch.min(tracks_tensor[is_defined], dim=0).values
        shape = (torch.max(tracks_tensor[is_defined], dim=0).values - offset).round().to(torch.int32) + 1
        image = np.zeros((*shape, 3), dtype=np.uint8)

    for track_i, points in enumerate(tracks_tensor.permute(1, 0, 2)):
        if not is_defined[:, track_i].any():  # No points for this track
            continue

        points = points[is_defined[:, track_i]] - offset

        if color_by_time:
            valid_frames = frames[is_defined[:, track_i]]

            previous = points[0]
            for frame_id, point in zip(valid_frames[1:], points[1:], strict=True):
                cv2.line(
                    image,
                    previous.numpy().round().astype(np.int32)[::-1],
                    point.numpy().round().astype(np.int32)[::-1],
                    colors[frame_id % len(colors)],
                )
                previous = point
        else:
            cv2.polylines(
                image,
                points.numpy().round().astype(np.int32)[None, :, None, ::-1],
                isClosed=False,
                color=colors[track_i % len(colors)],
            )

    return image


def transform_points(flow_map: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply the given flow to the given points.

    Args:
        flow (np.ndarray): Flow to apply
            Shape: (2, H, W)
        points (np.ndarray): Points to transform
            Shape: (N, 2)

    Returns:    np.ndarray: Transformed points"""
    flow = np.empty_like(points)

    for axis in range(flow_map.shape[0]):
        flow[:, axis] = scipy.ndimage.map_coordinates(flow_map[axis], points.T, mode="nearest")

    return points + flow


class FlowVisualizer:
    """Optical flow visualization using a moving grid above the video.

    The video is displayed with a uniform grid (control points) above that moves according to the given flow maps.
    If the optical flow algorithm is good, the points should moves along the video.

    It only supports 2D videos.

    Keys:
        * `space`: Pause/Unpause the video
        * w/x: Move backward/forward in the video (when paused)
        * g: Reset the control points as a grid

    Keys can be modified in the dict `keys` (PAUSE, RIGHT, LEFT, GRID).

    Note:
        To prevent lagging, even when precompute is False, optical flow map are saved and never computed twice.
        You should not use this function with too large videos. (Use slicing to reduce the size of a video)

    Attributes:
        video (Sequence[np.ndarray] | np.ndarray): The video of interest (2D)
            Shape: (T, H, W, C)
        optflow (byotrack.OpticalFlow): Optical flow algorithm to visualize
        grid_step (int): size (in pixels) between control points
        points (np.ndarray): Control points (grid)
            Shape: (N, 2), dtype: float64
        precompute (bool): Compute all the flows first (bi directional) to prevent lagging
            Default: False
        tracks_colors (list[tuple[int, int, int]]): Colors to use for tracks
            Colors are picked circularly given the track identifier.
            By default we use the hsv colormaps from matplotlib
        scale (int): Up or Down scale the display.
            Default: 1 (no scaling)
    """

    # KEY MAPPING
    keys: ClassVar[dict[str, str]] = {
        "QUIT": "q",
        "PAUSE": " ",
        "RIGHT": "x",
        "LEFT": "w",
        "GRID": "g",
    }

    window_name = "ByoTrack FlowViz"

    def __init__(
        self,
        video: Sequence[np.ndarray] | np.ndarray,
        flow_maps: dict[tuple[int, int], np.ndarray],
        grid_step=20,
        gap=1,
        grid_type="uniform",
    ):
        self.video = video
        self.flow_maps = flow_maps
        self.gap = gap
        self.grid_step = grid_step
        self.grid_type = grid_type
        self.frame_shape = video[0].shape[:-1]
        print("frame shape", self.frame_shape)
        if len(self.frame_shape) != 2:
            raise ValueError("This visualization only supports 2D videos")

        self.n_frames = len(video)
        self.colors = _colors
        self.scale = 1
        self._frame_id = 0
        if self.grid_type not in ("uniform", "personalized"):
            raise ValueError(f"grid_type should be either 'uniform' or 'personalized', got {self.grid_type}")
        elif self.grid_type == "uniform":
            self._grid = (
                np.indices(self.frame_shape, dtype=np.float64)[:, ::grid_step, ::grid_step].reshape(2, -1).transpose()
            )
        elif self.grid_type == "personalized":
            self._grid = self.personnalized_grid()
        self.points = self._grid.copy()
        print("points shape", self.points.shape)

        self._running = False

    def run(self, frame_id=0, fps=20) -> None:
        """Run the visualization.

        Args:
            frame_id (int): Starting frame_id
            fps (int): Frame rate
        """
        try:
            self._frame_id = frame_id
            self._run(fps)
        finally:
            cv2.destroyWindow(self.window_name)

    def _run(self, fps=20) -> None:
        self.points = self._grid

        src_frame_id = self._frame_id
        # src = self.optflow.preprocess(self.video[self._frame_id])
        # dst = src

        while True:
            self._frame_id += self._running

            if self._frame_id >= self.n_frames:
                self._running = False
                self._frame_id = self.n_frames - 1

            frame = self.video[self._frame_id]

            if src_frame_id != self._frame_id:  # Change of frame, let's update
                pair = (src_frame_id, self._frame_id)
                if pair in self.flow_maps:
                    self.points = transform_points(self.flow_maps[pair], self.points)
                else:
                    raise ValueError(f"Flow map for frames {pair} not found in given flow_maps")

                src_frame_id = self._frame_id

            # Drawing
            draw = _convert_to_uint8(frame)
            if draw.shape[2] not in (1, 3):
                raise ValueError("We only support Grayscale images (C=1) or RGB images (C=3)")

            if draw.shape[2] == 1:
                draw = np.concatenate([draw] * 3, axis=2)

            if self.scale != 1:
                draw = cv2.resize(draw, None, fx=self.scale, fy=self.scale)

            for k, (i, j) in enumerate(self.points * self.scale):
                color = self.colors[k % len(self.colors)] if self.grid_type == "uniform" else (255, 0, 0)
                cv2.circle(draw, (round(j), round(i)), 3, color)

            # Display the resulting frame
            cv2.imshow(self.window_name, np.flip(draw, 2))
            cv2.setWindowTitle(self.window_name, f"Frame {self._frame_id} / {self.n_frames}")

            # Handle user actions
            key = cv2.waitKey(1000 // fps) & 0xFF
            if self.handle_actions(key):
                break

    def handle_actions(self, key: int) -> bool:
        """Handle inputs from user.

        Return True to quit

        Args:
            key (int): Key input from user

        Returns:
            bool: True to quit visualization
        """
        if key == ord(self.keys["QUIT"]):
            return True

        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return True

        if key == ord(self.keys["PAUSE"]):
            self._running = not self._running

        if not self._running and key == ord(self.keys["LEFT"]):  # Prev
            self._frame_id = (self._frame_id - self.gap) % self.n_frames

        if not self._running and key == ord(self.keys["RIGHT"]):  # Next
            self._frame_id = (self._frame_id + self.gap) % self.n_frames

        if key == ord(self.keys["GRID"]):
            self.points = self._grid

        return False

    def personnalized_grid(self):
        """Set a personnalized grid with points only at the border of cells."""
        start_frame = self.video[self._frame_id]
        model = models.CellposeModel(gpu=True)
        seg, _, _ = model.eval(start_frame, flow_threshold=0.4, cellprob_threshold=0.0)  # TODO pb segmentation

        labels = np.unique(seg)
        cells_points = []

        for label in labels:
            print("label", label)
            if label == 0:
                continue
            mask = seg == label
            contour = find_contours(mask)
            # contour[:, 1] = X, contour[:, 0] = Y
            step = 1
            sampled_contour = contour[0][::step]

            for point in sampled_contour:
                print(point[0])
                print(type(point))
                print(type(cells_points))
                cells_points.append(point)
        cells_points = np.array(cells_points, dtype=np.float64)
        return cells_points


def flow_to_rgb(flow: np.ndarray, clip: float | None = None) -> np.ndarray:
    """Converts a 2D optical flow into a RBG map.

    The angle is converted into a color, the magnitude into luminosity.
    (Follow what is done by OpenCV for visualization)

    See `get_flow_wheel` to get the color wheel for each direction.

    Args:
        flow (np.ndarray): Flow to convert to rgb
            Shape: (2, H, W)
        clip (float | None): Clip the magnitude to this maximum value
            Default: None

    Returns:
        np.ndarray: RGB flow visualization
            Shape: (H, W, 3)

    """
    hsv = np.zeros((*flow.shape[1:], 3), dtype=np.uint8)

    magnitude, angle = cv2.cartToPolar(flow[1], flow[0])
    if clip is not None:
        magnitude = np.clip(magnitude, 0, clip)
    else:
        clip = magnitude.max()

    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = magnitude / clip * 255

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def get_flow_wheel(size=128, *, mask=True) -> np.ndarray:
    """Build the color wheel that is used for `flow_to_rgb`.

    Args:
        size (int): The generated wheel will have 2 * size + 1 width and height.
            Default: 128
        mask (bool): Mask values outside of the circle centered on the middle.
            Default: True

    Returns:
        np.ndarray: Color wheel
            Shape: (2 * size + 1, 2 * size + 1, 3)

    """
    wheel = np.indices((2 * size, 2 * size), dtype=np.float32) - size

    if mask:
        mask = np.sqrt(np.sum(wheel**2, axis=0)) > size
        wheel[:, mask] = 0

    return flow_to_rgb(wheel, size)
