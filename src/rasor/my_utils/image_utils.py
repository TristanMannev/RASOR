from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tifffile as tiff
from skimage.measure import find_contours, regionprops
from torch import nn

from deepfluxoptique.my_utils.flow_viz import flow_to_image


def normalisation_params(frames_path_list):
    """Return the global minimum and maximum pixel values across all frame's paths in the video."""
    global_min = np.min(tiff.imread(str(frames_path_list[0])))
    global_max = np.max(tiff.imread(str(frames_path_list[0])))

    for i in range(1, len(frames_path_list)):
        current_min = np.min(tiff.imread(str(frames_path_list[i])))
        current_max = np.max(tiff.imread(str(frames_path_list[i])))

        global_min = min(global_min, current_min)
        global_max = max(global_max, current_max)

    return global_min, global_max


def interpole_and_pad(image, target_shape=(1024, 1024)):
    """Interpolate and pad image to given shape 2D or 3D."""
    if any(dim > target for dim, target in zip(image.shape, target_shape, strict=False)):
        dim_factors = [target / dim for target, dim in zip(target_shape, image.shape, strict=False)]
        image = scipy.ndimage.interpolation.zoom(image, dim_factors, order=3)
    return pad(image, target_shape)[0]


def interpole_and_pad_mask(image, target_shape=(1024, 1024)):
    """Interpolate and pad image to given shape 2D or 3D."""
    if any(dim > target for dim, target in zip(image.shape, target_shape, strict=False)):
        dim_factors = [target / dim for target, dim in zip(target_shape, image.shape, strict=False)]
        image = scipy.ndimage.interpolation.zoom(image, dim_factors, order=0)
    return pad(image, target_shape)[0]


def pad(array, shape):
    """Zero-pads an array to a given shape. Returns the padded array and crop slices."""
    if array.shape == tuple(shape):
        return array, ...

    padded = np.zeros(shape, dtype=array.dtype)
    offsets = [int((p - v) / 2) for p, v in zip(shape, array.shape, strict=False)]
    slices = tuple([slice(offset, l + offset) for offset, l in zip(offsets, array.shape, strict=False)])
    padded[slices] = array

    return padded, slices


def create_grid(shape, spacing=20):
    """Create a grid image with lines every `spacing` pixels."""
    grid = np.zeros(shape)
    grid[::spacing, :] = 1
    grid[:, ::spacing] = 1
    return grid


def iou(array1, array2, labels=None, include_zero=False):
    """Computes the IoU overlap between two arrays.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    ioum = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(np.logical_or(array1 == label, array2 == label))
        ioum[idx] = top / bottom if bottom > 0 else 0.0
    return ioum


def dice(array1, array2, labels=None, include_zero=False):
    """Computes the dice overlap between two arrays for a given set of integer labels.

    Parameters:
        array1: Input array 1.
        array2: Input array 2.
        labels: List of labels to compute dice on. If None, all labels will be used.
        include_zero: Include label 0 in label list. Default is False.
    """
    if labels is None:
        labels = np.concatenate([np.unique(a) for a in [array1, array2]])
        labels = np.sort(np.unique(labels))
    if not include_zero:
        labels = np.delete(labels, np.argwhere(labels == 0))

    dicem = np.zeros(len(labels))
    for idx, label in enumerate(labels):
        top = 2 * np.sum(np.logical_and(array1 == label, array2 == label))
        bottom = np.sum(array1 == label) + np.sum(array2 == label)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon
        dicem[idx] = top / bottom
    return dicem


def relabel_cells(array, with_labels):
    """Relabel an segmentation mask with labels from another one."""
    relabeled = np.zeros_like(array)
    for region in regionprops(array):
        if region.label == 0:
            continue
        # Get the most common label in the region from with_labels
        labels_in_region = with_labels[region.coords[:, 0], region.coords[:, 1]]
        label_count = Counter(labels_in_region)
        if max(label_count.keys()) == 0:
            most_common_label = 0
        else:
            label_count.pop(0, None)
            most_common_label = label_count.most_common(1)[0][0]
        relabeled[region.coords[:, 0], region.coords[:, 1]] = most_common_label
    return relabeled


def visualize_labels(mask, ax=None):
    """Visualize a segmentation mask with written labels."""
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask, interpolation="nearest")
    ax.axis("off")

    for region in regionprops(mask):
        if region.label == 0:
            continue
        y, x = region.centroid
        ax.text(
            x,
            y,
            str(region.label),
            color="white",
            fontsize=10,
            ha="center",
            va="center",
        )


def viz(flo, ax=None):
    """Visualisation of flow, code from raft."""
    flo = flo.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_to_image(flo)
    if ax is None:
        plt.imshow(flo)
    else:
        ax.imshow(flo)


def normalise(image, global_min, global_max):
    """Normalize image to [0, 1] range using global min and max."""
    return (image - global_min) / (global_max - global_min)


def mask_to_gaussian(segmentation, sigma_scale=0.5):
    """Convert a segmentation mask to a Gaussian heatmap."""
    heatmap = np.zeros_like(segmentation, dtype=np.float32)
    y_grid, x_grid = np.indices(segmentation.shape)
    for region in regionprops(segmentation):
        if region.label == 0:
            continue
        y, x = region.centroid
        cell_sigma = (region.equivalent_diameter_area / 2) * sigma_scale
        cell_sigma = max(cell_sigma, 1)  # Ensure sigma is at least 1 to avoid very sharp peaks
        heatmap += np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * cell_sigma**2)).astype(np.float32)
    return heatmap


def linear_augment(start, end, max_weight, epoch):
    """Linearly increase the weight from 0 to max_weight between start and end epochs."""
    if epoch < start:
        return 0.0
    if epoch >= end:
        return max_weight
    return max_weight * ((epoch - start) / (end - start))


class NormalizeMinMax(nn.Module):
    """Return normalized tensor based on min, max."""

    def __init__(self, global_min, global_max, eps=1e-7):
        super().__init__()
        self.min = global_min
        self.max = global_max
        self.eps = eps

    def forward(self, x):
        """Forward."""
        return (x - self.min) / (self.max - self.min + self.eps)


def plot_contours(seg, color, ax=None):
    """Plot contours of a segmentation mask on a given axis."""
    if ax is None:
        ax = plt.gca()

    labels = np.unique(seg)

    for label in labels:
        if label == 0:
            continue
        mask = seg == label
        contours = find_contours(mask, level=0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], color=color, linewidth=1)


if __name__ == "__main__":
    img = tiff.imread("data/train_data/Fluo-N2DH-GOWT1_train/01_ERR_SEG_TRA/relabeled_00.tif")
    hm = mask_to_gaussian(img, sigma_scale=0.5)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.imshow(hm, cmap="hot")
    plt.show()
