# Core library imports
import random
from pathlib import Path

# Third-party imports
import numpy as np
import tifffile as tiff
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.transforms import v2

# Local imports
from deepfluxoptique.my_utils.configs import (
    SHAPE,
)
from deepfluxoptique.my_utils.image_utils import interpole_and_pad, interpole_and_pad_mask


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


class VxmVideoDataset(Dataset):
    """PyTorch Dataset for VoxelMorph registration data on video."""

    def __init__(
        self,
        video_path,
        seg_path=None,
        mode="train",
        seg_type=None,
        gap=1,
        multiplier=3,
    ) -> None:
        """Parameters

        video_path : str
            Path to the video frames.
        seg_path : str
            Path to the segmentation frames (optional).
        mode : str
            Mode of the dataset: "train" or "test".
        seg_type : str
            Type of segmentation to use for training: None, "binary", "labels" or "heatmaps".
        gap : int
            In training mode : Maximum gap between the two frames to select for registration, in test mode : fixed gap between the two frames to select for registration.
        seg_type : str
            Type of segmentation to use for training: None, labels or heatmaps.
        """
        self.video_path = Path(video_path)
        self.seg_path = Path(seg_path) if seg_path is not None else None
        self.mode = mode
        self.seg_type = seg_type
        self.get_path_list()
        self.len_video = len(self.frames_path_list)
        self.transform = self.get_transforms(n=self.len_video)
        if not gap < self.len_video // 2:
            raise ValueError(f"max_gap {gap} should be smaller than frames number // 2 {self.len_video // 2}.")
        self.gap = gap
        self.multiplier = multiplier
        if self.mode == "train":
            self.nb_pairs = self.len_video * self.multiplier
        elif self.mode == "train_in_order":
            # sum from 1 to gap of the (len_video - gap)
            self.nb_pairs = self.gap * self.len_video - self.gap * (self.gap - 1) // 2
        elif self.mode == "test":
            self.nb_pairs = self.len_video - self.gap

    def __getitem__(self, idx):
        """Normalize video and generate pairs of frame from a single video.

        dict
            A dictionary containing the source and target frames.
        """
        if self.mode == "train":
            # Select 2 frames with a random gap between -self.gap and self.gap (inclusive) (excluding 0)
            gap = np.random.randint(-self.gap, self.gap + 1)
            if gap == 0:
                gap = 1
            idx1 = idx % (self.len_video - 1)  # to avoid out of index error when selecting pairs of frames
            idx2 = idx1 - gap if (idx1 + gap > (self.len_video - 1) or idx1 + gap < 0) else idx1 + gap
        elif self.mode == "train_in_order":
            # Select 2 frames with a gap between 1 and self.gap (inclusive) in order
            current_gap = 1
            nb_pairs_current_gap = self.len_video - current_gap
            while idx >= nb_pairs_current_gap:
                idx -= nb_pairs_current_gap
                current_gap += 1
                nb_pairs_current_gap = self.len_video - current_gap
            idx1 = idx
            idx2 = idx1 + current_gap
        elif self.mode == "test":
            # Select 2 frames with a fixed gap
            idx1 = idx
            idx2 = idx1 + self.gap

        source_img = tiff.imread(self.frames_path_list[idx1])
        target_img = tiff.imread(self.frames_path_list[idx2])

        # Interpole and pad images to HEIGHT, WIDTH if needed
        moving_resized = interpole_and_pad(source_img, SHAPE)
        fixed_resized = interpole_and_pad(target_img, SHAPE)

        # To tensor
        source = torch.from_numpy(moving_resized).float()
        target = torch.from_numpy(fixed_resized).float()

        # Normalisation + augmentation if augment
        source = self.transform(source)
        target = self.transform(target)

        if self.seg_path is not None:
            source_seg = tiff.imread(self.seg_path_list[idx1])
            target_seg = tiff.imread(self.seg_path_list[idx2])
            source_seg = interpole_and_pad_mask(source_seg, SHAPE)
            target_seg = interpole_and_pad_mask(target_seg, SHAPE)
            # For testing, we don't need further processing
            if self.mode == "train" or self.mode == "train_in_order":
                if self.seg_type == "binary" or self.seg_type is None:
                    source_seg = source_seg > 0
                    target_seg = target_seg > 0
                # elif self.seg_type == "labels":
                #     target_seg_oh = (
                #         torch.nn.functional.one_hot(
                #             target_seg.long(), num_classes=NB_CLASSES
                #         )
                #         .permute(0, 3, 1, 2)
                #         .float()
                #     )
                #     source_seg_oh = (
                #         torch.nn.functional.one_hot(
                #             source_seg.long(), num_classes=NB_CLASSES
                #         )
                #         .permute(0, 3, 1, 2)
                #         .float()
                #     )
                # Not enough memory to use labels (and one hot)
                source_seg = torch.from_numpy(source_seg).float()
                target_seg = torch.from_numpy(target_seg).float()
            return {
                "source": source,
                "target": target,
                "source_seg": source_seg,
                "target_seg": target_seg,
                "idx1": idx1,
                "idx2": idx2,
            }

        return {"source": source, "target": target}

    def get_transforms(self, n):
        """Get global min and global max from n frames and return normalisation and transformation functions."""
        global_min = np.min(tiff.imread(str(self.frames_path_list[0])))
        global_max = np.max(tiff.imread(str(self.frames_path_list[0])))

        for i in range(1, n):
            current_min = np.min(tiff.imread(str(self.frames_path_list[i])))
            current_max = np.max(tiff.imread(str(self.frames_path_list[i])))

            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

        base_transforms = [NormalizeMinMax(global_min=global_min, global_max=global_max)]

        return v2.Compose(base_transforms)

    def __len__(self):
        """Return nbr of paires to train on per epoch."""
        return self.nb_pairs

    def get_path_list(self):
        """Create a list with all the frame's path and segmentation's path."""
        self.frames_path_list = sorted(self.video_path.iterdir())
        if self.seg_path is not None:
            if self.seg_type == "heatmaps":
                self.seg_path = self.seg_path.parent / "SEG_HEATMAP"
            self.seg_path_list = sorted(self.seg_path.iterdir())


class VxmMultipleVideoDataset(Dataset):
    """PyTorch Dataset for VoxelMorph registration data on multiple videos."""

    def __init__(
        self,
        video_paths,
        seg_paths=None,
        mode="train",
        seg_type=None,
        gap=1,
        multiplier=3,
    ) -> None:
        """Parameters

        video_paths : list
            List of paths to the video frames.
        seg_paths : list
            List of paths to the segmentation frames (optional).
        mode : str
            Mode of the dataset: "train" or "test".
        seg_type : str
            Type of segmentation to use for training: None, "binary", "labels" or "heatmaps".
        gap : int
            In training mode : Maximum gap between the two frames to select for registration, in test mode : fixed gap between the two frames to select for registration.
        seg_type : str
            Type of segmentation to use for training: None, labels or heatmaps.
        """
        self.video_paths = video_paths
        self.seg_paths = seg_paths
        self.mode = mode
        self.seg_type = seg_type
        self.get_path_list()
        self.compute_video_min_max()
        self.gap = gap
        self.multiplier = multiplier
        if self.mode == "train":
            self.nb_pairs = len(self.frames_path_list) * self.multiplier
        elif self.mode == "test":
            self.nb_pairs = len(self.frames_path_list) - self.gap

    def __getitem__(self, idx):
        """Normalize video and generate pairs of frame.

        dict
            A dictionary containing the source and target frames.
        """
        if self.mode == "train":
            # Select 2 frames with a random gap between -self.gap and self.gap (inclusive) (excluding 0)
            gap = np.random.randint(-self.gap, self.gap + 1)
            possible_gap = [k for k in range(-self.gap, self.gap + 1) if k != 0]
            gap = random.choice(possible_gap)
            idx1 = idx % (len(self.frames_path_list) - 1)
            # find the limts of the video that idx1 belongs to
            video_idx = 0
            limit = self.limits[video_idx]
            while idx1 >= limit:
                video_idx += 1
                limit = self.limits[video_idx]
            start_idx = self.limits[video_idx - 1]
            end_idx = self.limits[video_idx] - 1

            idx2 = idx1 + gap

            if idx2 < start_idx or idx2 > end_idx:
                # if out of limits, select the other frame in the opposite direction
                idx2 = idx1 - gap
                if idx2 < start_idx:
                    idx2 = start_idx
                elif idx2 > end_idx:
                    idx2 = end_idx

        elif self.mode == "test":
            # Select 2 frames with a fixed gap
            idx1 = idx
            idx2 = idx1 + self.gap

        source_img = tiff.imread(self.frames_path_list[idx1])
        target_img = tiff.imread(self.frames_path_list[idx2])

        # Interpole and pad images to HEIGHT, WIDTH if needed
        moving_resized = interpole_and_pad(source_img, SHAPE)
        fixed_resized = interpole_and_pad(target_img, SHAPE)

        # To tensor
        source = torch.from_numpy(moving_resized).float()
        target = torch.from_numpy(fixed_resized).float()

        # Normalisation
        vmin = self.video_min[video_idx - 1]
        vmax = self.video_max[video_idx - 1]

        source = (source - vmin) / (vmax - vmin + 1e-8)
        target = (target - vmin) / (vmax - vmin + 1e-8)

        if self.seg_paths is not None:
            source_seg = tiff.imread(self.seg_path_list[idx1])
            target_seg = tiff.imread(self.seg_path_list[idx2])
            source_seg = interpole_and_pad_mask(source_seg, SHAPE)
            target_seg = interpole_and_pad_mask(target_seg, SHAPE)
            # For testing, we don't need further processing
            if self.mode == "train" or self.mode == "train_in_order":
                if self.seg_type == "binary" or self.seg_type is None:
                    source_seg = source_seg > 0
                    target_seg = target_seg > 0
                # elif self.seg_type == "labels":
                # target_seg_oh = (
                #     torch.nn.functional.one_hot(
                #         target_seg.long(), num_classes=NB_CLASSES
                #     )
                #     .permute(0, 3, 1, 2)
                #     .float()
                # )
                # source_seg_oh = (
                #     torch.nn.functional.one_hot(
                #         source_seg.long(), num_classes=NB_CLASSES
                #     )
                #     .permute(0, 3, 1, 2)
                #     .float()
                # )
                # Not enough memory to use labels (and one hot)
                source_seg = torch.from_numpy(source_seg).float()
                target_seg = torch.from_numpy(target_seg).float()
            return {
                "source": source,
                "target": target,
                "source_seg": source_seg,
                "target_seg": target_seg,
                "idx1": idx1,
                "idx2": idx2,
            }

        return {"source": source, "target": target}

    def get_transforms(self, n):
        """Get global min and global max from n frames and return normalisation and transformation functions."""
        global_min = np.min(tiff.imread(str(self.frames_path_list[0])))
        global_max = np.max(tiff.imread(str(self.frames_path_list[0])))

        for i in range(1, n):
            current_min = np.min(tiff.imread(str(self.frames_path_list[i])))
            current_max = np.max(tiff.imread(str(self.frames_path_list[i])))

            global_min = min(global_min, current_min)
            global_max = max(global_max, current_max)

        base_transforms = [NormalizeMinMax(global_min=global_min, global_max=global_max)]

        return v2.Compose(base_transforms)

    def compute_video_min_max(self):
        """Compute min max of each video independently"""
        self.video_min = []
        self.video_max = []
        for k in range(len(self.limits) - 1):
            start_idx = self.limits[k]
            end_idx = self.limits[k + 1]

            global_min = np.min(tiff.imread(str(self.frames_path_list[start_idx])))
            global_max = np.max(tiff.imread(str(self.frames_path_list[start_idx])))

            for i in range(start_idx, end_idx):
                img = tiff.imread(str(self.frames_path_list[i]))

                global_min = min(global_min, np.min(img))
                global_max = max(global_max, np.max(img))

            self.video_min.append(global_min)
            self.video_max.append(global_max)

    def __len__(self):
        """Return nbr of paires to train on per epoch."""
        return self.nb_pairs

    def get_path_list(self):
        """Create a list with all the frame's path and segmentation's path."""
        self.frames_path_list = []
        self.seg_path_list = []
        self.limits = [0]  # list to store the limits of each video in the frames_path_list
        for i, video_path in enumerate(self.video_paths):
            video_path = Path(video_path)
            self.frames_path_list += sorted(video_path.iterdir())
            self.limits.append(len(self.frames_path_list))  # store the limit of the current video
            if self.seg_paths is not None:
                # Get the corresponding segmentation path of the video i and create a list of all the segmentation frames
                seg_path = Path(self.seg_paths[i])
                if self.seg_type == "heatmaps":
                    seg_path = seg_path.parent / "SEG_HEATMAP"
                self.seg_path_list += sorted(seg_path.iterdir())

                if len(self.frames_path_list) != len(self.seg_path_list):
                    raise ValueError("Synchronisation problem between video and segmentation")
