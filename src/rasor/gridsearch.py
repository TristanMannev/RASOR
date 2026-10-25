"""Grid search for tvl1."""

# Core library imports
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile as tiff

# Third-party imports
import torch
from skimage.registration import optical_flow_tvl1
from tqdm import tqdm

from deepfluxoptique.my_utils import image_utils, spatial_transform
from deepfluxoptique.my_utils.configs import (
    HEIGHT,
    TRAIN_DATA,
    TRAIN_SEG_DATA,
    WIDTH,
    device,
)


def evaluate_registration(segmentation_moving, segmentation_fixed, field):
    """Evaluate the registration results."""
    # Load segmentation masks
    seg_moving = image_utils.interpole_and_pad(segmentation_moving, [HEIGHT, WIDTH])
    seg_fixed = image_utils.interpole_and_pad(segmentation_fixed, [HEIGHT, WIDTH])
    # Take only pairs of frame with same nbr of cells (to have max dice = 1 in case of disappaering cell in vid)
    moving_labels = set(np.unique([seg_moving]))
    fixed_labels = set(np.unique([seg_fixed]))
    if moving_labels != fixed_labels:
        return None
    # Warp moving segmentation with the predicted field
    tensor_seg_moving = torch.from_numpy(seg_moving).to(device).float().unsqueeze(0).unsqueeze(0)
    spatial_transformer = spatial_transform.SpatialTransformer(interpolation_mode="nearest").to(device)
    warped_seg_moving = spatial_transformer(tensor_seg_moving, field)
    # Compute Dice score
    dice = image_utils.dice(warped_seg_moving.squeeze().cpu().numpy(), seg_fixed, include_zero=False)

    return np.mean(dice)


def gridsearch(args):
    """Perform gridsearch for TVL1 model.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {device}")

    # 1. Load video and segmentations frames
    frames_path_list = sorted(Path(args.video_path).iterdir())
    seg_path_list = (sorted(Path(args.seg_path).iterdir())) if args.seg_path is not None else None
    global_min, global_max = image_utils.normalisation_params(frames_path_list)

    # Create output directory if it doesn't exist
    results_dir = Path("outputs/tvl1")
    results_dir.mkdir(parents=True, exist_ok=True)

    # Param grid search
    attachement_param = [5, 10, 15, 20]
    tightness = [0.1, 0.3, 0.5]
    num_warp = [1, 3, 5]

    best_dice = -1
    best_params = None

    for attachement in attachement_param:
        for tight in tightness:
            for num in num_warp:
                print(f"Running grid search with attachement={attachement}, tightness={tight}, num_warp={num}")
                args.attachement = attachement
                args.tightness = tight
                args.num_warp = num

                dice_per_pair = []

                for idx in tqdm(range(len(frames_path_list) - args.gap), desc="Registration..."):
                    # Load moving and fixed images
                    idx2 = idx + args.gap
                    moving = tiff.imread(frames_path_list[idx])
                    fixed = tiff.imread(frames_path_list[idx2])

                    # Compute registration
                    # Interpole/Pad images to HEIGHT, WIDTH if needed
                    moving_resized = image_utils.interpole_and_pad(moving, [HEIGHT, WIDTH])
                    fixed_resized = image_utils.interpole_and_pad(fixed, [HEIGHT, WIDTH])

                    # Normalisation
                    input_moving = image_utils.normalise(moving_resized, global_min, global_max)
                    input_fixed = image_utils.normalise(fixed_resized, global_min, global_max)

                    # Predict
                    field = optical_flow_tvl1(
                        reference_image=input_fixed,
                        moving_image=input_moving,
                        attachment=attachement,
                        tightness=tight,
                        num_warp=num,
                    )
                    tensor_field = torch.from_numpy(field).float().to(device).unsqueeze(0)

                    # Evaluation for the pair
                    # Load segmentation masks
                    seg_moving = tiff.imread(seg_path_list[idx])
                    seg_fixed = tiff.imread(seg_path_list[idx2])
                    dice = evaluate_registration(seg_moving, seg_fixed, tensor_field)
                    if dice is None:
                        continue
                    dice_per_pair.append(dice)

                # 5. Global evaluation
                # Compute de global dice score an iou score
                global_dice = np.mean(dice_per_pair)

                if global_dice > best_dice:
                    best_dice = global_dice
                    best_params = (attachement, tight, num)

    with Path.open(results_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)


if __name__ == "__main__":
    # Parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default=f"{TRAIN_DATA}", help="Moving image (source) filename")
    parser.add_argument(
        "--seg_path",
        default=TRAIN_SEG_DATA,
        help="Segmentations",
    )
    parser.add_argument("--gap", default=10, help="Temporal gap between 2 frames")
    args = parser.parse_args()

    gridsearch(args)
