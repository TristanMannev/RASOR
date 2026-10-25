"""Script to test registration of frames of a video with different models.

Adapted from :
VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
pp 1788-1800. 2019.
"""

# Core library imports
import argparse
import datetime
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tifffile as tiff
import torch

# Local imports
from rasor.flow_visualizer import FlowVisualizer
from rasor.model import WarperVxm
from rasor.my_utils import configs, image_utils
from rasor.my_utils.spatial_transform import SpatialTransformer
from rasor.register_cells_old import (
    register_pair,
    register_pair_fb,
    register_pair_raft,
    register_pair_tvl1,
)

# for reproductability
torch.manual_seed(configs.SEED)
np.random.seed(configs.SEED)
torch.cuda.manual_seed(configs.SEED)
torch.cuda.manual_seed_all(configs.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate_registration_with_seg(segmentation_moving, segmentation_fixed, displacement, fixed):
    """Evaluate the registration results."""
    score = {}
    # Load segmentation masks
    seg_moving = image_utils.interpole_and_pad_mask(segmentation_moving, configs.SHAPE)
    seg_fixed = image_utils.interpole_and_pad_mask(segmentation_fixed, configs.SHAPE)

    # Warp moving segmentation with the predicted displacement
    tensor_seg_moving = torch.from_numpy(seg_moving).to(configs.device).float().unsqueeze(0).unsqueeze(0)
    spatial_transformer = SpatialTransformer(interpolation_mode="nearest").to(configs.device)
    warped_seg_moving = spatial_transformer(tensor_seg_moving, displacement)
    warped_seg_moving_np = warped_seg_moving.squeeze().cpu().numpy()

    # Compute scores
    # Take only pairs of frame with same nbr of cells (to have max dice = 1 in case of disappaering cell in vid)
    moving_labels = set(np.unique([seg_moving]))
    fixed_labels = set(np.unique([seg_fixed]))
    if moving_labels != fixed_labels:
        score = None
    else:
        # Compute Dice score only for pairs of frames with same number of cells
        dice = image_utils.dice(warped_seg_moving_np, seg_fixed, include_zero=False)
        ref_dice = image_utils.dice(
            seg_moving, seg_fixed, include_zero=False
        )  # reference dice to compare with the dice after registration
        score["dice"] = np.mean(dice)
        score["ref_dice"] = np.mean(ref_dice)
        # Compute IoU score only for pairs of frames with same number of cells
        iou_score = image_utils.iou(warped_seg_moving_np, seg_fixed, include_zero=False)
        ref_iou_score = image_utils.iou(
            seg_moving, seg_fixed, include_zero=False
        )  # reference iou to compare with the iou after registration
        score["iou"] = np.mean(iou_score)
        score["ref_iou"] = np.mean(ref_iou_score)

    # To visualise seg evaluation
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    plt.subplot(2, 2, 1)
    ax[0, 0].set_title("Fixed image with moving (red) and fixed (green) segmentation contours")
    ax[0, 0].imshow(fixed, cmap="gray")
    image_utils.plot_contours(seg_moving, color="red", ax=ax[0, 0])
    image_utils.plot_contours(seg_fixed, color="chartreuse", ax=ax[0, 0])
    ax[0, 1].set_title("Warped moving segmentation")
    ax[0, 1].imshow(warped_seg_moving_np, cmap="gray")
    plt.subplot(2, 2, 3)
    ax[1, 0].set_title("Fixed segmentation - Warped segmentation")
    ax[1, 0].imshow(np.abs(seg_fixed - warped_seg_moving_np), cmap="gray")
    plt.subplot(2, 2, 4)
    ax[1, 1].set_title("Green : Fixed segmentation, Magenta : Warped moving segmentation")
    black_background = np.zeros(fixed.shape)
    ax[1, 1].imshow(black_background, cmap="gray")
    image_utils.plot_contours(seg_fixed, color="chartreuse", ax=ax[1, 1])
    image_utils.plot_contours(warped_seg_moving_np, color="magenta", ax=ax[1, 1])
    return score, fig


def evaluate_registration(moving, fixed, displacement):
    """Evaluate the registration results with MSE (ideally find another method to evaluate the visual aspect)."""
    score = {}
    moving = image_utils.interpole_and_pad(moving, configs.SHAPE)
    fixed = image_utils.interpole_and_pad(fixed, configs.SHAPE)
    # Warp moving with the predicted displacement
    tensor_moving = torch.from_numpy(moving).to(configs.device).float().unsqueeze(0).unsqueeze(0)
    spatial_transformer = SpatialTransformer(interpolation_mode="linear").to(configs.device)
    warped_moving = spatial_transformer(tensor_moving, displacement)
    # Compute RMSE score
    rmse = np.sqrt(np.mean((fixed - warped_moving.squeeze().cpu().numpy()) ** 2))
    ref_rmse = np.sqrt(np.mean((fixed - moving) ** 2))  # reference dice to compare with the mse after registration
    score["rmse"] = rmse
    score["ref_rmse"] = ref_rmse
    return score


def evaluate_segmentation(segmentation_pred, segmentation_gt, img_moving):
    """Evaluate the segmentation results."""
    score = {}
    # Load segmentation masks
    seg_pred = image_utils.interpole_and_pad_mask(segmentation_pred, configs.SHAPE)
    seg_gt = image_utils.interpole_and_pad_mask(segmentation_gt, configs.SHAPE)
    # Binarise
    seg_gt = seg_gt > 0
    seg_pred = (seg_pred > 0.5).astype(int)
    # Compute Dice score
    dice = image_utils.dice(seg_pred, seg_gt, include_zero=False)
    score["dice_segmentation"] = np.mean(dice)
    # To visulaise seg evaluation
    fig = plt.figure(figsize=(6, 6))
    plt.title("Segmentation prediction vs GT")
    plt.imshow(img_moving, cmap="gray")
    image_utils.plot_contours(seg_pred, "magenta")
    image_utils.plot_contours(seg_gt, "chartreuse")
    plt.axis("off")
    return score, fig


def create_model(args):
    """Create the model for registration."""
    nb_features = [16, 32, 64, 128, 256]
    if args.model_type in {"voxelmorph", "rasor-cnn", "rasor", "twoflows", "flowseg", "transflowseg", "transmorph2d"}:
        model = WarperVxm(
            ndim=2,
            source_channels=1,
            target_channels=1,
            nb_features=nb_features,
            model_type=args.model_type,
            device=configs.device,
        ).to(configs.device)
        model_path = f"{args.output_dir}/best.pt"
        model.load_state_dict(torch.load(model_path, map_location=configs.device))
        print(f"Load model : {model_path}")  # noqa: T201
        model.eval()
        if mlflow.active_run():
            mlflow.log_param("model_path", model_path)
    elif args.model_type == "tvl1":
        model = None
    elif args.model_type == "farnback":
        model = cv2.FarnebackOpticalFlow.create()
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")
    return model


def register_video(args):
    """Register video frames.

    Parameters
    ----------
    args : argparse.Namespace
        Commandline arguments.
    """
    print(f"Using device: {configs.device}")  # noqa: T201
    timestamp = datetime.datetime.now(tz=datetime.UTC)

    # Load video and segmentations frames
    frames_path_list = sorted(Path(args.video_path).iterdir())
    seg_path_list = (sorted(Path(args.seg_path).iterdir())) if args.seg_path is not None else None
    global_min, global_max = image_utils.normalisation_params(frames_path_list)

    # MLFLOW
    if args.model_type in {"voxelmorph", "twoflows", "rasor-cnn", "rasor", "flowseg", "transflowseg", "transmorph2d"}:
        with Path(f"{args.output_dir}/config.json").open() as j:
            config = json.load(j)  # To get informations from vxm training
    else:
        config = {}
        args.output_dir = f"outputs/{args.model_type}/{args.video_path.split('/')[-2]}"

    if args.mlflow:
        mlflow.set_tracking_uri(configs.MLFLOW_TRACKING_URI)
        parent_run_id = config.get("mlflow_run_id")
        mlflow.start_run(
            run_name=f"Test gap {args.gap} model type {args.model_type} on {args.video_path}",
            nested=True,
            parent_run_id=parent_run_id,  # To link the test run to the training run
        )
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

    # Create model
    model = create_model(args)

    # Create results directory if it doesn't exist
    results_dir = Path(f"{args.output_dir}/results_gap{args.gap}")
    results_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = results_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)
    results = []
    flow_maps = {}
    video = []  # list of frames to visualize the flow maps on (only for gap=1 and if args.visualize is True)
    for idx in range(min((len(frames_path_list) - args.gap), 100)):  # Limit to 100 pairs for testing
        result = {}
        # Load moving and fixed images
        idx2 = idx + args.gap
        print(f"Registering pair : {idx} and {idx2}")  # noqa: T201
        # Read images
        moving = tiff.imread(frames_path_list[idx])
        fixed = tiff.imread(frames_path_list[idx2])
        # Interpole/Pad images to HEIGHT, WIDTH if needed
        moving = image_utils.interpole_and_pad(moving, configs.SHAPE)
        fixed = image_utils.interpole_and_pad(fixed, configs.SHAPE)
        # Normalise
        moving = image_utils.normalise(moving, global_min, global_max)
        fixed = image_utils.normalise(fixed, global_min, global_max)
        # 3. Compute registration
        if args.model_type in {"voxelmorph", "rasor-cnn", "rasor", "twoflows", "flowseg", "transflowseg", "transmorph2d"}:
            res = register_pair(moving, fixed, model)
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
            if "displacement_bwd" in res:
                displacement_bwd = res["displacement_bwd"]
            if "source_segmentation" in res:
                source_seg_pred = res["source_segmentation"]
        elif args.model_type == "tvl1":
            res = register_pair_tvl1(moving, fixed, global_min, global_max)
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
        elif args.model_type == "raft":
            res = register_pair_raft(
                frames_path_list[idx],
                frames_path_list[idx2],
                model,
                global_min,
                global_max,
            )
            displacement, fig, input_moving, input_fixed = (
                res["displacement"],
                res["fig"],
                res["input_moving"],
                res["input_fixed"],
            )
        elif args.model_type == "farnback":
            displacement, fig, input_moving, input_fixed = register_pair_fb(
                moving, fixed, model, global_min, global_max
            )
        fig.savefig(viz_dir / f"registration_result_for_paire_{idx}-{idx2}.png")
        plt.close("all")

        if args.visualize or args.save_flow_maps:
            flow_maps[(idx2, idx)] = displacement.squeeze().cpu().numpy()
            if idx == 0:
                video.append(np.expand_dims(input_moving, axis=-1))  # channel dimension needed for visualization
            video.append(np.expand_dims(input_fixed, axis=-1))  # channel dimension neededfor visualization
            if args.model_type in {"voxelmorph", "rasor-cnn", "rasor", "flowseg", "transflowseg", "transmorph2d"}:
                res = register_pair(fixed, moving, model)
                displacement_bwd = res["displacement"]
            elif args.model_type == "tvl1":
                res = register_pair_tvl1(fixed, moving, global_min, global_max)
                displacement_bwd = res["displacement"]
            elif args.model_type == "raft":
                res = register_pair_raft(
                    frames_path_list[idx2],
                    frames_path_list[idx],
                    model,
                    global_min,
                    global_max,
                )
                displacement_bwd = res["displacement"]
            elif args.model_type == "farnback":
                displacement_bwd, _, _, _ = register_pair_fb(fixed, moving, model, global_min, global_max)
            flow_maps[(idx, idx2)] = displacement_bwd.squeeze().cpu().numpy()

        if mlflow.active_run():
            mlflow.log_figure(fig, artifact_file=f"viz/registration_result_for_paire_{idx}-{idx2}.png")

        # Evaluation for the pair
        eval_res = evaluate_registration(moving, fixed, displacement)
        score_pair = eval_res
        result.update(score_pair)
        # If segmentations available, load them and warp the moving seg with the predicted displacement for evaluation
        if args.seg_path is not None:
            # Load segmentation masks
            seg_moving = tiff.imread(seg_path_list[idx])
            seg_fixed = tiff.imread(seg_path_list[idx2])
            eval_res = evaluate_registration_with_seg(seg_moving, seg_fixed, displacement, fixed)
            score_pair, fig = eval_res
            fig.savefig(viz_dir / f"seg_eval_for_paire_{idx}-{idx2}.png")
            if mlflow.active_run():
                mlflow.log_figure(fig, artifact_file=f"seg_eval_for_paire_{idx}-{idx2}.png")
            plt.close("all")

            if score_pair is not None:
                result.update(score_pair)
            # ------------------------- SEGMENTATION EVALUATION ------------------------------------
            if "source_segmentation" in res:
                eval_seg_res = evaluate_segmentation(source_seg_pred.squeeze().cpu().numpy(), seg_moving, moving)
                if eval_seg_res is None:
                    continue
                score_pair, fig = eval_seg_res
                result.update(score_pair)
                fig.savefig(viz_dir / f"segmentation_result_for_{idx}.png")
                if mlflow.active_run():
                    mlflow.log_figure(fig, artifact_file=f"segmentation_result_for_{idx}.png")
                plt.close("all")
        results.append(result)

    metrics = {
        "data": args.video_path,
        "date": timestamp.strftime("%Y%m%d_%H%M%S"),
        "model": args.output_dir,
        "device": str(configs.device),
        "gap": args.gap,
    }
    # Global evaluation
    # Compute mse to evaluate the global visual aspect
    rmse_per_pair = [result["rmse"] for result in results if "rmse" in result]
    ref_rmse_scores_per_pair = [result["ref_rmse"] for result in results if result is not None]
    global_rmse = np.mean(rmse_per_pair)
    global_ref_rmse = np.mean(ref_rmse_scores_per_pair)
    metrics["global_rmse"] = global_rmse
    metrics["global_ref_rmse"] = global_ref_rmse
    # Compute de global dice score an iou score
    if args.seg_path is not None:
        dice_per_pair = [result["dice"] for result in results if "dice" in result]
        ref_dice_scores_per_pair = [result["ref_dice"] for result in results if "ref_dice" in result]
        iou_scores_per_pair = [result["iou"] for result in results if "iou" in result]
        ref_iou_scores_per_pair = [result["ref_iou"] for result in results if "ref_iou" in result]
        global_dice = np.mean(dice_per_pair)
        global_ref_dice = np.mean(ref_dice_scores_per_pair)
        global_iou = np.mean(iou_scores_per_pair)
        global_ref_iou = np.mean(ref_iou_scores_per_pair)
        if "dice_segmentation" in results[0]:
            dice_segmentation_per_pair = [
                result["dice_segmentation"] for result in results if "dice_segmentation" in result
            ]
            global_dice_segmentation = np.mean(dice_segmentation_per_pair)
        else:
            global_dice_segmentation = None
        # Save metrics in a json file
        metrics["global_dice"] = global_dice
        metrics["global_ref_dice"] = global_ref_dice
        metrics["global_iou"] = global_iou
        metrics["global_ref_iou"] = global_ref_iou
        metrics["global_rmse"] = float(global_rmse)
        metrics["global_ref_rmse"] = float(global_ref_rmse)
        metrics["dice_per_pair"] = dice_per_pair
        metrics["ref_dice_per_pair"] = ref_dice_scores_per_pair
        metrics["iou_per_pair"] = iou_scores_per_pair
        metrics["ref_iou_per_pair"] = ref_iou_scores_per_pair
        metrics["global_dice_segmentation"] = global_dice_segmentation
        if mlflow.active_run():
            mlflow.log_metrics(
                {
                    "global_dice": global_dice,
                    "global_ref_dice": global_ref_dice,
                    "global_iou": global_iou,
                    "global_ref_iou": global_ref_iou,
                    "global_rmse": global_rmse,
                    "global_ref_rmse": global_ref_rmse,
                }
            )
    with Path.open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    if args.mlflow:
        mlflow.end_run()

    if args.save_flow_maps:
        flow_dir = results_dir / "flowmaps"
        flow_dir.mkdir(parents=True, exist_ok=True)
        for name, value in flow_maps.items():
            np.save(flow_dir / f"{name}.npy", value)

    if args.visualize:
        visualizer = FlowVisualizer(video, flow_maps, grid_step=10, gap=args.gap, grid_type="uniform")
        visualizer.run()

    end_timestamp = datetime.datetime.now(tz=datetime.UTC)
    exec_time = (end_timestamp - timestamp).seconds

    # Relativ Error reduction
    rer_dice = (global_dice - global_ref_dice) / (1 - global_ref_dice) if (global_ref_dice != 1) else 0

    rer_rmse = (global_ref_rmse - global_rmse) / (global_ref_rmse) if (global_ref_rmse != 0) else 0

    return rer_dice, rer_rmse, exec_time


if __name__ == "__main__":
    # Parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlflow", action="store_true", help="Enable mlflow tracking during training")
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of flow maps",
    )
    parser.add_argument(
        "--video_path",
        default=configs.TEST_DATA_U373,
        help="Moving image (source) filename",
    )
    parser.add_argument("--gap", default=1, help="Temporal gap between 2 frames")
    parser.add_argument(
        "--seg_path",
        default=configs.TEST_SEG_DATA_U373,
        help="Segmentations for evaluation (if segmentations not available, set to None)",
    )
    parser.add_argument(
        "--ndim",
        default=2,
        help="Number of dimensions of the images (2D or 3D)",
    )
    parser.add_argument(
        "--model_type",
        default="voxelmorph",
        choices=[
            "voxelmorph",
            "twoflows",
            "rasor-cnn",
            "rasor",
            "flowseg",
            "transflowseg",
            "transmorph2d",
            "tvl1",
            "raft",
            "farnback",
        ],
        help="Type of model to use for registration",
    )
    parser.add_argument(
        "--output_dir",
        default="outputs/rasor-cnn/output_20260904_142224",
        help="Directory to find model and to save results",
    )
    parser.add_argument(
        "--save_flow_maps",
        action="store_true",
        help="Save flow maps",
    )
    args = parser.parse_args()

    _ = register_video(args)
