#!/usr/bin/env python
"""Train RASOR, VoxelMorph, and related registration models.

Adapted from :
VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
pp 1788-1800. 2019.
"""  # noqa: CPY001

# Core library imports
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path

os.environ["NEURITE_BACKEND"] = "pytorch"

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import mlflow
import neurite as ne
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
from rasor.dataset import VxmMultipleVideoDataset, VxmVideoDataset
from rasor.model import WarperVxm
from rasor.my_utils import configs, image_utils, spatial_transform

if TYPE_CHECKING:
    from collections.abc import Sequence

# for reproductability
torch.manual_seed(configs.SEED)
np.random.seed(configs.SEED)  # noqa: NPY002
torch.cuda.manual_seed(configs.SEED)
torch.cuda.manual_seed_all(configs.SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class LossComputer(nn.Module):
    """Compute different type of loss."""

    def __init__(self, image_loss_fn, grad_loss_fn, flow_with_seg_loss_fn, seg_loss_fn):
        super().__init__()
        self.image_loss_fn = image_loss_fn
        self.flow_with_seg_loss_fn = flow_with_seg_loss_fn
        self.grad_loss_fn = grad_loss_fn
        self.seg_loss_fn = seg_loss_fn

    def compute_image_loss(self, img1, img2):
        """Compute the image loss to compare intensity between two images."""
        if isinstance(self.image_loss_fn, ne.nn.modules.NCC):
            return (1 - self.image_loss_fn(img1, img2)).mean()  # to minimize ncc loss
        return self.image_loss_fn(img1, img2)

    def compute_flow_with_seg_loss(self, img1, img2):
        """Compute the loss to compare alignments of segmentations between two images."""
        if isinstance(self.flow_with_seg_loss_fn, ne.nn.modules.Dice):
            return 1 - self.flow_with_seg_loss_fn(img1, img2)  # to minimize dice loss
        return self.flow_with_seg_loss_fn(img1, img2)

    def compute_grad_loss(self, displacement):
        """Compute the loss to penalize local spatial variations in the displacement field."""
        return self.grad_loss_fn(displacement)

    def compute_cycle_loss(self, displacement, warped_displacement_bwd):
        """Compute the cycle loss to enforce cycle consistency between forward and backward displacement fields."""
        return torch.norm(displacement + warped_displacement_bwd, p=2, dim=1).mean()

    def compute_background_loss(self, source_seg, target_seg, displacement):
        """Compute the loss to avoid displacement in the background."""
        union_seg = torch.clamp(target_seg + source_seg, 0.0, 1.0)
        background_mask = 1.0 - union_seg
        return torch.mean(torch.abs(displacement) * background_mask)  # L1 loss on the displacement in the background

    def compute_seg_loss(self, seg_pred, seg_gt):
        """Compute the loss to compare predicted segmentations and ground truth."""
        return self.seg_loss_fn(seg_pred, seg_gt)


def train_epoch(  # noqa: PLR0913, PLR0917
    epoch: int,
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_computer: nn.Module,
    loss_weights: Sequence[float],
    background: bool,
    cycle_consistency: bool = False,
    seg_type: str = "binary",
) -> float:
    """Train for one epoch.

    Parameters
    ----------
    model : nn.Module
        The VoxelMorph model to train.
    dataloader : torch.utils.data.DataLoader
        The dataloader to use for training.
    optimizer : torch.optim.Optimizer
        The optimizer to use for training.
    loss_computer : LossComputer
        Respnsible of computing loss functions
    loss_weights : Sequence[float]
        The weights for the different losses.
    background : Bool
        If True, add a loss to avoid displacement in the background.
    cycle_consistency : Bool
        If True, compute backward displpacement and corresponding losses.
    seg_type : str
        Segmentation type to use.
    """
    model.train()
    total_loss = 0.0
    epoch_img_loss = 0.0
    epoch_grad_loss = 0.0
    epoch_flow_with_seg_loss = 0.0
    epoch_seg_loss = 0.0
    epoch_background_loss = 0.0
    epoch_cycle_loss = 0.0

    for batch in tqdm(dataloader, total=len(dataloader), desc="Training"):
        optimizer.zero_grad()

        # Define warper for later use #TODO : mettre or de cette fonoction ou créer classe
        spatial_transformer = spatial_transform.SpatialTransformer(interpolation_mode="linear").to(configs.device)

        source = batch["source"]
        target = batch["target"]
        source = source.to(configs.device).unsqueeze(1)  # Add channel dimension
        target = target.to(configs.device).unsqueeze(1)  # Add channel dimension

        # Loss initialisation
        loss = 0.0
        img_loss = 0.0
        grad_loss = 0.0
        background_loss = 0.0

        # Get the displacement and the warped source image from the model
        res = model(source, target, return_warped_source=True)
        displacement = res["displacement"]
        warped_source = res["warped_source"]

        if cycle_consistency and "displacement_bwd" not in res:
            res_bwd = model(target, source, return_warped_source=True)
            res["displacement_bwd"] = res_bwd["displacement"]

        # Compute image loss and grad loss for displacement forward
        img_loss += loss_computer.compute_image_loss(target, warped_source)
        grad_loss += loss_computer.compute_grad_loss(displacement)  # Smoothness

        # ------------------------------------------- TWO FLOWS / CYCLE CONSISTENCY -----------------------------------
        # If model_type = twoflows, the backward displacement (displacement_bwd) is computed as well
        # if cycle_consistency is True, the backward displacement is computed by inversing source and target
        # Add the corresponding loss (image loss and grad loss for bwd displacement + cycle loss)
        if "displacement_bwd" in res:
            displacement_bwd = res["displacement_bwd"]
            warped_target = spatial_transformer(target, displacement_bwd)

            # Image loss for backward displacement
            img_loss += loss_computer.compute_image_loss(source, warped_target)

            # Grad loss for backward displacement
            grad_loss += loss_computer.compute_grad_loss(displacement_bwd)

            # Cycle loss
            cycle_loss = 0.0
            warped_displacement_bwd = spatial_transformer(displacement_bwd, displacement)
            cycle_loss += loss_computer.compute_cycle_loss(displacement, warped_displacement_bwd)
            # Add a progressiv weight to loss (start at 0, max at loss_weights[4])
            cycle_weight = image_utils.linear_augment(start=5, end=15, max_weight=loss_weights[4], epoch=epoch)
            loss += cycle_weight * cycle_loss
            epoch_cycle_loss += cycle_loss.item()
        # ---------------------------------------------------------------------------------------------------------

        # Add final img_loss and final grad_loss to the total loss
        loss += loss_weights[0] * img_loss + loss_weights[1] * grad_loss

        # If segmentations are available, compute segmentation loss
        if "source_seg" in batch and "target_seg" in batch:
            source_seg = batch["source_seg"].to(configs.device).unsqueeze(1)  # add channel dim
            target_seg = batch["target_seg"].to(configs.device).unsqueeze(1)  # add channel dim
            flow_with_seg_loss = 0.0
            warped_source_seg = spatial_transformer(source_seg, displacement)
            warped_source_seg = torch.clamp(warped_source_seg, 0, 1)
            flow_with_seg_loss += loss_computer.compute_flow_with_seg_loss(target_seg, warped_source_seg).squeeze()
            # ------------------------------------------- TWO FLOWS / CYCLE CONSISTENCY -------------------------------
            if "displacement_bwd" in res:
                warped_target_seg = spatial_transformer(target_seg, displacement_bwd)
                warped_target_seg = torch.clamp(warped_target_seg, 0, 1)
                flow_with_seg_loss += loss_computer.compute_flow_with_seg_loss(source_seg, warped_target_seg).squeeze()
                if background:
                    # Add loss to avoid displacement in the background
                    background_loss += loss_computer.compute_background_loss(source_seg, target_seg, displacement_bwd)
                    loss += loss_weights[3] * background_loss
                    epoch_background_loss += background_loss.item()
            # ------------------------------------------- FLOW SEG ---------------------------------------------------
            # RASOR-CNN and RASOR also predict a segmentation map.
            # Add the corresponding loss
            if "source_segmentation" in res:
                seg_loss = 0.0
                source_seg_pred = res["source_segmentation"]
                warped_source_seg_pred = spatial_transformer(source_seg_pred, displacement)

                seg_loss += loss_computer.compute_seg_loss(source_seg_pred, source_seg)
                seg_loss += loss_computer.compute_seg_loss(warped_source_seg_pred, target_seg)

                loss += loss_weights[5] * seg_loss
                epoch_seg_loss += seg_loss.item()
            # ---------------------------------------------------------------------------------------------------------
            if background:
                # Add loss to avoid displacement in the background
                background_loss += loss_computer.compute_background_loss(source_seg, target_seg, displacement)
                loss += loss_weights[3] * background_loss
                epoch_background_loss += background_loss.item()
            if seg_type is not None:
                loss += loss_weights[2] * flow_with_seg_loss
                epoch_flow_with_seg_loss += flow_with_seg_loss.item()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        epoch_img_loss += img_loss.item()
        epoch_grad_loss += grad_loss.item()

    return {
        "loss": total_loss / len(dataloader),
        "img_loss": epoch_img_loss / len(dataloader),
        "grad_loss": epoch_grad_loss / len(dataloader),
        "seg_loss": epoch_seg_loss / len(dataloader) if epoch_seg_loss > 0 else None,
        "flow_with_seg_loss": epoch_flow_with_seg_loss / len(dataloader) if epoch_flow_with_seg_loss > 0 else None,
        "background_loss": epoch_background_loss / len(dataloader) if epoch_background_loss > 0 else None,
        "cycle_loss": epoch_cycle_loss / len(dataloader) if epoch_cycle_loss > 0 else None,
    }


def train(args):  # noqa: C901, PLR0912, PLR0915
    """Create the dataloader, create the model, train and save.

    Returns:
    -----------
    output_dir :
        Relative path where checkpoints, config, best and final model are saved.
    exec_time :
        Training time.
    """
    # Set device
    print(f"Using device: {configs.device}")  # noqa: T201
    timestamp = datetime.datetime.now(tz=datetime.UTC)
    if args.seg_path is None:
        args.seg_type = None
        args.gamma_param = 0.0
    # MLflow set up
    if args.mlflow:
        mlflow.set_tracking_uri(configs.MLFLOW_TRACKING_URI)
        mlflow.start_run(
            run_name=f"train_cells_{args.model_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{args.video_path}"
        )
        print("MLflow tracking enabled.")  # noqa: T201
        mlflow.log_params(vars(args))

    # Create model
    print(f"Load model type: {args.model_type}")  # noqa: T201
    print(f"Using segmentation type: {args.seg_type}")  # noqa: T201

    nb_features = [16, 32, 64, 128, 256]
    model = WarperVxm(
        ndim=2,
        source_channels=1,
        target_channels=1,
        nb_features=nb_features,
        model_type=args.model_type,
        device=configs.device,
    ).to(configs.device)

    # Setup losses and optimizer
    if args.image_loss == "mse":
        image_loss_fn = ne.nn.modules.MSE()
    elif args.image_loss == "cc":
        image_loss_fn = ne.nn.modules.NCC(window_size=4)
    else:
        raise ValueError(f"Unsupported image loss: {args.image_loss} use 'mse' or 'cc'.")
    grad_loss_fn = ne.nn.modules.SpatialGradient(args.grad_loss)
    # use the same loss as for the image (MSE or CC) for heatmaps
    flow_with_seg_loss_fn = image_loss_fn if args.seg_type == "heatmaps" else ne.nn.modules.Dice()
    seg_loss_fn = nn.BCEWithLogitsLoss()

    # Reguralisation parameters for each loss
    loss_weights = [
        1.0,
        args.lambda_param,
        args.gamma_param,
        args.background_param,
        args.cycle_param,
        args.seg_param,
    ]
    loss_computer = LossComputer(image_loss_fn, grad_loss_fn, flow_with_seg_loss_fn, seg_loss_fn)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Create dataloader
    train_dataset = VxmVideoDataset(
        video_path=args.video_path,
        seg_path=args.seg_path,
        seg_type=args.seg_type,
        mode="train",
        gap=args.max_gap,
        multiplier=args.multiplier,
    )
    if args.multiple_videos:  # TODO : add args to choose videos
        train_dataset = VxmMultipleVideoDataset(
            video_paths=[
                configs.TRAIN_DATA_MUSC,
                configs.TRAIN_DATA_HELA,
                configs.TRAIN_DATA_GOWT,
                configs.TRAIN_DATA_U373,
            ],
            seg_paths=[
                configs.TRAIN_SEG_DATA_MUSC,
                configs.TRAIN_SEG_DATA_HELA,
                configs.TRAIN_SEG_DATA_GOWT,
                configs.TRAIN_SEG_DATA_U373,
            ],
            seg_type=args.seg_type,
            mode="train",
            gap=args.max_gap,
            multiplier=args.multiplier,
        )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,  # To speed up, if not enough RAM, set to False
    )

    # Create output directory
    output_dir = f"{args.output_dir}/{args.model_type}/output_{timestamp.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    # Training loop
    print(f"Training for {args.epochs} epochs...")  # noqa: T201
    best_loss = float("inf")
    train_loss = []
    for epoch in tqdm(range(args.epochs), desc="Epochs"):
        # Train for one epoch
        avg_loss = train_epoch(
            epoch=epoch,
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_computer=loss_computer,
            loss_weights=loss_weights,
            background=args.background,
            cycle_consistency=args.cycle_consistency,
            seg_type=args.seg_type,
        )
        train_loss.append(avg_loss["loss"])
        if mlflow.active_run():
            mlflow.log_metric("Train_loss", avg_loss["loss"], step=epoch)
            mlflow.log_metric("Train_img_loss", avg_loss["img_loss"], step=epoch)
            mlflow.log_metric(
                f"Train_grad_loss lambda:{args.lambda_param}",
                avg_loss["grad_loss"],
                step=epoch,
            )
            if avg_loss["seg_loss"] is not None:
                mlflow.log_metric(
                    f"Train_seg_loss:{args.seg_param}",
                    avg_loss["seg_loss"],
                    step=epoch,
                )
            if avg_loss["flow_with_seg_loss"] is not None:
                mlflow.log_metric(
                    f"Train_flow_with_seg_loss gamma:{args.gamma_param}",
                    avg_loss["flow_with_seg_loss"],
                    step=epoch,
                )
            if avg_loss["background_loss"] is not None:
                mlflow.log_metric(
                    f"Train_background_loss:{args.background_param}",
                    avg_loss["background_loss"],
                    step=epoch,
                )
            if avg_loss["cycle_loss"] is not None:
                mlflow.log_metric(
                    f"Train_cycle_loss:{args.cycle_param}",
                    avg_loss["cycle_loss"],
                    step=epoch,
                )
        # Print progress
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss['loss']:.6f}")  # noqa: T201

        # Save periodic checkpoints
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = output_dir / f"checkpoints/checkpoint_epoch{epoch + 1}.pt"
            torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        if avg_loss["loss"] < best_loss:
            best_loss = avg_loss["loss"]
            best_path = output_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            if mlflow.active_run():
                mlflow.pytorch.log_model(model)

    # Save final model
    final_path = output_dir / "final.pt"
    torch.save(model.state_dict(), final_path)
    config = {
        "model": args.model_type,
        "max_gap": args.max_gap,
        "segmentations": args.seg_path is not None,
        "dataset": args.video_path,
        "epochs": args.epochs,
        "image_loss": args.image_loss,
        "grad_loss": args.grad_loss,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "lambda": args.lambda_param,
        "gamma": args.gamma_param,
        "background": args.background,
        "background_param": args.background_param,
        "cycle_consistency": args.cycle_consistency,
        "cycle_param": args.cycle_param,
        "seg_param": args.seg_param,
        "multiple_videos": args.multiple_videos,
        "device": str(configs.device),
        "date": timestamp.strftime("%Y%m%d_%H%M%S"),
    }
    # create loss figure
    plt.plot(train_loss)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig(output_dir / "train_loss.png")
    if mlflow.active_run():
        run_id = mlflow.active_run().info.run_id
        config["mlflow_run_id"] = run_id
    with Path.open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    print(f"Final model saved to {final_path}")  # noqa: T201

    # End MLflow run
    if mlflow.active_run():
        mlflow.end_run()

    end_timestamp = datetime.datetime.now(tz=datetime.UTC)
    exec_time = (end_timestamp - timestamp).seconds

    return output_dir, exec_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train")
    parser.add_argument("--mlflow", action="store_true", help="Enable mlflow tracking during training")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    # Directory and paths
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--video-path",
        default=configs.TRAIN_DATA_MUSC,
        help="Path to the training video",
    )
    parser.add_argument(
        "--seg-path",
        default=configs.TRAIN_SEG_DATA_MUSC,
        help="Path to the segmentation of the training video, if not available set to None",
    )
    # Dataset parameters
    parser.add_argument("--max_gap", type=int, default=1, help="Maximum temporal gap between 2 frames")
    parser.add_argument(
        "--multiplier",
        type=int,
        default=1,
        help="Multiplier to have more pairs per epoch",
    )
    parser.add_argument(
        "--seg_type",
        type=str,
        default="binary",
        choices=["binary", "heatmaps", None],
        help="Type of segmentation to use for training: None, 'binary' or heatmaps",
    )
    # Training parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="voxelmorph",
        choices=["voxelmorph", "twoflows", "rasor-cnn", "rasor", "transmorph2d", "flowseg", "transflowseg"],
        help="Registration backbone to train. Use 'rasor-cnn' or 'rasor' for the RASOR models.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--save-every", type=int, default=10, help="Checkpoint every N epochs")
    parser.add_argument(
        "--cycle_consistency",
        action="store_true",
        help="Add loss to enforce cycle consistency, automatic True if model_type is twoflows",
    )
    parser.add_argument(
        "--cycle_param",
        type=float,
        dest="cycle_param",
        default=0.005,
        help="Regularisation parameter for cycle loss",
    )
    parser.add_argument("--image-loss", type=str, default="mse", help="Image loss to use: 'mse' or 'cc'")
    parser.add_argument("--grad_loss", type=str, default="l2", help="Gradient loss to use: 'l1' or 'l2'")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_param",
        default=0.01,
        help="Regularisation parameter for smooth loss",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        dest="gamma_param",
        default=0.01,
        help="Regularisation parameter for loss between source and warped target, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--seg_param",
        type=float,
        dest="seg_param",
        default=0.01,
        help="Regularisation parameter for segmentation loss, not used if seg_path is set to None",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Add loss to avoid displacement in the background, only available if segmentations available",
    )
    parser.add_argument(
        "--background_param",
        type=float,
        default=0.01,
        help="Parameter for background loss, only available if segmentations available",
    )
    parser.add_argument(
        "--multiple_videos",
        action="store_true",
        help="Use multiple videos for training, to allow the model to generalize better",
    )
    args = parser.parse_args()
    _ = train(args)
