"""Class definitions for models with transformers."""

from collections.abc import Sequence
from typing import Any

import timm
import torch
from torch import nn


class ConvBlock(nn.Module):
    """A convolutional block witha convolutional layer, optional activation, and optional batch normalization."""

    def __init__(
        self,
        in_channels,
        out_channels,
        ndim=2,
        kernel_size=3,
        padding=1,
        activation=True,
        batchnorm=True,
    ):
        """Initialize `ConvBlock`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        kernel_size : int or tuple, default=3
            Size of the convolving kernel.
        """
        super().__init__()
        conv_cls_name = f"Conv{ndim}d"
        conv_cls = getattr(nn, conv_cls_name)
        modules = [conv_cls(in_channels, out_channels, kernel_size=kernel_size, padding=padding)]
        if activation:
            modules.append(nn.ReLU())  # the one used by default in vxm
        if batchnorm:
            batchnorm_cls_name = f"BatchNorm{ndim}d"
            batchnorm_cls = getattr(nn, batchnorm_cls_name)
            modules.append(batchnorm_cls(out_channels))  # None by default in vxm
        self.conv = nn.Sequential(*modules)  # TODO append à sequential directement

    def forward(self, x):
        return self.conv(x)


def upsampling_2conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    skip_channels: Sequence[int] | None = None,
    in_channels: int | None = None,
) -> tuple[nn.ModuleList, list[int]]:
    """Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer.

    Parameters:
    ----------
    ndim : int
        Dimensionality of the convolution (1 for Conv1d, 2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each upsampling conv block. Use `0` to indicate a pass-through
        level (upsample + skip concat only, no convolution). When `0` is used, the actual
        output channels equal the upsampled channels plus skip channels (if accepts_skip=True).
    kernel_size : int, default=3
        Size of the convolving kernel at each level.
    skip_channels : Sequence[int] or None, default=None
        List of channel counts for skip connections, one per upsampling block. If provided and
        accepts_skip=True, enables asymmetric downsampling/upsampling architectures. If None and
        accepts_skip=True, assumes symmetric architecture (skip channels equal upsampleda_bl
        channels).
    in_channels : int or None, default=None
        Actual input channels to the first upsampling block (from bottleneck). If None, uses
        nb_features[0]. This is needed when nb_features[0] is 0 (pass-through) but the actual
        input comes from a bottleneck layer with non-zero channels.

    Returns:
    -------
    nn.ModuleList
        blocks: ModuleList of upsampling blocks (UpsampleConvBlock or UpsamplePassthrough)
    """
    # Init upsampling conv blocks container and track actual channels
    blocks = nn.ModuleList()

    in_ch = in_channels

    # Make upsampling conv block and append to list of them
    for out_ch, skip_ch in zip(nb_features, skip_channels, strict=False):
        block1 = ConvBlock(in_ch + skip_ch, out_ch, ndim, batchnorm=False)
        block2 = ConvBlock(out_ch, out_ch, ndim, batchnorm=False)
        blocks.append(nn.Sequential(block1, block2))

        in_ch = out_ch

    return blocks


class TransFlowSegUNet(nn.Module):
    """UNet with SwinTransformer encoder and 2 decoders, one for the source segmentation and one for the flow.

    Widely inspired by :
    Chen, Junyu, Eric C. Frey, Yufan He, William P. Segars, Ye Li, and Yong Du.
    “TransMorph: Transformer for Unsupervised Medical Image Registration.”
    Medical Image Analysis 82 (November 2022): 102615. https://doi.org/10.1016/j.media.2022.102615.
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_class_segmentation: int = 1,
        embed_dim: int = 96,
        reg_head_channels: int = 16,
    ):
        """Initialize `TransFlowSegUNet`.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        embed_dim : int
            Dimensionality of the embedding space.
        reg_head_channels : int
            Number of channels in the registration head.
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_class_segmentation = nb_class_segmentation
        self.reg_head_channels = reg_head_channels

        # Symmetric Unet
        nb_features = [
            self.reg_head_channels,
            embed_dim // 2,
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
        ]
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.convrelu1 = ConvBlock(ndim=ndim, in_channels=2, out_channels=embed_dim // 2)
        self.convrelu2 = ConvBlock(ndim=ndim, in_channels=2, out_channels=self.reg_head_channels)

        self.transformer = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            img_size=1024,
            in_chans=2,
            features_only=True,
            out_indices=[0, 1, 2, 3],
        )  # source code at : https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py

        # -------- Bottleneck---------
        bottleneck_channels = embed_dim * 8

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # -------- Decoder Flow FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_2conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.reg_head = nn.Conv2d(self.reg_head_channels, out_channels, kernel_size=1, padding=0)

        # -------- Decoder Segmentation ---------
        self.upsampling_conv_blocks_seg = upsampling_2conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )
        self.seg_head = nn.Conv2d(self.reg_head_channels, self.nb_class_segmentation, kernel_size=1, padding=0)

    def forward(self, feature_tensor: torch.Tensor) -> tuple[Any, Any]:
        """Forward pass through the `TransFlowSegUNet` model."""
        # -------------------------- Downsampling path ----------------------------------------

        f4 = self.convrelu1(self.avg_pool(feature_tensor))
        f5 = self.convrelu2(feature_tensor.clone())

        out = self.transformer(feature_tensor)

        f1 = out[-2].permute(0, 3, 1, 2).contiguous()
        f2 = out[-3].permute(0, 3, 1, 2).contiguous()
        f3 = out[-4].permute(0, 3, 1, 2).contiguous()

        skip_connections = [f1, f2, f3, f4, f5]

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = out[-1].permute(0, 3, 1, 2).contiguous()

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, skip_connections, strict=False):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)

        feature_tensor_fwd = self.reg_head(feature_tensor_fwd)

        # -------------------------- Upsampling path segmentation ----------------------------------------
        feature_tensor_seg = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_seg, skip_connections, strict=False):
            feature_tensor_seg = self.upsample(feature_tensor_seg)
            feature_tensor_seg = torch.cat([feature_tensor_seg, skip], dim=1)
            feature_tensor_seg = block(feature_tensor_seg)
        feature_tensor_seg = self.seg_head(feature_tensor_seg)

        return feature_tensor_fwd, feature_tensor_seg


class TransMorph2D(nn.Module):
    """UNet with a SwinTransformer encoder for 2D images.

    From the paper:
    Chen, Junyu, Eric C. Frey, Yufan He, William P. Segars, Ye Li, and Yong Du.
    “TransMorph: Transformer for Unsupervised Medical Image Registration.”
    Medical Image Analysis 82 (November 2022): 102615. https://doi.org/10.1016/j.media.2022.102615.

    Attributes:
    ----------
    downsampling_conv_blocks : nn.ModuleList
        Downsampling convolutional blocks.
    lowest_resolution_conv_block : nn.Module
        Central convolutional block at the lowest spatial resolution.
    upsampling_conv_blocks : nn.ModuleList
        Upsampling convolutional blocks.
    out_layer : nn.Module
        Final output layer.
    down_actual_channels : list[int]
        Actual output channel counts for each downsampling block (accounts for pass-through levels).
    """

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_class_segmentation: int = 1,
        embed_dim: int = 96,
        reg_head_channels: int = 16,
    ):
        """Initialize `TransMorph2D`.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        embed_dim : int
            Embeding dimension.
        reg_head_channels : int
            Number of input channels of the last layer.
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_class_segmentation = nb_class_segmentation
        self.reg_head_channels = reg_head_channels

        # Symmetric Unet
        nb_features = [
            self.reg_head_channels,
            embed_dim // 2,
            embed_dim,
            embed_dim * 2,
            embed_dim * 4,
        ]
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.convrelu1 = ConvBlock(ndim=ndim, in_channels=2, out_channels=embed_dim // 2)
        self.convrelu2 = ConvBlock(ndim=ndim, in_channels=2, out_channels=self.reg_head_channels)

        self.transformer = timm.create_model(
            "swinv2_tiny_window8_256",
            pretrained=True,
            img_size=1024,
            in_chans=2,
            features_only=True,
            out_indices=[0, 1, 2, 3],
        )  # source code at : https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/swin_transformer_v2.py

        # -------- Bottleneck---------
        bottleneck_channels = embed_dim * 8

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")

        # -------- Decoder Flow FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_2conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.reg_head = nn.Conv2d(self.reg_head_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, feature_tensor: torch.Tensor):
        """Forward pass through the `TransMorph2D` model."""
        # -------------------------- Downsampling path ----------------------------------------
        f4 = self.convrelu1(self.avg_pool(feature_tensor))
        f5 = self.convrelu2(feature_tensor.clone())

        out = self.transformer(feature_tensor)

        f1 = out[-2].permute(0, 3, 1, 2).contiguous()
        f2 = out[-3].permute(0, 3, 1, 2).contiguous()
        f3 = out[-4].permute(0, 3, 1, 2).contiguous()

        skip_connections = [f1, f2, f3, f4, f5]

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = out[-1].permute(0, 3, 1, 2).contiguous()

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, skip_connections, strict=False):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)

        return self.reg_head(feature_tensor_fwd)
