"""Class definitions for different model + a warper class."""

from collections.abc import Sequence
from typing import Any

import neurite as ne
import torch
import torch.nn.functional as F
from torch import nn

from deepfluxoptique.model_transformer import TransFlowSegUNet, TransMorph2D
from deepfluxoptique.my_utils import spatial_transform


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


def downsampling_conv_blocks(
    ndim: int,
    in_channels: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
) -> tuple[nn.ModuleList, list[int]]:
    """Create an `nn.ModuleList` of downsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each level of the downsampling convs.
    kernel_size : int, default=3
        Size of the convolving kernel at each level.

    Returns:
    -------
    tuple[nn.ModuleList, list[int]]
        A tuple of (blocks, actual_channels) where:
        - blocks: ModuleList of downsampling blocks (DownsampleConvBlock or DownsamplePassthrough)
        - actual_channels: List of actual output channel counts for each block, accounting for
          pass-through levels where channels are preserved
    """
    # Init container for downsampling convs and track actual channels
    blocks = nn.ModuleList()

    in_ch = in_channels

    # Make downsampling conv block and append to list of them
    for out_ch in nb_features:
        block = ConvBlock(in_ch, out_ch, ndim, kernel_size)
        blocks.append(block)

        in_ch = out_ch

    return blocks


def upsampling_conv_blocks(
    ndim: int,
    nb_features: Sequence[int],
    kernel_size: int = 3,
    skip_channels: Sequence[int] | None = None,
    in_channels: int | None = None,
) -> tuple[nn.ModuleList, list[int]]:
    """Create an `nn.ModuleList` of upsampling conv blocks based the number of features per layer.

    Parameters
    ----------
    ndim : int
        Dimensionality of the convolution (2 for Conv2d, 3 for Conv3d).
    nb_features : Sequence[int]
        Number of features at each level of the upsampling convs.
    kernel_size : int, default=3
        Size of the convolving kernel at each level.
    in_channels : int or None, default=None
        Actual input channels to the first upsampling block (from bottleneck). If None, uses
        nb_features[0]. This is needed when nb_features[0] is 0 (pass-through) but the actual
        input comes from a bottleneck layer with non-zero channels.

    Returns:
    -------
    nn.ModuleList
        blocks: ModuleList of upsampling blocks (UpsampleConvBlock)
    """
    # Init upsampling conv blocks container and track actual channels
    blocks = nn.ModuleList()

    in_ch = in_channels

    # Make upsampling conv block and append to list of them
    for out_ch, skip_ch in zip(nb_features, skip_channels, strict=False):
        block = ConvBlock(in_ch + skip_ch, out_ch, ndim, kernel_size)
        blocks.append(block)

        in_ch = out_ch

    return blocks


class TwoFlowsUNet(nn.Module):
    """Two-Flows UNet with two decoders."""

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_features: Sequence[int] | Sequence[Sequence[int]] = (
            16,
            16,
            16,
            16,
            16,
        ),
    ):
        """Initialize `TwoFlowsUNet`.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : Union[Sequence[int], Sequence[Sequence[int]]], default=(16, 16, 16, 16, 16)
            Number of features at each level of the unet. Can be:
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Symmetric Unet
        nb_downsampling_features = tuple(nb_features)
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.downsampling_conv_blocks = downsampling_conv_blocks(
            ndim=ndim,
            in_channels=self.in_channels,
            nb_features=nb_downsampling_features,
        )

        # -------- Bottleneck---------
        bottleneck_channels = nb_downsampling_features[-1]
        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        final_channels = nb_upsampling_features[-1]

        # -------- Decoder FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_fwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

        # -------- Decoder BWD ---------
        self.upsampling_conv_blocks_bwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_bwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

    def forward(self, feature_tensor: torch.Tensor) -> tuple[Any, Any]:
        """Forward pass through the `TwoFlowsUNet` model."""
        skip_connections = []

        # -------------------------- Downsampling path ----------------------------------------
        for block in self.downsampling_conv_blocks:
            feature_tensor = block(feature_tensor)
            skip_connections.append(feature_tensor)
            maxpool_cls_name = f"max_pool{self.ndim}d"
            maxpool_cls = getattr(F, maxpool_cls_name)
            feature_tensor = maxpool_cls(feature_tensor, 2)

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, reversed(skip_connections), strict=False):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
        feature_tensor_fwd = self.out_layer_fwd(feature_tensor_fwd)

        # -------------------------- Upsampling path BWD ----------------------------------------
        feature_tensor_bwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_bwd, reversed(skip_connections), strict=False):
            feature_tensor_bwd = self.upsample(feature_tensor_bwd)
            feature_tensor_bwd = torch.cat([feature_tensor_bwd, skip], dim=1)
            feature_tensor_bwd = block(feature_tensor_bwd)
        feature_tensor_bwd = self.out_layer_bwd(feature_tensor_bwd)

        return feature_tensor_fwd, feature_tensor_bwd


class FlowSegUNet(nn.Module):
    """FlowSegUNet with two decoders one for flow and one for segmentation."""

    def __init__(
        self,
        ndim: int,
        in_channels: int,
        out_channels: int,
        nb_class_segmentation: int = 1,
        nb_features: Sequence[int] | Sequence[Sequence[int]] = (
            16,
            32,
            64,
            128,
            256,
        ),
    ):
        """Initialize `FlowSegUNet`.

        Parameters
        ----------
        ndim : int
            Number of spatial dimensions of the input (1, 2, or 3).
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        nb_features : Union[Sequence[int], Sequence[Sequence[int]]], default=(16, 16, 16, 16, 16)
            Number of features at each level of the unet
        nb_class_segmentation : int, default=1
            Number of classes for segmentation output. Default is 1 (binary segmentation).
        """
        super().__init__()

        self.ndim = ndim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nb_class_segmentation = nb_class_segmentation

        # Symmetric Unet
        nb_downsampling_features = tuple(nb_features)
        nb_upsampling_features = tuple(reversed(nb_features))
        nb_skip_channels = tuple(reversed(nb_features))

        # -------- Encoder ---------
        self.downsampling_conv_blocks = downsampling_conv_blocks(
            ndim=ndim,
            in_channels=self.in_channels,
            nb_features=nb_downsampling_features,
        )

        # -------- Bottleneck---------
        bottleneck_channels = nb_downsampling_features[-1]
        self.lowest_resolution_conv_block = ConvBlock(
            ndim=ndim,
            in_channels=bottleneck_channels,
            out_channels=bottleneck_channels,
        )

        # Upsampling
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        final_channels = nb_upsampling_features[-1]

        # -------- Decoder Flow FWD ---------
        self.upsampling_conv_blocks_fwd = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_fwd = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

        # -------- Decoder Segmentation ---------
        self.upsampling_conv_blocks_seg = upsampling_conv_blocks(
            ndim=ndim,
            nb_features=nb_upsampling_features,
            in_channels=bottleneck_channels,
            skip_channels=nb_skip_channels,
        )

        self.out_layer_seg = ConvBlock(
            ndim=ndim,
            in_channels=final_channels,
            out_channels=self.nb_class_segmentation,
            kernel_size=1,
            padding=0,
            activation=False,
            batchnorm=False,
        )

    def forward(self, feature_tensor: torch.Tensor) -> tuple[Any, Any]:
        """Forward pass through the `FlowSegUNet` model."""
        skip_connections = []

        # -------------------------- Downsampling path ----------------------------------------
        for block in self.downsampling_conv_blocks:
            feature_tensor = block(feature_tensor)
            skip_connections.append(feature_tensor)
            maxpool_cls_name = f"max_pool{self.ndim}d"
            maxpool_cls = getattr(F, maxpool_cls_name)
            feature_tensor = maxpool_cls(feature_tensor, 2)

        # -------------------------- Bottleneck ----------------------------------------
        feature_tensor = self.lowest_resolution_conv_block(feature_tensor)

        # -------------------------- Upsampling path FWD ----------------------------------------
        feature_tensor_fwd = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_fwd, reversed(skip_connections), strict=False):
            feature_tensor_fwd = self.upsample(feature_tensor_fwd)
            feature_tensor_fwd = torch.cat([feature_tensor_fwd, skip], dim=1)
            feature_tensor_fwd = block(feature_tensor_fwd)
        feature_tensor_fwd = self.out_layer_fwd(feature_tensor_fwd)

        # -------------------------- Upsampling path segmentation ----------------------------------------
        feature_tensor_seg = feature_tensor
        for block, skip in zip(self.upsampling_conv_blocks_seg, reversed(skip_connections), strict=False):
            feature_tensor_seg = self.upsample(feature_tensor_seg)
            feature_tensor_seg = torch.cat([feature_tensor_seg, skip], dim=1)
            feature_tensor_seg = block(feature_tensor_seg)
        feature_tensor_seg = self.out_layer_seg(feature_tensor_seg)

        return feature_tensor_fwd, feature_tensor_seg


class WarperVxm(nn.Module):
    """A warper architecture to perform 2D or 3D image registration using a flow field.

    Inspired by 'VxmPairWise' from :
    VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
    Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
    pp 1788-1800. 2019.

    Parameters
    ----------
    ndim : int
        Number of spatial dimensions (2 or 3).
    source_channels : int
        Number of channels in the source image.
    target_channels : int
        Number of channels in the target image.
    nb_features : List[int], optional
        List of integers specifying the number of features in each
        level of the UNet architecture. Default is `[16, 16, 16, 16, 16]`.
    model_type : str
        Model to use, choices are "voxelmorph", "flowseg", "transflowseg", "transmorph2d". Default is "voxelmorph".
    flow_initializer : float, optional
        Standard deviation for initializing the flow layer weights with a
        normal distribution (mean=0). Default is `1e-5`.
    device : str
        Device : 'cpu' or 'cuda'.

    Attributes:
    ----------
    flow_layer : nn.Module
        A custom convolutional block used to generate the flow field
        from the combined source and target features.

    Methods:
    -------
    forward(source, target, return_warped_source, return_field_type)
        Combines source and target images, processes them through the UNet and the flow layer,
        and returns the displacement field. Optionally returns warped source and/or
        target images.
    """

    def __init__(
        self,
        ndim: int,
        source_channels: int,
        target_channels: int,
        nb_features: Sequence[int] = (16, 16, 16, 16, 16),
        model_type: str = "voxelmorph",
        flow_initializer: float = 1e-5,
        device: str = "cpu",
    ):
        """Initialize the `WarperVxm`.

        Parameters
        ----------
        ndim : int
            Dimensionality of the input (1, 2, or 3).
        source_channels : int
            Number of channels in the `source_tensor` input to the forward method of this class.
        target_channels : int
            Number of channels in the `target_tensor` input to the forward method of this class.
        nb_features : List[int]
            Number of features at each level of the unet. Must be a list of
            positive integers.
        model_type : str
            Model to use, choices are "voxelmorph", "flowseg", "transflowseg", "transmorph2d".
        flow_initializer : float, optional
            Standard deviation for initializing the flow layer weights with a
            normal distribution (mean=0). Default is `1e-5`.
        device : str, optional
            Device identifier (e.g., 'cpu' or 'cuda') to place/run the model on.
        """
        super().__init__()

        self.device = device

        self.model_type = model_type

        if self.model_type == "twoflows":
            self.model = TwoFlowsUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
            ).to(self.device)
        elif self.model_type == "flowseg":
            self.model = FlowSegUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
            ).to(self.device)
        elif self.model_type == "voxelmorph":
            self.model = ne.nn.models.BasicUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
                nb_features=nb_features,
                activations=nn.ReLU,
                final_activation=None,
            ).to(self.device)
        elif self.model_type == "transflowseg":
            self.model = TransFlowSegUNet(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
            ).to(self.device)
        elif self.model_type == "transmorph2d":
            self.model = TransMorph2D(
                ndim=ndim,
                in_channels=(source_channels + target_channels),
                out_channels=ndim,
            ).to(self.device)

        self.flow_initializer = flow_initializer
        self._init_flow_layer(ndim, self.flow_initializer)

        # Initialize the spatial transformer
        self.spatial_transformer = spatial_transform.SpatialTransformer().to(self.device)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        return_warped_source: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        """Forward pass of `WarperVxm`.

        This forward pass concatenates the `source` and `target` images, processes them with the
        chosen backbone, and uses a flow layer to predict a displacement field.

        Parameters
        ----------
        source : torch.Tensor
            Source image tensor with shape (B, C_source, *spatial_dims).
        target : torch.Tensor
            Target image tensor with shape (B, C_target, *spatial_dims).
            Must have the same spatial dimensions as `source`.
        return_warped_source : bool, optional
            If `True`, include the warped source image in the return tuple. Default is `False`.

        Returns:
        -------
        Dict[str, torch.Tensor]
            A dictionary containing:
            - "displacement" : shape (B, ndim, *spatial_dims)
            - "warped_source" : shape (B, C_source, *spatial_dims) if `return_warped_source=True`
        """
        res = {}

        # Pass combined features through the model's backbone & flow layer
        combined_features = torch.cat([source, target], dim=1)
        if self.model_type == "twoflows":
            displacement, displacement_bwd = self.model(combined_features)
            res["displacement_bwd"] = displacement_bwd
        elif self.model_type == "voxelmorph":
            unet_out = self.model(combined_features)
            displacement = self.flow_layer(unet_out)
        elif self.model_type in {"flowseg", "transflowseg"}:
            displacement, segmentation = self.model(combined_features)
            res["source_segmentation"] = segmentation
        elif self.model_type == "transmorph2d":
            displacement = self.model(combined_features)

        if return_warped_source:
            warped_source = self.spatial_transformer(source, displacement)
            res["warped_source"] = warped_source
        res["displacement"] = displacement
        return res

    def _init_flow_layer(self, ndim: int, flow_initializer: float = 1e-5) -> None:
        """Initialize the flow layer with custom weight initialization.

        This layer is a convolutional block that produces a displacement (flow)
        field. The weights of its initial convolution are initialized from a
        normal distribution with mean=0 and std=flow_initializer, and biases
        are set to zero.

        Parameters
        ----------
        ndim : int
            **Spatial** dimensionality of the input (1, 2, or 3).
        features : int
            Number of input and output features for the flow layer.
        flow_initializer : float, optional
            Standard deviation for initializing the flow layer weights with a
            normal distribution (mean=0). Default is `1e-5`.
        """
        # Initialize weights from Normal(mean=0, std=flow_initializer)
        if self.model_type == "twoflows":
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_fwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_fwd.conv[0].bias is not None:
                    self.model.out_layer_fwd.conv[0].bias.zero_()

            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_bwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_bwd.conv[0].bias is not None:
                    self.model.out_layer_bwd.conv[0].bias.zero_()
        elif self.model_type == "voxelmorph":
            flow_layer = ne.nn.modules.ConvBlock(ndim, 2, 2).to(self.device)
            # Initialize weights from Normal(mean=0, std=flow_initializer)
            with torch.no_grad():
                torch.nn.init.normal_(flow_layer.conv0.weight, mean=0.0, std=flow_initializer)
                # Set the bias term(s) to zero for the first (and only) conv
                if flow_layer.conv0.bias is not None:
                    flow_layer.conv0.bias.zero_()
            # Register the flow layer as a submodule
            self.add_module("flow_layer", flow_layer)
        elif self.model_type == "flowseg":
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.out_layer_fwd.conv[0].weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.out_layer_fwd.conv[0].bias is not None:
                    self.model.out_layer_fwd.conv[0].bias.zero_()

        elif self.model_type in {"transflowseg", "transmorph2d"}:
            with torch.no_grad():
                torch.nn.init.normal_(
                    self.model.reg_head.weight,
                    mean=0.0,
                    std=flow_initializer,
                )
                # Set the bias term(s) to zero for the first (and only) conv
                if self.model.reg_head.bias is not None:
                    self.model.reg_head.bias.zero_()
