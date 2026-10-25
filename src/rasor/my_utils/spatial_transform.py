"""
Code from :
VoxelMorph: A Learning Framework for Deformable Medical Image Registration G. Balakrishnan, A.
Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. IEEE TMI: Transactions on Medical Imaging. 38(8).
pp 1788-1800. 2019."""

from collections.abc import Sequence
from typing import Literal

import neurite as ne
import torch
from torch import nn


class SpatialTransformer(nn.Module):
    """
    N-D Spatial transformation according to a deformation field.

    Wrapper around voxelmorph.nn.functional.spatial_transform that maintains the
    nn.Module interface for composability in neural network architectures.

    References
    ----------
    If you find this helpful, please cite the following paper:

    VoxelMorph: A Learning Framework for Deformable Medical Image Registration
    G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca.
    IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019.
    """

    def __init__(
        self,
        interpolation_mode: str = "linear",
        align_corners: bool = True,
        device: str | torch.device | None = None,
    ):
        """
        Initialize `SpatialTransformer`.

        Parameters
        ----------
        interpolation_mode : str, default='linear'
            Algorithm used for interpolating the warped image. Options are: {'linear', 'nearest'}
        align_corners : bool, default=True
            Map the corner points of the moving image to the corner points of the warped image.
        device : str or torch.device or None, optional
            Deprecated. No longer used. Kept for backward compatibility.
        """
        super().__init__()

        self.interpolation_mode = interpolation_mode
        self.align_corners = align_corners

    def forward(
        self, moving_image: torch.Tensor, deformation_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of `SpatialTransformer`.

        Parameters
        ----------
        moving_image : torch.Tensor
            Tensor to be spatially transformed by `deformation_field`.
            Shape: (B, C, *spatial_dims).
        deformation_field : torch.Tensor
            Field causing the spatial transformation of `moving_image`.
            Shape: (B, ndim, *spatial_dims).

        Returns
        -------
        torch.Tensor
            Warped `moving_image` according to the `deformation_field`.
            Output shape matches `moving_image` shape: (B, C, *spatial_dims).

        Notes
        -----
        - Expects deformation_field in channels-first format: (B, ndim, *spatial_dims)
        - Processes each batch element independently since vxm.spatial_transform
          expects displacement fields without batch dimension
        """
        assert moving_image.dim() >= 4, (
            f"moving_image must have >=4 dims (B,C,*spatial), got {moving_image.shape}"
        )
        assert deformation_field.dim() == moving_image.dim(), (
            f"dim mismatch: moving={moving_image.dim()}, field={deformation_field.dim()}"
        )

        # Allocate or reallocate meshgrid if spatial shape changed
        spatial_shape = moving_image.shape[2:]
        if not hasattr(self, "meshgrid") or self.meshgrid.shape[1:] != spatial_shape:
            self.meshgrid = ne.volshape_to_ndgrid(
                size=spatial_shape,
                device=moving_image.device,
                dtype=moving_image.dtype,
                stack=True,
            )

        return spatial_transform(
            image=moving_image,
            trf=deformation_field,
            mode=self.interpolation_mode,
            isdisp=True,
            meshgrid=self.meshgrid,
            non_spatial_dims=(0, 1),
            align_corners=self.align_corners,
            padding_mode="zeros",
        )


def spatial_transform(
    image: torch.Tensor,
    trf: torch.Tensor | None,
    mode: Literal["linear", "nearest"] = "linear",
    isdisp: bool = True,
    meshgrid: torch.Tensor | None = None,
    origin_at_center: bool = True,
    non_spatial_dims: tuple[int, ...] | None = None,
    align_corners: bool = True,
    padding_mode: Literal["zeros", "border", "reflection"] = "zeros",
) -> torch.Tensor:
    """
    Apply spatial transformation to image using displacement or coordinate field.

    Shape-agnostic implementation that works with any tensor dimensionality.

    Parameters
    ----------
    image : torch.Tensor
        Input image to transform. Shape depends on non_spatial_dims:
        - (*spatial,) if non_spatial_dims=None
        - (C, *spatial) if non_spatial_dims=(0,)
        - (B, C, *spatial) if non_spatial_dims=(0, 1)
        - etc...
    trf : torch.Tensor or None
        Transformation field. Can be:
        - Affine matrix: shape (N+1, N+1) or (N, N+1)
        - Batched affine matrix: shape (B, N+1, N+1) or (B, N, N+1)
        - Displacement field: shape (N, *spatial) or (B, N, *spatial) - channels-first
        - None: returns image unchanged
    mode : {'linear', 'nearest'}, default='linear'
        Interpolation mode. 'linear' will auto-detect appropriate mode (bilinear/trilinear) based
        on spatial dimensionality.
    isdisp : bool, default=True
        If True, treat trf as displacement field (ndim, *spatial) and normalize to [-1, 1].
        If False, treat trf as already-normalized coordinates (ndim, *spatial).
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed from image shape.
    origin_at_center : bool, default=True
        Place origin at image center when converting affine matrices to displacement.
    non_spatial_dims : Tuple[int, ...] or None, default=None
        Which dimensions of image are non-spatial:
        - None: pure spatial tensor
        - (0,): first dimension is non-spatial (e.g., channel)
        - (0, 1): first two dimensions are non-spatial (e.g., batch, channel)
        - etc...
    align_corners : bool, default=True
        Align corners parameter for grid_sample.
    padding_mode : {'zeros', 'border', 'reflection'}, default='zeros'
        Padding mode for grid_sample when sampling outside the input bounds.

    Returns
    -------
    torch.Tensor
        Transformed image with same shape as input.

    Examples
    --------
    >>> # Pure spatial image (H, W) with displacement (ndim, H, W)
    >>> image = torch.randn(64, 64)
    >>> disp = torch.randn(2, 64, 64)
    >>> warped = spatial_transform(image, disp)
    >>> warped.shape
    torch.Size([64, 64])

    >>> # Image with channel dimension (C, H, W)
    >>> image = torch.randn(3, 64, 64)
    >>> disp = torch.randn(2, 64, 64)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0,))
    >>> warped.shape
    torch.Size([3, 64, 64])

    >>> # Image with batch and channel (B, C, H, W), batched displacement (B, ndim, H, W)
    >>> image = torch.randn(2, 3, 64, 64)
    >>> disp = torch.randn(2, 2, 64, 64)
    >>> warped = spatial_transform(image, disp, non_spatial_dims=(0, 1))
    >>> warped.shape
    torch.Size([2, 3, 64, 64])

    >>> # Batched affine transformations (different transform per batch)
    >>> image = torch.randn(2, 3, 64, 64)
    >>> affines = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (2, 3, 3)
    >>> warped = spatial_transform(image, affines, non_spatial_dims=(0, 1))
    >>> warped.shape
    torch.Size([2, 3, 64, 64])
    """
    # Early return for no transformation
    if trf is None:
        return image

    # Parse image dimensions to understand shape
    num_non_spatial, num_spatial = ne.functional.parse_non_spatial_dims(
        non_spatial_dims, image.ndim
    )
    spatial_shape = image.shape[num_non_spatial:]

    is_affine = False
    if trf.ndim == 2 and is_affine_shape(trf.shape):
        is_affine = True
    elif trf.ndim == 3 and is_affine_shape(trf.shape):
        trf_spatial_like = trf.shape[1:]
        if trf_spatial_like == spatial_shape:
            is_affine = False
        else:
            rows, cols = trf.shape[-2], trf.shape[-1]
            # Last dimensions are small. probably an affine. Could misclassify small disp field
            if rows <= 4 and cols <= 5:
                is_affine = True

    if is_affine:
        # Invert affine to get source-to-target mapping for warping
        trf = torch.linalg.inv(trf)
        trf = affine_to_disp(
            trf, meshgrid, shape=spatial_shape, origin_at_center=origin_at_center
        )
        isdisp = True

    # Detect batch dimension in transform
    # trf is (ndim, *spatial) or (B, ndim, *spatial)
    trf_has_batch_dim = trf.ndim > (num_spatial + 1)

    if isdisp:
        trf_non_spatial = (0,) if trf_has_batch_dim else None
        trf = disp_to_coords(trf, meshgrid=meshgrid, non_spatial_dims=trf_non_spatial)

    # Convert (ndim, *spatial) -> (*spatial, ndim) for grid_sample
    # and flip coordinate order (grid_sample expects reversed spatial dims)
    ndim_dim = 1 if trf_has_batch_dim else 0
    trf = trf.movedim(ndim_dim, -1).flip(-1)

    if mode == "linear":
        mode = ne.utils.infer_linear_interpolation_mode(num_spatial)

    # grid_sample only accepts 'bilinear', 'nearest', 'bicubic'
    if mode == "trilinear":
        mode = "bilinear"

    # Prepare image for grid_sample (must have B, C)
    original_dtype = None
    if not torch.is_floating_point(image):
        if mode == "nearest":
            original_dtype = image.dtype
        image = image.type(torch.float32)

    # Add batch/channel dimensions to reach (B, C, *spatial) format
    dims_added = 2 - num_non_spatial
    for _ in range(dims_added):
        image = image.unsqueeze(0)

    # Prepare coordinates for grid_sample (requires batch dimension)
    # After conversion, trf is (*spatial, ndim) or (B, *spatial, ndim)
    trf_has_batch_dim = trf.ndim > (num_spatial + 1)
    if not trf_has_batch_dim:
        trf = trf.unsqueeze(0)

    # Apply transformation
    transformed = torch.nn.functional.grid_sample(
        image, trf, align_corners=align_corners, mode=mode, padding_mode=padding_mode
    )

    # Restore original format by removing added dimensions
    for _ in range(dims_added):
        transformed = transformed.squeeze(0)
    if original_dtype is not None:
        transformed = transformed.type(original_dtype)

    return transformed


def is_affine_shape(shape: tuple) -> bool:
    """
    Determine whether the given shape represents an N-dimensional affine matrix.

    An affine matrix has shape (..., M, N+1) where:
    - N is the spatial dimensionality (2 or 3)
    - M is either N or N+1 (compact or square form)

    Parameters
    ----------
    shape : tuple
        Shape of the tensor to check.

    Returns
    -------
    bool
        True if shape represents an affine matrix, False otherwise.
    """
    if len(shape) < 2:
        return False

    rows, cols = shape[-2], shape[-1]

    # Cols should be N+1 where N is 2 or 3
    ndim = cols - 1
    if ndim not in (2, 3):
        return False

    # rows should be N or N+1
    return rows in (ndim, ndim + 1)  # Bool


def disp_to_coords(
    disp: torch.Tensor,
    meshgrid: torch.Tensor | None = None,
    non_spatial_dims: tuple[int, ...] | None = None,
) -> torch.Tensor:
    """
    Convert displacement field to normalized coordinates in [-1, 1] range for grid_sample.

    Adds displacement to base meshgrid coordinates and normalizes to [-1, 1].

    Parameters
    ----------
    disp : torch.Tensor
        Displacement field with shape (ndim, *spatial) or (B, ndim, *spatial) if batched.
    meshgrid : torch.Tensor or None, default=None
        Pre-computed coordinate grid of shape (ndim, *spatial). If None, computed
        from displacement field shape.
    non_spatial_dims : tuple[int, ...] or None, default=None
        Indices of non-spatial dimensions preceding the ndim dimension:
        - None: tensor is (ndim, *spatial), unbatched
        - (0,): tensor is (B, ndim, *spatial), batched

    Returns
    -------
    torch.Tensor
        Normalized coordinates in range [-1, 1] with same shape as input.

    Examples
    --------
    >>> # 2d displacement field (ndim, H, W)
    >>> disp = torch.randn(2, 64, 64)
    >>> coords = disp_to_coords(disp)
    >>> coords.shape
    torch.Size([2, 64, 64])

    >>> # Batched displacement field (B, ndim, H, W)
    >>> disp = torch.randn(4, 2, 64, 64)
    >>> coords = disp_to_coords(disp, non_spatial_dims=(0,))
    >>> coords.shape
    torch.Size([4, 2, 64, 64])
    """
    num_non_spatial, num_spatial = ne.functional.parse_non_spatial_dims(
        non_spatial_dims=non_spatial_dims,
        tensor_ndim=disp.ndim - 1,  # subtract 1 for ndim dimension
    )

    has_batch = num_non_spatial == 1
    ndim_axis = 1 if has_batch else 0
    ndim = disp.shape[ndim_axis]
    spatial_shape = disp.shape[ndim_axis + 1 :]

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(
            size=spatial_shape,
            device=disp.device,
            dtype=disp.dtype,
            stack=True,
        )

    coords = meshgrid + disp

    # Normalize each spatial dimension to [-1, 1]
    sizes = torch.tensor(spatial_shape, device=disp.device, dtype=disp.dtype)
    scales = 2.0 / (sizes - 1).clamp(min=1)  # avoid div by zero for size=1
    broadcast_shape = (ndim,) + (1,) * num_spatial
    scales = scales.view(broadcast_shape)

    return coords * scales - 1.0


def affine_to_disp(
    affine: torch.Tensor,
    meshgrid: torch.Tensor | None = None,
    origin_at_center: bool = True,
    shape: Sequence[int] | None = None,
    warp_right: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Convert an affine transformation matrix to a displacement field.

    Parameters
    ----------
    affine : Tensor
        Affine transformation matrix of shape (N, N+1) or (N+1, N+1), or batched
        affine of shape (B, N, N+1) or (B, N+1, N+1).
        Expected to be a vox2vox target-to-source transformation.
    meshgrid : Tensor, optional
        Pre-computed meshgrid tensor of shape (N, *spatial_shape), where N is the spatial
        dimensionality. If None, will be computed from `shape` parameter.
    origin_at_center : bool, optional
        If True, place the coordinate system origin at the image center. If False, origin
        is at the top-left corner. Default is True.
    shape : Sequence[int], optional
        Spatial shape (N dimensions) to create meshgrid if `meshgrid` is not provided.
        Required if `meshgrid` is None.
    warp_right : Tensor, optional
        Right-compose the affine with this displacement field of shape (N, *spatial_shape)
        or (B, N, *spatial_shape) for batched.
        Computes affine(x + warp_right(x)) - x. Useful for composing transforms.

    Returns
    -------
    Tensor
        Displacement field of shape (N, *spatial_shape) for single affine, or
        (B, N, *spatial_shape) for batched affine.

    Examples
    --------
    >>> # Basic usage with pre-computed meshgrid
    >>> import neurite as ne
    >>> affine = torch.tensor(
    >>> ... [[1., 0., 5.],
    >>> ... [0., 1., 3.]]
    >>> )
    >>> grid = ne.volshape_to_ndgrid((64, 64), stack=True)
    >>> disp = affine_to_disp(affine, meshgrid=grid)

    >>> # Using shape parameter instead
    >>> disp = affine_to_disp(affine, shape=(64, 64))
    >>> disp.shape
    torch.Size([2, 64, 64])

    >>> # Compose affine with existing displacement field
    >>> warp = torch.randn(2, 64, 64)  # (ndim, H, W)
    >>> composed = affine_to_disp(affine, shape=(64, 64), warp_right=warp)

    >>> # Batched affine matrices
    >>> affines = torch.eye(3).unsqueeze(0).repeat(2, 1, 1)  # (2, 3, 3)
    >>> disp = affine_to_disp(affines, shape=(64, 64))
    >>> disp.shape
    torch.Size([2, 2, 64, 64])
    """
    assert (meshgrid is None) != (shape is None), (
        "Provide exactly one of `meshgrid` or `shape`"
    )

    if meshgrid is None:
        meshgrid = ne.volshape_to_ndgrid(
            size=shape, device=affine.device, dtype=affine.dtype, stack=True
        )

    assert isinstance(meshgrid, torch.Tensor)
    ndim = meshgrid.shape[0]
    spatial_shape = meshgrid.shape[1:]
    is_batched = affine.ndim == 3

    assert affine.shape[-1] == ndim + 1, (
        f"affine dim ({affine.shape[-1] - 1}D) != meshgrid dim ({ndim}D)"
    )

    # Center origin if requested
    grid = meshgrid
    if origin_at_center:
        center_offsets = [(s - 1) / 2 for s in spatial_shape]
        center_offsets = torch.tensor(center_offsets, device=meshgrid.device).view(
            -1, *[1] * ndim
        )
        grid = meshgrid - center_offsets

    # Flatten grid: (ndim, *spatial) -> (ndim, num_voxels)
    coords = grid.reshape(ndim, -1)

    # Right-compose with displacement field if provided
    if warp_right is not None:
        assert warp_right.shape[-ndim:] == spatial_shape, (
            f"warp_right shape {warp_right.shape[-ndim:]} != meshgrid {spatial_shape}"
        )
        coords = coords + warp_right.reshape(*warp_right.shape[:-ndim], -1)

    # Apply affine: A @ coords + t, then subtract original to get displacement
    transformed = affine[..., :ndim, :ndim] @ coords + affine[..., :ndim, -1:]
    disp_flat = transformed - grid.reshape(ndim, -1)

    # Reshape back to spatial
    output_shape = (
        (affine.shape[0], ndim, *spatial_shape)
        if is_batched
        else (ndim, *spatial_shape)
    )
    return disp_flat.reshape(*output_shape)
