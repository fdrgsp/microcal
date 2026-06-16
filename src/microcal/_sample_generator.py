"""Generate synthetic multi-channel bead images with controllable chromatic shift."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _make_bead_image(
    shape: tuple[int, int],
    centers: NDArray,
    intensity: float = 1.0,
    sigma: float = 2.0,
) -> NDArray:
    """Render a 2D image of point-like beads convolved with a Gaussian PSF.

    The impulse is pre-scaled so that each bead's peak intensity after
    Gaussian blurring equals `intensity`.  For a 2-D Gaussian with std
    `sigma`, the integral is 1 and the peak is 1/(2*pi*sigma**2), so we
    multiply by 2*pi*sigma**2 upfront.
    """
    # pre-scale so post-blur peak ≈ intensity
    impulse_value = float(intensity) * 2.0 * np.pi * sigma**2
    img = np.zeros(shape, dtype=np.float32)
    for cy, cx in centers:
        iy, ix = round(cy), round(cx)
        if 0 <= iy < shape[0] and 0 <= ix < shape[1]:
            img[iy, ix] = impulse_value
    return gaussian_filter(img, sigma=sigma)


def generate_beads_image(
    n_channels: int = 3,
    shape: tuple[int, int] = (512, 512),
    n_beads: int = 50,
    bead_sigma: float = 2.0,
    bead_intensity: float = 60.0,
    bit_depth: int = 8,
    # per-channel shift relative to channel 0, list of (dy, dx) in pixels
    shifts: list[tuple[float, float]] | None = None,
    # per-channel rotation in degrees relative to channel 0
    rotations: list[float] | None = None,
    # per-channel scale factors (sy, sx) relative to channel 0
    scales: list[tuple[float, float]] | None = None,
    offset: int = 10,
    snr: float = 20.0,
    seed: int | None = 42,
) -> tuple[NDArray, dict]:
    """
    Generate a synthetic multi-channel bead image with chromatic shift.

    Parameters
    ----------
    n_channels : int
        Number of fluorescence channels.
    shape : (H, W)
        Spatial size of each channel image.
    n_beads : int
        Number of beads placed in the reference (channel-0) frame.
    bead_sigma : float
        Gaussian PSF sigma in pixels (controls bead size; FWHM ≈ 2.35 * sigma).
    bead_intensity : float
        Peak bead intensity as a percentage of the full dynamic range
        (0-100).  E.g. 60 means 60 % of 2**bit_depth - 1.
    bit_depth : int
        Camera bit depth (e.g. 8 or 16).  Determines the output dtype
        (uint8 / uint16) and the maximum pixel value.
    shifts : list of (dy, dx), length n_channels
        Translation applied to each channel.  Channel 0 entry is ignored.
        None means no shift (identity) for all channels.
    rotations : list of float, length n_channels
        Rotation in degrees applied to each channel.  None means 0 for all.
    scales : list of (sy, sx), length n_channels
        Anisotropic scale applied to each channel.  None means (1,1) for all.
    offset : float
        Camera baseline offset in absolute pixel counts (int) added to every
        pixel before noise, mimicking dark current / electronics bias.  Must
        be an int in [0, 2**bit_depth - 1].  Default 10.
    snr : float
        Signal-to-noise ratio at the bead peak, defined as
        peak / readout_noise_std.  Controls the Gaussian readout noise floor
        (std = peak / snr) applied uniformly to every pixel.  On top of that,
        Poisson shot noise (std ≈ sqrt(signal)) is always added — it scales
        with signal so it affects bright beads more than the dark background.
        Use np.inf to suppress all noise (noiseless simulation).
    seed : int or None
        NumPy random seed for reproducibility.

    Returns
    -------
    stack : NDArray, shape (n_channels, H, W), dtype uint8 or uint16
        Multi-channel bead image.
    ground_truth : dict
        Dictionary with keys 'centers', 'shifts', 'rotations', 'scales' that
        describe the applied transformations (useful for validation).
    """
    if not (0.0 < bead_intensity <= 100.0):
        raise ValueError("bead_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    max_val = float(2**bit_depth - 1)
    if not isinstance(offset, int):
        raise TypeError("offset must be an int")
    if not (0 <= offset <= int(max_val)):
        raise ValueError(
            f"offset ({offset}) must be between 0 and {int(max_val)} "
            f"for {bit_depth}-bit images"
        )
    peak = max_val * bead_intensity / 100.0
    offset_counts = float(offset)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    rng = np.random.default_rng(seed)
    H, W = shape

    # --- defaults (None = identity, no transform) ------------------------
    if shifts is None:
        shifts = [(0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0)] * n_channels

    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    # --- bead centres in channel-0 frame ---------------------------------
    margin = int(bead_sigma * 4)
    centers = np.column_stack(
        [
            rng.uniform(margin, H - margin, size=n_beads),
            rng.uniform(margin, W - margin, size=n_beads),
        ]
    )

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, H, W), dtype=np.float64)

    for ch in range(n_channels):
        ref_img = _make_bead_image(shape, centers, intensity=peak, sigma=bead_sigma)

        dy, dx = shifts[ch]
        rot = rotations[ch]
        sy, sx = scales[ch]

        if dy == 0.0 and dx == 0.0 and rot == 0.0 and sy == 1.0 and sx == 1.0:
            ch_img = ref_img.astype(np.float64)
        else:
            theta = np.deg2rad(rot)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            cy_c, cx_c = H / 2.0, W / 2.0

            A_fwd = np.array(
                [[sy * cos_t, sy * sin_t], [-sx * sin_t, sx * cos_t]], dtype=np.float64
            )
            t_fwd = np.array([dy, dx], dtype=np.float64)

            A_inv = np.linalg.inv(A_fwd)
            centre = np.array([cy_c, cx_c])
            affine_offset = centre - A_inv @ centre - A_inv @ t_fwd

            ch_img = affine_transform(
                ref_img.astype(np.float64),
                A_inv,
                offset=affine_offset,
                output_shape=shape,
                order=3,
                mode="constant",
                cval=0.0,
            )

        ch_img = ch_img + offset_counts
        if np.isfinite(snr):
            # shot noise: Poisson sampling scales with sqrt(signal)
            ch_img = rng.poisson(ch_img).astype(np.float64)
            # readout noise: Gaussian floor, std = peak / snr
            if readout_std > 0.0:
                ch_img += rng.standard_normal(shape) * readout_std

        stack[ch] = ch_img

    # clip lower bound to offset_counts: a camera with a fixed bias never
    # reads below that baseline
    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "centers": centers,
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth


# ---------------------------------------------------------------------------
# 3-D (volumetric) bead generation
# ---------------------------------------------------------------------------


def _as_sigma_tuple(sigma: float | tuple[float, ...], n: int) -> tuple[float, ...]:
    """Coerce a scalar or per-axis sigma into an ``n``-tuple of floats."""
    if isinstance(sigma, (int, float)):
        return (float(sigma),) * n
    t = tuple(float(s) for s in sigma)
    if len(t) != n:
        raise ValueError(f"sigma must have {n} entries, got {len(t)}")
    return t


def _rotation_matrix_3d(rot: float | tuple[float, float, float]) -> NDArray:
    """Build a 3x3 rotation matrix (array order z, y, x) from degrees.

    ``rot`` may be a scalar (rotation about the optical/z axis, i.e. in the
    y-x plane — the direct analogue of the 2-D ``rotations`` argument) or a
    ``(rz, ry, rx)`` tuple of Euler angles composed as ``Rz @ Ry @ Rx``.
    """
    if isinstance(rot, (int, float)):
        rz, ry, rx = float(rot), 0.0, 0.0
    else:
        rz, ry, rx = float(rot[0]), float(rot[1]), float(rot[2])
    az, ay, ax = np.deg2rad([rz, ry, rx])
    cz, sz = np.cos(az), np.sin(az)
    cy, sy = np.cos(ay), np.sin(ay)
    cx, sx = np.cos(ax), np.sin(ax)
    rot_z = np.array([[1.0, 0.0, 0.0], [0.0, cz, sz], [0.0, -sz, cz]])
    rot_y = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    rot_x = np.array([[cx, sx, 0.0], [-sx, cx, 0.0], [0.0, 0.0, 1.0]])
    return rot_z @ rot_y @ rot_x


def _make_bead_image_3d(
    shape: tuple[int, int, int],
    centers: NDArray,
    intensity: float = 1.0,
    sigma: tuple[float, ...] = (1.5, 2.0, 2.0),
) -> NDArray:
    """Render a 3D volume of point-like beads convolved with a Gaussian PSF.

    The impulse is pre-scaled so each bead's peak after Gaussian blurring equals
    `intensity`.  For a separable 3-D Gaussian the peak is
    ``1 / ((2*pi)**1.5 * sz*sy*sx)``, so we multiply by its reciprocal upfront.
    """
    impulse_value = float(intensity) * (2.0 * np.pi) ** 1.5 * float(np.prod(sigma))
    vol = np.zeros(shape, dtype=np.float32)
    for cz, cy, cx in centers:
        iz, iy, ix = round(cz), round(cy), round(cx)
        if 0 <= iz < shape[0] and 0 <= iy < shape[1] and 0 <= ix < shape[2]:
            vol[iz, iy, ix] = impulse_value
    return gaussian_filter(vol, sigma=sigma)


def generate_beads_image_3d(
    n_channels: int = 3,
    shape: tuple[int, int, int] = (32, 256, 256),
    n_beads: int = 50,
    bead_sigma: float | tuple[float, float, float] = (1.5, 2.0, 2.0),
    bead_intensity: float = 60.0,
    bit_depth: int = 8,
    # per-channel shift relative to channel 0, list of (dz, dy, dx) in voxels
    shifts: list[tuple[float, float, float]] | None = None,
    # per-channel rotation in degrees: scalar (about z) or (rz, ry, rx) Euler
    rotations: list[float | tuple[float, float, float]] | None = None,
    # per-channel scale factors (sz, sy, sx) relative to channel 0
    scales: list[tuple[float, float, float]] | None = None,
    offset: int = 10,
    snr: float = 20.0,
    seed: int | None = 42,
) -> tuple[NDArray, dict]:
    """
    Generate a synthetic multi-channel bead volume with chromatic shift.

    The 3-D analogue of :func:`generate_beads_image`.  Beads are rendered with
    an anisotropic Gaussian PSF and each channel is warped by a per-channel 3-D
    affine (shift + rotation + anisotropic scale) about the volume centre.

    Parameters
    ----------
    n_channels : int
        Number of fluorescence channels.
    shape : (Z, Y, X)
        Spatial size of each channel volume.
    n_beads : int
        Number of beads placed in the reference (channel-0) frame.
    bead_sigma : float or (sz, sy, sx)
        Gaussian PSF sigma in voxels.  A tuple makes the PSF axially elongated
        (real PSFs have sz > sxy).
    bead_intensity : float
        Peak bead intensity as a percentage of the full dynamic range (0-100).
    bit_depth : int
        Camera bit depth (8 or 16).  Determines the output dtype and max value.
    shifts : list of (dz, dy, dx), length n_channels
        Translation applied to each channel.  Channel 0 entry is ignored.
        None means no shift for all channels.
    rotations : list, length n_channels
        Rotation in degrees per channel: a scalar (about the z/optical axis) or
        a ``(rz, ry, rx)`` Euler triple.  None means 0 for all.
    scales : list of (sz, sy, sx), length n_channels
        Anisotropic scale applied to each channel.  None means (1, 1, 1).
    offset : int
        Camera baseline offset in absolute counts added before noise.  Must be
        an int in [0, 2**bit_depth - 1].  Default 10.
    snr : float
        Signal-to-noise ratio at the bead peak (peak / readout_noise_std).
        Poisson shot noise is always added on top.  Use np.inf for no noise.
    seed : int or None
        NumPy random seed for reproducibility.

    Returns
    -------
    stack : NDArray, shape (n_channels, Z, Y, X), dtype uint8 or uint16
        Multi-channel bead volume.
    ground_truth : dict
        Keys 'centers' ((N, 3) in z, y, x), 'shifts', 'rotations', 'scales',
        'bead_sigma', 'bit_depth', 'peak_value'.
    """
    if not (0.0 < bead_intensity <= 100.0):
        raise ValueError("bead_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")

    max_val = float(2**bit_depth - 1)
    if not isinstance(offset, int):
        raise TypeError("offset must be an int")
    if not (0 <= offset <= int(max_val)):
        raise ValueError(
            f"offset ({offset}) must be between 0 and {int(max_val)} "
            f"for {bit_depth}-bit images"
        )
    peak = max_val * bead_intensity / 100.0
    offset_counts = float(offset)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    rng = np.random.default_rng(seed)
    Z, Y, X = shape
    sigma = _as_sigma_tuple(bead_sigma, 3)

    # --- defaults (None = identity, no transform) ------------------------
    if shifts is None:
        shifts = [(0.0, 0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0, 1.0)] * n_channels

    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    # --- bead centres in channel-0 frame (z, y, x) -----------------------
    margins = [int(4 * s) for s in sigma]
    centers = np.column_stack(
        [
            rng.uniform(margins[0], Z - margins[0], size=n_beads),
            rng.uniform(margins[1], Y - margins[1], size=n_beads),
            rng.uniform(margins[2], X - margins[2], size=n_beads),
        ]
    )

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, Z, Y, X), dtype=np.float64)

    for ch in range(n_channels):
        ref_vol = _make_bead_image_3d(shape, centers, intensity=peak, sigma=sigma)

        dz, dy, dx = shifts[ch]
        rot = rotations[ch]
        sz, sy, sx = scales[ch]
        rot_is_zero = (
            rot == 0.0 if isinstance(rot, (int, float)) else all(a == 0.0 for a in rot)
        )
        is_identity = (
            dz == 0.0
            and dy == 0.0
            and dx == 0.0
            and rot_is_zero
            and sz == 1.0
            and sy == 1.0
            and sx == 1.0
        )

        if is_identity:
            ch_vol = ref_vol.astype(np.float64)
        else:
            a_fwd = np.diag([sz, sy, sx]) @ _rotation_matrix_3d(rot)
            t_fwd = np.array([dz, dy, dx], dtype=np.float64)

            a_inv = np.linalg.inv(a_fwd)
            centre = np.array([Z / 2.0, Y / 2.0, X / 2.0])
            affine_offset = centre - a_inv @ centre - a_inv @ t_fwd

            ch_vol = affine_transform(
                ref_vol.astype(np.float64),
                a_inv,
                offset=affine_offset,
                output_shape=shape,
                order=3,
                mode="constant",
                cval=0.0,
            )

        ch_vol = ch_vol + offset_counts
        if np.isfinite(snr):
            ch_vol = rng.poisson(ch_vol).astype(np.float64)
            if readout_std > 0.0:
                ch_vol += rng.standard_normal(shape) * readout_std

        stack[ch] = ch_vol

    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "centers": centers,
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "bead_sigma": sigma,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth
