"""
Generate synthetic multi-channel fluorescence bead images with controllable chromatic shift.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, affine_transform


def _make_bead_image(
    shape: tuple[int, int],
    centers: NDArray,
    intensity: float = 1.0,
    sigma: float = 2.0,
) -> NDArray:
    """Render a 2D image of point-like beads convolved with a Gaussian PSF.

    The impulse is pre-scaled so that each bead's peak intensity after
    Gaussian blurring equals `intensity`.  For a 2-D Gaussian with std σ the
    integral is 1 and the peak is 1/(2πσ²), so we multiply by 2πσ² upfront.
    """
    # pre-scale so post-blur peak ≈ intensity
    impulse_value = float(intensity) * 2.0 * np.pi * sigma**2
    img = np.zeros(shape, dtype=np.float32)
    for cy, cx in centers:
        iy, ix = int(round(cy)), int(round(cx))
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
        raise ValueError(f"offset ({offset}) must be between 0 and {int(max_val)} for {bit_depth}-bit images")
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
