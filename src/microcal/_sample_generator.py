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
    intensities: NDArray,
    sigmas: NDArray,
) -> NDArray:
    """Render a 2-D image of Gaussian spots with per-bead intensity and PSF size.

    Uses direct analytic Gaussian placement with a local bounding box per spot,
    so each bead can have its own peak intensity and sigma.
    """
    img = np.zeros(shape, dtype=np.float32)
    for (y0, x0), s, amp in zip(centers, sigmas, intensities, strict=True):
        r = int(s * 4)
        ys = slice(max(0, round(y0) - r), min(shape[0], round(y0) + r + 1))
        xs = slice(max(0, round(x0) - r), min(shape[1], round(x0) + r + 1))
        yg = np.arange(ys.start, ys.stop, dtype=np.float64) - y0
        xg = np.arange(xs.start, xs.stop, dtype=np.float64) - x0
        YY, XX = np.meshgrid(yg, xg, indexing="ij")
        img[ys, xs] += amp * np.exp(-0.5 * ((YY / s) ** 2 + (XX / s) ** 2))
    return img


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
    # Physical PSF parameters — when na + pixel_size are set, sigma is derived
    # from optics and overrides bead_sigma
    pixel_size: float | None = None,
    na: float | None = None,
    em_wvl_um: float = 0.520,
    # Per-bead variability
    sigma_scale_range: tuple[float, float] | None = None,
    intensity_range: tuple[float, float] | None = None,
    # Autofluorescence background
    background: float = 0.0,
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
        Ignored when `na` and `pixel_size` are both set.
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
    offset : int
        Camera baseline offset in absolute pixel counts added to every pixel
        before noise, mimicking dark current / electronics bias.  Must be an
        int in [0, 2**bit_depth - 1].  Default 10.
    snr : float
        Signal-to-noise ratio at the bead peak, defined as
        peak / readout_noise_std.  Controls the Gaussian readout noise floor
        (std = peak / snr) applied uniformly to every pixel.  On top of that,
        Poisson shot noise (std ≈ sqrt(signal)) is always added — it scales
        with signal so it affects bright beads more than the dark background.
        Use np.inf to suppress all noise (noiseless simulation).
    seed : int or None
        NumPy random seed for reproducibility.
    pixel_size : float, optional
        Physical pixel size in µm.  Required together with `na` to derive
        the PSF sigma from optics instead of using `bead_sigma`.
    na : float, optional
        Objective numerical aperture.  When set alongside `pixel_size`, PSF
        sigma is computed as `sigma_xy = 0.21 * em_wvl_um / na`, then
        converted to pixels.
    em_wvl_um : float
        Emission wavelength in µm.  Default 0.520 (GFP).
    sigma_scale_range : (lo, hi), optional
        Per-bead random size jitter: each bead's sigma is multiplied by a
        factor drawn uniformly from `[lo, hi]`.  Requires `0 < lo <= hi`.
        None means all beads have the same sigma.
    intensity_range : (lo, hi), optional
        Per-bead intensity as a fraction of peak, drawn uniformly from
        `[lo, hi]`.  E.g. `(0.5, 1.0)` gives beads between 50 % and 100 %
        of peak intensity.  None means all beads are at peak intensity.
    background : float
        Smooth autofluorescence background level as a fraction of peak
        intensity (e.g. 0.05 = 5 % of peak).  A spatially smooth random field
        is added per channel before noise.  Default 0.0 (no background).

    Returns
    -------
    stack : NDArray, shape (n_channels, H, W), dtype uint8 or uint16
        Multi-channel bead image.
    ground_truth : dict
        Keys 'centers', 'intensities_per_bead' (N,), 'sigmas_per_bead' (N,),
        'shifts', 'rotations', 'scales', 'bead_sigma', 'bit_depth',
        'peak_value'.
    """
    if not (0.0 < bead_intensity <= 100.0):
        raise ValueError("bead_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    if sigma_scale_range is not None:
        lo, hi = sigma_scale_range
        if lo <= 0 or lo > hi:
            raise ValueError("sigma_scale_range must satisfy 0 < lo <= hi")
    if intensity_range is not None:
        lo, hi = intensity_range
        if lo < 0 or lo > hi:
            raise ValueError("intensity_range must satisfy 0 <= lo <= hi")
    if background < 0.0:
        raise ValueError("background must be >= 0")

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

    # --- PSF sigma: derive from optics or use raw pixel value --------------
    if na is not None and pixel_size is not None:
        sigma = 0.21 * em_wvl_um / na / pixel_size
    else:
        sigma = float(bead_sigma)

    # --- defaults (None = identity, no transform) --------------------------
    if shifts is None:
        shifts = [(0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0)] * n_channels

    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    # --- bead centres in channel-0 frame ----------------------------------
    margin_sigma = (
        sigma * max(sigma_scale_range) if sigma_scale_range is not None else sigma
    )
    margin = int(margin_sigma * 4)
    centers = np.column_stack(
        [
            rng.uniform(margin, H - margin, size=n_beads),
            rng.uniform(margin, W - margin, size=n_beads),
        ]
    )

    # --- per-bead sigmas and intensities -----------------------------------
    if sigma_scale_range is not None:
        sigmas_per_bead = sigma * rng.uniform(*sigma_scale_range, size=n_beads)
    else:
        sigmas_per_bead = np.full(n_beads, sigma)

    if intensity_range is not None:
        intensities_per_bead = peak * rng.uniform(*intensity_range, size=n_beads)
    else:
        intensities_per_bead = np.full(n_beads, peak)

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, H, W), dtype=np.float64)

    for ch in range(n_channels):
        ref_img = _make_bead_image(
            shape, centers, intensities_per_bead, sigmas_per_bead
        )

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

        if background > 0.0:
            bg = gaussian_filter(rng.uniform(0, 1, shape).astype(np.float32), sigma=20)
            ch_img = ch_img + bg.astype(np.float64) / bg.max() * (background * peak)

        ch_img = ch_img + offset_counts
        if np.isfinite(snr):
            ch_img = rng.poisson(ch_img).astype(np.float64)
            if readout_std > 0.0:
                ch_img += rng.standard_normal(shape) * readout_std

        stack[ch] = ch_img

    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "centers": centers,
        "intensities_per_bead": intensities_per_bead,
        "sigmas_per_bead": sigmas_per_bead,
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "bead_sigma": sigma,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth


# ---------------------------------------------------------------------------
# 3-D (volumetric) bead generation
# ---------------------------------------------------------------------------


def _as_sigma_tuple(sigma: float | tuple[float, ...], n: int) -> tuple[float, ...]:
    """Coerce a scalar or per-axis sigma into an `n`-tuple of floats."""
    if isinstance(sigma, (int, float)):
        return (float(sigma),) * n
    t = tuple(float(s) for s in sigma)
    if len(t) != n:
        raise ValueError(f"sigma must have {n} entries, got {len(t)}")
    return t


def _rotation_matrix_3d(rot: float | tuple[float, float, float]) -> NDArray:
    """Build a 3x3 rotation matrix (array order z, y, x) from degrees.

    `rot` may be a scalar (rotation about the optical/z axis, i.e. in the
    y-x plane — the direct analogue of the 2-D `rotations` argument) or a
    `(rz, ry, rx)` tuple of Euler angles composed as `Rz @ Ry @ Rx`.
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
    intensities: NDArray,
    sigmas: NDArray,
) -> NDArray:
    """Render a 3-D volume of Gaussian spots with per-bead intensity and PSF size.

    Uses direct analytic Gaussian placement with a local bounding box per spot,
    so each bead can have its own peak intensity and sigma.
    """
    volume = np.zeros(shape, dtype=np.float32)
    for (z0, y0, x0), (sz, sy, sx), amp in zip(
        centers, sigmas, intensities, strict=True
    ):
        r_z, r_y, r_x = int(sz * 4), int(sy * 4), int(sx * 4)
        zs = slice(max(0, round(z0) - r_z), min(shape[0], round(z0) + r_z + 1))
        ys = slice(max(0, round(y0) - r_y), min(shape[1], round(y0) + r_y + 1))
        xs = slice(max(0, round(x0) - r_x), min(shape[2], round(x0) + r_x + 1))
        zg = np.arange(zs.start, zs.stop, dtype=np.float64) - z0
        yg = np.arange(ys.start, ys.stop, dtype=np.float64) - y0
        xg = np.arange(xs.start, xs.stop, dtype=np.float64) - x0
        ZZ, YY, XX = np.meshgrid(zg, yg, xg, indexing="ij")
        volume[zs, ys, xs] += amp * np.exp(
            -0.5 * ((ZZ / sz) ** 2 + (YY / sy) ** 2 + (XX / sx) ** 2)
        )
    return volume


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
    # Physical PSF parameters — when na + voxel_size are set, sigma is derived
    # from optics and overrides bead_sigma
    voxel_size: tuple[float, float, float] | None = None,
    na: float | None = None,
    ri: float = 1.0,
    em_wvl_um: float = 0.520,
    # Per-bead variability
    sigma_scale_range: tuple[float, float] | None = None,
    intensity_range: tuple[float, float] | None = None,
    # Autofluorescence background
    background: float = 0.0,
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
        (real PSFs have sz > sxy).  Ignored when `na` and `voxel_size` are
        both set.
    bead_intensity : float
        Peak bead intensity as a percentage of the full dynamic range (0-100).
    bit_depth : int
        Camera bit depth (8 or 16).  Determines the output dtype and max value.
    shifts : list of (dz, dy, dx), length n_channels
        Translation applied to each channel.  Channel 0 entry is ignored.
        None means no shift for all channels.
    rotations : list, length n_channels
        Rotation in degrees per channel: a scalar (about the z/optical axis) or
        a `(rz, ry, rx)` Euler triple.  None means 0 for all.
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
    voxel_size : (dz, dy, dx) in µm, optional
        Physical voxel size.  Required together with `na` to derive the PSF
        sigma from optics instead of using `bead_sigma`.
    na : float, optional
        Objective numerical aperture.  When set alongside `voxel_size`, PSF
        sigmas are computed as:
        `sigma_xy = 0.21 * em_wvl_um / na` and
        `sigma_z = 0.45 * em_wvl_um * ri / na**2`, then converted to voxels.
    ri : float
        Refractive index of the immersion medium.  Default 1.0 (air).
    em_wvl_um : float
        Emission wavelength in µm.  Default 0.520 (GFP).
    sigma_scale_range : (lo, hi), optional
        Per-bead random size jitter: each bead's sigma is multiplied by a
        factor drawn uniformly from `[lo, hi]`.  Requires `0 < lo <= hi`.
        None means all beads have the same sigma.
    intensity_range : (lo, hi), optional
        Per-bead intensity as a fraction of peak, drawn uniformly from
        `[lo, hi]`.  E.g. `(0.5, 1.0)` gives beads between 50 % and 100 %
        of peak intensity.  None means all beads are at peak intensity.
    background : float
        Smooth autofluorescence background level as a fraction of peak intensity
        (e.g. 0.05 = 5 % of peak).  A spatially smooth random field is added
        per channel before noise.  Default 0.0 (no background).

    Returns
    -------
    stack : NDArray, shape (n_channels, Z, Y, X), dtype uint8 or uint16
        Multi-channel bead volume.
    ground_truth : dict
        Keys 'centers' ((N, 3) in z, y, x), 'intensities_per_bead' (N,),
        'sigmas_per_bead' ((N, 3)), 'shifts', 'rotations', 'scales',
        'bead_sigma', 'bit_depth', 'peak_value'.
    """
    if not (0.0 < bead_intensity <= 100.0):
        raise ValueError("bead_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    if sigma_scale_range is not None:
        lo, hi = sigma_scale_range
        if lo <= 0 or lo > hi:
            raise ValueError("sigma_scale_range must satisfy 0 < lo <= hi")
    if intensity_range is not None:
        lo, hi = intensity_range
        if lo < 0 or lo > hi:
            raise ValueError("intensity_range must satisfy 0 <= lo <= hi")
    if background < 0.0:
        raise ValueError("background must be >= 0")

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

    # --- PSF sigma: derive from optics or use raw voxel value --------------
    if na is not None and voxel_size is not None:
        dz, _, dx = voxel_size
        sigma_xy_um = 0.21 * em_wvl_um / na
        sigma_z_um = 0.45 * em_wvl_um * ri / na**2
        sigma: tuple[float, ...] = (sigma_z_um / dz, sigma_xy_um / dx, sigma_xy_um / dx)
    else:
        sigma = _as_sigma_tuple(bead_sigma, 3)

    # --- defaults (None = identity, no transform) --------------------------
    if shifts is None:
        shifts = [(0.0, 0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0, 1.0)] * n_channels

    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    # --- bead centres in channel-0 frame (z, y, x) ------------------------
    # margins use the maximum possible sigma (after jitter) to keep beads in-frame
    if sigma_scale_range is not None:
        margin_sigma = tuple(s * max(sigma_scale_range) for s in sigma)
    else:
        margin_sigma = sigma
    margins = [int(4 * s) for s in margin_sigma]
    centers = np.column_stack(
        [
            rng.uniform(margins[0], Z - margins[0], size=n_beads),
            rng.uniform(margins[1], Y - margins[1], size=n_beads),
            rng.uniform(margins[2], X - margins[2], size=n_beads),
        ]
    )

    # --- per-bead sigmas and intensities -----------------------------------
    sigma_arr = np.array(sigma, dtype=np.float64)
    if sigma_scale_range is not None:
        jitter = rng.uniform(*sigma_scale_range, size=n_beads)
        sigmas_per_bead = np.outer(jitter, sigma_arr)
    else:
        sigmas_per_bead = np.tile(sigma_arr, (n_beads, 1))

    if intensity_range is not None:
        intensities_per_bead = peak * rng.uniform(*intensity_range, size=n_beads)
    else:
        intensities_per_bead = np.full(n_beads, peak)

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, Z, Y, X), dtype=np.float64)

    for ch in range(n_channels):
        ref_vol = _make_bead_image_3d(
            shape, centers, intensities_per_bead, sigmas_per_bead
        )

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

        if background > 0.0:
            bg = gaussian_filter(rng.uniform(0, 1, shape).astype(np.float32), sigma=20)
            ch_vol = ch_vol + bg.astype(np.float64) / bg.max() * (background * peak)

        ch_vol = ch_vol + offset_counts
        if np.isfinite(snr):
            ch_vol = rng.poisson(ch_vol).astype(np.float64)
            if readout_std > 0.0:
                ch_vol += rng.standard_normal(shape) * readout_std

        stack[ch] = ch_vol

    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "centers": centers,
        "intensities_per_bead": intensities_per_bead,
        "sigmas_per_bead": sigmas_per_bead,
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "bead_sigma": sigma,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth


# ---------------------------------------------------------------------------
# Continuous-structure generation (for the marker-free correlation method)
# ---------------------------------------------------------------------------


def _affine_warp_2d(
    base: NDArray,
    shape: tuple[int, int],
    dy: float,
    dx: float,
    rot: float,
    sy: float,
    sx: float,
) -> NDArray:
    """Warp a 2-D image by a per-channel affine about the image centre.

    Mirrors the per-channel transform used by :func:`generate_beads_image`.
    """
    if dy == 0.0 and dx == 0.0 and rot == 0.0 and sy == 1.0 and sx == 1.0:
        return base.astype(np.float64)
    h, w = shape
    theta = np.deg2rad(rot)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    a_fwd = np.array([[sy * cos_t, sy * sin_t], [-sx * sin_t, sx * cos_t]])
    a_inv = np.linalg.inv(a_fwd)
    centre = np.array([h / 2.0, w / 2.0])
    affine_offset = centre - a_inv @ centre - a_inv @ np.array([dy, dx])
    return affine_transform(
        base.astype(np.float64),
        a_inv,
        offset=affine_offset,
        output_shape=shape,
        order=3,
        mode="constant",
        cval=0.0,
    )


def generate_structures_image(
    n_channels: int = 2,
    shape: tuple[int, int] = (512, 512),
    structure_scale: float = 8.0,
    contour_width: float = 0.3,
    structure_intensity: float = 60.0,
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
    background: float = 0.0,
) -> tuple[NDArray, dict]:
    """
    Generate a synthetic multi-channel *continuous-structure* image.

    Unlike :func:`generate_beads_image`, the signal is a continuous,
    filament/membrane-like network — the bright level-set contours of a smoothed
    random field — mimicking a sample where the same construct (e.g.
    mitochondria) is stained in several colours.  Such samples have no isolated
    point maxima, so they are the target use-case for the marker-free
    `method="correlation"` calibration.  Per-channel chromatic shift, rotation
    and scale are applied exactly as in :func:`generate_beads_image`.

    Parameters
    ----------
    n_channels : int
        Number of fluorescence channels.
    shape : (H, W)
        Spatial size of each channel image.
    structure_scale : float
        Smoothing sigma (px) of the underlying random field; larger values give
        coarser/thicker structures.  Default 8.
    contour_width : float
        Width of the bright contours as a fraction of the field's standard
        deviation; larger values give denser, thicker structures.  Default 0.3.
    structure_intensity : float
        Peak structure intensity as a percentage of the full dynamic range
        (0-100).  Default 60.
    bit_depth : int
        Camera bit depth (8 or 16); determines the output dtype and max value.
    shifts : list of (dy, dx), length n_channels
        Translation applied to each channel.  Channel 0 entry is ignored.
        None means no shift for all channels.
    rotations : list of float, length n_channels
        Rotation in degrees applied to each channel.  None means 0 for all.
    scales : list of (sy, sx), length n_channels
        Anisotropic scale applied to each channel.  None means (1, 1) for all.
    offset : int
        Camera baseline offset in absolute counts added before noise.  Must be
        an int in [0, 2**bit_depth - 1].  Default 10.
    snr : float
        Signal-to-noise ratio at the structure peak (peak / readout_noise_std).
        Poisson shot noise is always added on top.  Use np.inf for no noise.
    seed : int or None
        NumPy random seed for reproducibility.
    background : float
        Smooth autofluorescence background level as a fraction of peak intensity.
        Default 0.0.

    Returns
    -------
    stack : NDArray, shape (n_channels, H, W), dtype uint8 or uint16
        Multi-channel continuous-structure image.
    ground_truth : dict
        Keys 'shifts', 'rotations', 'scales', 'structure_scale',
        'contour_width', 'bit_depth', 'peak_value'.
    """
    if not (0.0 < structure_intensity <= 100.0):
        raise ValueError("structure_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    if contour_width <= 0.0:
        raise ValueError("contour_width must be > 0")
    if background < 0.0:
        raise ValueError("background must be >= 0")

    max_val = float(2**bit_depth - 1)
    if not isinstance(offset, int):
        raise TypeError("offset must be an int")
    if not (0 <= offset <= int(max_val)):
        raise ValueError(
            f"offset ({offset}) must be between 0 and {int(max_val)} "
            f"for {bit_depth}-bit images"
        )
    peak = max_val * structure_intensity / 100.0
    offset_counts = float(offset)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    rng = np.random.default_rng(seed)
    H, W = shape

    if shifts is None:
        shifts = [(0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0)] * n_channels
    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    # Base structure in the channel-0 frame: bright contours of a smooth field.
    field = gaussian_filter(rng.standard_normal(shape), sigma=structure_scale)
    std = float(field.std()) or 1.0
    base = np.exp(-0.5 * (field / (std * contour_width)) ** 2) * peak

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, H, W), dtype=np.float64)

    for ch in range(n_channels):
        dy, dx = shifts[ch]
        sy, sx = scales[ch]
        ch_img = _affine_warp_2d(base, shape, dy, dx, rotations[ch], sy, sx)

        if background > 0.0:
            bg = gaussian_filter(rng.uniform(0, 1, shape).astype(np.float32), sigma=20)
            ch_img = ch_img + bg.astype(np.float64) / bg.max() * (background * peak)

        # Cubic warp can undershoot below 0 at high-contrast contour edges;
        # clip before Poisson sampling (intensities and lam must be >= 0).
        ch_img = np.clip(ch_img + offset_counts, 0.0, None)
        if np.isfinite(snr):
            ch_img = rng.poisson(ch_img).astype(np.float64)
            if readout_std > 0.0:
                ch_img += rng.standard_normal(shape) * readout_std
        stack[ch] = ch_img

    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "structure_scale": structure_scale,
        "contour_width": contour_width,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth


def _affine_warp_3d(
    base: NDArray,
    shape: tuple[int, int, int],
    shift: tuple[float, float, float],
    rot: float | tuple[float, float, float],
    scale: tuple[float, float, float],
) -> NDArray:
    """Warp a 3-D volume by a per-channel affine about the volume centre.

    Mirrors the per-channel transform used by :func:`generate_beads_image_3d`.
    """
    dz, dy, dx = shift
    sz, sy, sx = scale
    rot_is_zero = (
        rot == 0.0 if isinstance(rot, (int, float)) else all(a == 0.0 for a in rot)
    )
    if dz == 0.0 and dy == 0.0 and dx == 0.0 and rot_is_zero and sz == sy == sx == 1.0:
        return base.astype(np.float64)
    a_fwd = np.diag([sz, sy, sx]) @ _rotation_matrix_3d(rot)
    a_inv = np.linalg.inv(a_fwd)
    centre = np.array([shape[0] / 2.0, shape[1] / 2.0, shape[2] / 2.0])
    affine_offset = centre - a_inv @ centre - a_inv @ np.array([dz, dy, dx])
    return affine_transform(
        base.astype(np.float64),
        a_inv,
        offset=affine_offset,
        output_shape=shape,
        order=3,
        mode="constant",
        cval=0.0,
    )


def generate_structures_image_3d(
    n_channels: int = 2,
    shape: tuple[int, int, int] = (32, 256, 256),
    structure_scale: float | tuple[float, float, float] = (3.0, 8.0, 8.0),
    contour_width: float = 0.3,
    structure_intensity: float = 60.0,
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
    background: float = 0.0,
) -> tuple[NDArray, dict]:
    """
    Generate a synthetic multi-channel *continuous-structure* volume.

    The 3-D analogue of :func:`generate_structures_image`: the signal is the
    bright level-set surfaces (membrane-like sheets) of a smoothed 3-D random
    field, warped per channel by a 3-D affine (shift + rotation + anisotropic
    scale).  This is the target sample for marker-free `method="correlation"`
    calibration of volumetric chromatic shift.

    Parameters
    ----------
    n_channels : int
        Number of fluorescence channels.
    shape : (Z, Y, X)
        Spatial size of each channel volume.
    structure_scale : float or (sz, sy, sx)
        Smoothing sigma (voxels) of the underlying random field; a tuple sets a
        per-axis value (the default uses a smaller axial sigma for typical
        anisotropic stacks).  Default `(3, 8, 8)`.
    contour_width : float
        Width of the bright surfaces as a fraction of the field's standard
        deviation.  Default 0.3.
    structure_intensity : float
        Peak intensity as a percentage of the full dynamic range (0-100).
        Default 60.
    bit_depth : int
        Camera bit depth (8 or 16).
    shifts : list of (dz, dy, dx), length n_channels
        Translation per channel.  Channel 0 entry is ignored.  None ⇒ no shift.
    rotations : list, length n_channels
        Rotation in degrees per channel: a scalar (about the z/optical axis) or a
        `(rz, ry, rx)` Euler triple.  None means 0 for all.
    scales : list of (sz, sy, sx), length n_channels
        Anisotropic scale per channel.  None means (1, 1, 1).
    offset : int
        Camera baseline offset in absolute counts added before noise.  Default 10.
    snr : float
        Signal-to-noise ratio at the structure peak.  Use np.inf for no noise.
    seed : int or None
        NumPy random seed for reproducibility.
    background : float
        Smooth autofluorescence background as a fraction of peak.  Default 0.0.

    Returns
    -------
    stack : NDArray, shape (n_channels, Z, Y, X), dtype uint8 or uint16
        Multi-channel continuous-structure volume.
    ground_truth : dict
        Keys 'shifts', 'rotations', 'scales', 'structure_scale',
        'contour_width', 'bit_depth', 'peak_value'.
    """
    if not (0.0 < structure_intensity <= 100.0):
        raise ValueError("structure_intensity must be in (0, 100]")
    if bit_depth not in (8, 16):
        raise ValueError("bit_depth must be 8 or 16")
    if contour_width <= 0.0:
        raise ValueError("contour_width must be > 0")
    if background < 0.0:
        raise ValueError("background must be >= 0")

    max_val = float(2**bit_depth - 1)
    if not isinstance(offset, int):
        raise TypeError("offset must be an int")
    if not (0 <= offset <= int(max_val)):
        raise ValueError(
            f"offset ({offset}) must be between 0 and {int(max_val)} "
            f"for {bit_depth}-bit images"
        )
    peak = max_val * structure_intensity / 100.0
    offset_counts = float(offset)
    dtype = np.uint8 if bit_depth == 8 else np.uint16

    rng = np.random.default_rng(seed)
    Z, Y, X = shape

    if shifts is None:
        shifts = [(0.0, 0.0, 0.0)] * n_channels
    if rotations is None:
        rotations = [0.0] * n_channels
    if scales is None:
        scales = [(1.0, 1.0, 1.0)] * n_channels
    assert len(shifts) == n_channels, "shifts must have one entry per channel"
    assert len(rotations) == n_channels
    assert len(scales) == n_channels

    sigma = _as_sigma_tuple(structure_scale, 3)
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma)
    std = float(field.std()) or 1.0
    base = np.exp(-0.5 * (field / (std * contour_width)) ** 2) * peak

    readout_std = peak / snr if np.isfinite(snr) else 0.0
    stack = np.zeros((n_channels, Z, Y, X), dtype=np.float64)

    for ch in range(n_channels):
        ch_vol = _affine_warp_3d(base, shape, shifts[ch], rotations[ch], scales[ch])

        if background > 0.0:
            bg = gaussian_filter(rng.uniform(0, 1, shape).astype(np.float32), sigma=20)
            ch_vol = ch_vol + bg.astype(np.float64) / bg.max() * (background * peak)

        # Cubic warp can undershoot below 0 at high-contrast contour edges;
        # clip before Poisson sampling (intensities and lam must be >= 0).
        ch_vol = np.clip(ch_vol + offset_counts, 0.0, None)
        if np.isfinite(snr):
            ch_vol = rng.poisson(ch_vol).astype(np.float64)
            if readout_std > 0.0:
                ch_vol += rng.standard_normal(shape) * readout_std
        stack[ch] = ch_vol

    stack = np.clip(stack, offset_counts, max_val).astype(dtype)

    ground_truth = {
        "shifts": shifts,
        "rotations": rotations,
        "scales": scales,
        "structure_scale": sigma,
        "contour_width": contour_width,
        "bit_depth": bit_depth,
        "peak_value": peak,
    }
    return stack, ground_truth
