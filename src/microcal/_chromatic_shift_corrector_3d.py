"""
Chromatic shift correction for 3-D (volumetric, multi-channel) microscopy.

:class:`ChromaticShiftCorrector3D` mirrors the 2-D
:class:`~microcal.ChromaticShiftCorrector` for z-stacks shaped ``(C, Z, Y, X)``.
The detection / matching / RANSAC pipeline is inherited unchanged from
:class:`~microcal._base_corrector._BaseChromaticShiftCorrector`; this class only
supplies the 3-D warp backend (``scipy.ndimage.affine_transform`` — skimage's
``warp`` is 2-D only) and 3-D visualisation arrays.

Anisotropy
----------
Real z-stacks are anisotropic (axial step ≫ lateral pixel).  ``smooth_sigma``
and ``refine_radius`` therefore accept either a scalar or a per-axis
``(sz, sy, sx)`` tuple, and an optional ``voxel_size=(z, y, x)`` makes bead
matching physically isotropic and reports residuals in physical units.  The
fitted transform itself always lives in voxel space, so the correction is
correct regardless of ``voxel_size``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.ndimage import affine_transform

from ._base_corrector import CorrectionResult, _BaseChromaticShiftCorrector

if TYPE_CHECKING:
    from numpy.typing import NDArray

__all__ = ["ChromaticShiftCorrector3D"]

logger = logging.getLogger(__name__)


class ChromaticShiftCorrector3D(_BaseChromaticShiftCorrector):
    """
    Estimate and apply chromatic shift corrections from 3-D bead volumes.

    The pipeline is identical to the 2-D corrector — Gaussian smoothing +
    local-maxima detection, intensity-weighted sub-pixel refinement, centroid
    coarse alignment, mutual nearest-neighbour matching and RANSAC fitting —
    generalised to volumes.  Transforms are 4x4 homogeneous matrices and are
    applied with ``scipy.ndimage.affine_transform``.
    """

    _spatial_ndim: ClassVar[int] = 3
    _ransac_min_samples: ClassVar[int] = 4
    _logger: ClassVar[logging.Logger] = logger

    # ------------------------------------------------------------------
    # Public method 1 — measure
    # ------------------------------------------------------------------

    def measure(
        self,
        bead_stack: NDArray,
        *,
        reference_channel: int = 0,
        transform_type: str = "affine",
        smooth_sigma: float | tuple[float, float, float] = 2.0,
        min_distance: int | tuple[int, int, int] = 10,
        threshold_rel: float = 0.1,
        match_max_distance: float = 10.0,
        min_pairs: int = 4,
        subpixel_refine: bool = True,
        refine_radius: int | tuple[int, int, int] = 5,
        voxel_size: tuple[float, float, float] | None = None,
        verbose: bool = False,
    ) -> CorrectionResult:
        """
        Estimate the chromatic shift from a multi-channel bead volume.

        Parameters
        ----------
        bead_stack : NDArray, shape (C, Z, Y, X)
            Multi-channel bead volume (float or uint).
        reference_channel : int
            Channel index used as the geometric reference.  Default 0.
        transform_type : str
            Transform to fit: 'affine' (default), 'similarity', 'euclidean' or
            'translation'.
        smooth_sigma : float or (sz, sy, sx)
            Sigma (voxels) of the Gaussian pre-smoothing.  A tuple sets a
            different value per axis — useful for anisotropic stacks.  Default 2.
        min_distance : int or (mz, my, mx)
            Minimum voxel distance between accepted peaks.  A tuple builds an
            anisotropic exclusion footprint (e.g. a smaller axial separation).
            Default 10.
        threshold_rel : float
            Minimum smoothed peak intensity as a fraction of the volume maximum.
            Default 0.1.
        match_max_distance : float
            Maximum distance for a valid bead pair, in voxels (or physical units
            when ``voxel_size`` is given).  Default 10.
        min_pairs : int
            Minimum matched pairs before fitting; below this the corrector falls
            back to translation-only.  Default 4.
        subpixel_refine : bool
            If True (default), refine each peak with an intensity-weighted
            centroid.
        refine_radius : int or (rz, ry, rx)
            Half-width (voxels) of the centroid-refinement patch.  Default 5.
        voxel_size : (z, y, x) or None
            Physical voxel size.  When given, bead matching uses a physically
            isotropic metric and ``validate()`` additionally reports residuals
            in physical units.  ``None`` (default) ⇒ pure voxel units.
        verbose : bool
            If True, emit progress logs.  Default False.

        Returns
        -------
        CorrectionResult
            One ChannelTransform (4x4 matrix) per non-reference channel, plus:

            * ``result.detection_image`` — ``(2*C, Z, Y, X)`` float32, normalised
              channel volumes and filled-sphere bead masks interleaved.
            * ``result.pairs_image`` — ``(Z, Y, X)`` int32 label volume where
              every bead in a matched group shares the same non-zero integer.
        """
        if voxel_size is not None and len(voxel_size) != 3:
            raise ValueError("voxel_size must have 3 entries (z, y, x)")

        # Store params so validate(), save() and private helpers can read them.
        self.transform_type = transform_type
        self.smooth_sigma = smooth_sigma
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
        self.match_max_distance = match_max_distance
        self.min_pairs = min_pairs
        self.subpixel_refine = subpixel_refine
        self.refine_radius = refine_radius
        self.voxel_size = tuple(voxel_size) if voxel_size is not None else None
        self.verbose = verbose

        self._setup_logger(verbose)

        if bead_stack.ndim != 4:
            raise ValueError("bead_stack must be 4-D (C, Z, Y, X)")

        return self._fit_all(bead_stack, reference_channel, transform_type)

    # ------------------------------------------------------------------
    # Public method 2 — apply
    # ------------------------------------------------------------------

    def apply(
        self,
        image_or_stack: NDArray | list[NDArray] | list[str],
        result: CorrectionResult | None = None,
        *,
        crop: bool = True,
    ) -> NDArray:
        """
        Apply the estimated correction to a volume or stack of volumes.

        Each non-reference channel is remapped into the reference frame with
        tricubic interpolation (order=3) via ``scipy.ndimage.affine_transform``.
        Voxels outside the source volume are filled with 0.

        Parameters
        ----------
        image_or_stack : NDArray (C, Z, Y, X) | list[NDArray] | list[str]
            The multi-channel volume to correct.  Accepted forms:

            * ``NDArray`` shape ``(C, Z, Y, X)`` — channel-first stack.
            * ``list[NDArray]`` — one 3-D ``(Z, Y, X)`` volume per channel.
            * ``list[str]`` — one file path per channel (read with tifffile).

        result : CorrectionResult or None
            Use a previously computed result.  Defaults to the stored result.
        crop : bool
            If True (default), crop to the largest box containing valid data in
            every channel.  If False, keep the input size with zero-filled
            borders.

        Returns
        -------
        NDArray
            Corrected stack, shape (C, Z, Y, X) or (C, Z', Y', X') if crop=True.
        """
        result = self._resolve_result(result)
        image_or_stack = self._coerce_stack(image_or_stack)
        orig_dtype = image_or_stack.dtype

        if image_or_stack.ndim != 4:
            raise ValueError("image_or_stack must be 4-D (C, Z, Y, X)")

        n_channels = image_or_stack.shape[0]
        spatial = image_or_stack.shape[1:]
        corrected = image_or_stack.copy().astype(np.float64)

        valid_mask = np.ones(spatial, dtype=bool)
        ones = np.ones(spatial, dtype=np.float64)

        for ch in range(n_channels):
            if ch == result.reference_channel or ch not in result.transforms:
                continue

            tform = result.transforms[ch].transform
            matrix, offset = self._ndimage_params(tform)
            corrected[ch] = affine_transform(
                image_or_stack[ch].astype(np.float64),
                matrix,
                offset=offset,
                order=3,
                mode="constant",
                cval=0.0,
            )

            if crop:
                ch_valid = affine_transform(
                    ones, matrix, offset=offset, order=0, mode="constant", cval=0.0
                )
                valid_mask &= ch_valid > 0.5

        if crop:
            corrected = self._crop_to_valid(corrected, valid_mask)

        return corrected.astype(orig_dtype)

    @staticmethod
    def _ndimage_params(tform: object) -> tuple[NDArray, NDArray]:
        """Convert a skimage 4x4 transform to ``affine_transform`` arguments.

        ``scipy.ndimage.affine_transform`` is a *pull* map (output → input) and
        indexes in array order ``(z, y, x)``, whereas the fitted transform is in
        ``(x, y, z)`` order.  We take the inverse transform (ref → ch) and
        reverse the spatial axes: ``matrix = L[::-1, ::-1]`` and
        ``offset = t[::-1]`` (conjugation by the axis-reversal permutation).
        """
        inv = np.asarray(tform.inverse.params, dtype=np.float64)  # type: ignore[attr-defined]
        d = inv.shape[0] - 1
        linear = inv[:d, :d]
        translation = inv[:d, d]
        matrix = np.ascontiguousarray(linear[::-1, ::-1])
        offset = np.ascontiguousarray(translation[::-1])
        return matrix, offset

    @staticmethod
    def _crop_to_valid(arr: NDArray, valid_mask: NDArray) -> NDArray:
        """Crop ``(C, *spatial)`` ``arr`` to the bounding box of ``valid_mask``."""
        slices: list[slice] = []
        ndim = valid_mask.ndim
        for ax in range(ndim):
            other = tuple(i for i in range(ndim) if i != ax)
            present = np.where(valid_mask.any(axis=other))[0]
            if present.size == 0:
                return arr
            slices.append(slice(int(present[0]), int(present[-1]) + 1))
        return arr[(slice(None), *slices)]

    # ------------------------------------------------------------------
    # Residual reporting (adds per-axis / physical errors)
    # ------------------------------------------------------------------

    def _residual_stats(self, src: NDArray, dst: NDArray) -> dict:
        stats = super()._residual_stats(src, dst)
        if len(src) > 0:
            # per-axis mean absolute error in (z, y, x) voxels
            stats["mean_error_per_axis"] = np.abs(src - dst).mean(axis=0)
            if self.voxel_size is not None:
                phys = np.linalg.norm((src - dst) * np.asarray(self.voxel_size), axis=1)
                stats["mean_error_physical"] = float(np.mean(phys))
                stats["median_error_physical"] = float(np.median(phys))
                stats["max_error_physical"] = float(np.max(phys))
        return stats

    # ------------------------------------------------------------------
    # Visualisation array builders (3-D)
    # ------------------------------------------------------------------

    def _build_detection_image(
        self,
        bead_stack: NDArray,
        centers_per_ch: dict[int, NDArray],
    ) -> NDArray:
        """Return (2*C, Z, Y, X) float32: interleaved volumes and bead masks."""
        C = bead_stack.shape[0]
        spatial = bead_stack.shape[1:]
        vis = np.zeros((2 * C, *spatial), dtype=np.float32)
        radius = self._mask_radius()
        for ch in range(C):
            vis[2 * ch] = self._normalise(bead_stack[ch]).astype(np.float32)
            if ch in centers_per_ch and len(centers_per_ch[ch]) > 0:
                self._paint_spheres(vis[2 * ch + 1], centers_per_ch[ch], radius, 1.0)
        return vis

    def _build_pairs_image(
        self,
        bead_stack: NDArray,
        pairs_per_ch: dict[int, tuple[NDArray, NDArray]],
    ) -> NDArray:
        """Return (Z, Y, X) int32: matched groups share a non-zero integer label."""
        spatial = bead_stack.shape[1:]
        labels = np.zeros(spatial, dtype=np.int32)
        radius = self._mask_radius()
        ref_to_label: dict[tuple[int, ...], int] = {}
        next_label = 1
        for _ch, (src, dst) in pairs_per_ch.items():
            for s, d in zip(src, dst, strict=False):
                key = tuple(round(v) for v in s)
                if key not in ref_to_label:
                    ref_to_label[key] = next_label
                    next_label += 1
                lbl = ref_to_label[key]
                self._paint_spheres(labels, s[None, :], radius, float(lbl))
                self._paint_spheres(labels, d[None, :], radius, float(lbl))
        return labels

    def _mask_radius(self) -> int:
        return max(2, round(float(np.min(np.atleast_1d(self.smooth_sigma)))))

    @staticmethod
    def _sphere_offsets(radius: int) -> NDArray:
        rng = np.arange(-radius, radius + 1)
        gz, gy, gx = np.meshgrid(rng, rng, rng, indexing="ij")
        offs = np.stack([gz.ravel(), gy.ravel(), gx.ravel()], axis=1)
        keep = (offs**2).sum(axis=1) <= radius * radius
        return offs[keep]  # type: ignore[no-any-return]

    def _paint_spheres(
        self, vol: NDArray, centers: NDArray, radius: int, value: float
    ) -> None:
        """Stamp filled spheres of ``value`` at each ``(z, y, x)`` centre."""
        offsets = self._sphere_offsets(radius)
        shape = np.array(vol.shape)
        for c in centers:
            pts = np.round(np.asarray(c)).astype(int) + offsets  # (M, 3)
            inb = np.all((pts >= 0) & (pts < shape), axis=1)
            pts = pts[inb]
            vol[pts[:, 0], pts[:, 1], pts[:, 2]] = value
