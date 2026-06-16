"""
Shared, dimension-agnostic machinery for chromatic shift correction.

This module holds the parts of the bead-registration pipeline that are
identical for 2-D ``(C, H, W)`` and 3-D ``(C, Z, Y, X)`` data: normalisation,
bead detection / sub-pixel refinement, mutual nearest-neighbour matching,
RANSAC transform fitting, validation, and JSON save/load.

The concrete classes :class:`~microcal.ChromaticShiftCorrector` (2-D) and
:class:`~microcal.ChromaticShiftCorrector3D` (3-D) subclass
:class:`_BaseChromaticShiftCorrector` and only override the genuinely
dimension-specific pieces (the warp backend used by :meth:`apply` and the
visualisation array builders).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import ransac
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
)

if TYPE_CHECKING:
    import os

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers (shared between 2-D and 3-D)
# ---------------------------------------------------------------------------


@dataclass
class ChannelTransform:
    """Affine transform that maps a channel into the reference frame."""

    channel: int
    # skimage transform — a (D+1)x(D+1) homogeneous matrix (3x3 in 2-D, 4x4 in 3-D)
    transform: AffineTransform
    # residual RMS in pixels/voxels after fitting (None if not computed)
    rms_residual: float | None = None
    # number of bead pairs used to fit the transform
    n_pairs: int = 0


@dataclass
class CorrectionResult:
    """Return value of ``measure()``."""

    reference_channel: int
    transforms: dict[int, ChannelTransform] = field(default_factory=dict)
    # Set by measure(): interleaved normalised images and filled bead masks for
    # every channel.  ``(2*C, H, W)`` in 2-D, ``(2*C, Z, Y, X)`` in 3-D.
    detection_image: NDArray | None = None
    # Set by measure(): int32 label image where every bead in a matched group
    # shares the same non-zero integer.  ``(H, W)`` in 2-D, ``(Z, Y, X)`` in 3-D.
    pairs_image: NDArray | None = None

    def __repr__(self) -> str:
        lines = [f"CorrectionResult(reference={self.reference_channel})"]
        for ch, ct in self.transforms.items():
            rms = (
                f"{ct.rms_residual:.3f}px"
                if ct.rms_residual is not None
                else "N/A (translation fallback — too few bead pairs)"
            )
            lines.append(
                f"  ch{ch}: rms={rms}  n_pairs={ct.n_pairs}\n"
                f"       matrix=\n{ct.transform.params}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class _BaseChromaticShiftCorrector:
    """
    Dimension-agnostic base for the 2-D and 3-D chromatic shift correctors.

    Subclasses set :attr:`_spatial_ndim` (2 or 3) and :attr:`_ransac_min_samples`
    (3 or 4), point :attr:`_logger` at their own module logger, and override the
    dimension-specific methods ``apply``, ``_build_detection_image`` and
    ``_build_pairs_image``.
    """

    # Number of spatial axes (2 for (C, H, W), 3 for (C, Z, Y, X)).
    _spatial_ndim: ClassVar[int] = 2
    # Minimal subset size for RANSAC (3 for a 2-D affine, 4 for a 3-D affine).
    _ransac_min_samples: ClassVar[int] = 3
    # Per-class logger so tests / users get sensibly named log records.
    _logger: ClassVar[logging.Logger] = logger

    def __init__(self) -> None:
        self._result: CorrectionResult | None = None
        self._bead_stack: NDArray | None = None
        # Matched bead pairs from the last measure() call, keyed by channel index.
        # Stored so validate() can apply the transform directly to known positions
        # rather than re-detecting in the warped image (which creates cubic
        # interpolation artifacts).
        self._matched_pairs: dict[int, tuple[NDArray, NDArray]] = {}
        # Detection params stored after measure() so validate()/save() reuse them.
        self.transform_type: str = "affine"
        self.smooth_sigma: float | tuple[float, ...] = 2.0
        self.min_distance: int | tuple[int, ...] = 10
        self.threshold_rel: float = 0.1
        self.match_max_distance: float = 10.0
        self.min_pairs: int = 4
        self.subpixel_refine: bool = True
        self.refine_radius: int | tuple[int, ...] = 5
        # Physical voxel size (z, y, x); None ⇒ work in pixel/voxel units.
        self.voxel_size: tuple[float, ...] | None = None
        self.verbose: bool = False

    # ------------------------------------------------------------------
    # Shared core of measure()
    # ------------------------------------------------------------------

    def _setup_logger(self, verbose: bool) -> None:
        if verbose and not self._logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(
                logging.Formatter("%(name)s | %(levelname)s | %(message)s")
            )
            self._logger.addHandler(handler)
            self._logger.setLevel(logging.INFO)

    def _fit_all(
        self,
        bead_stack: NDArray,
        reference_channel: int,
        transform_type: str,
    ) -> CorrectionResult:
        """Run the full per-channel detect → match → fit loop and build viz.

        Assumes the detection parameters have already been stored on ``self``.
        """
        n_channels = bead_stack.shape[0]
        result = CorrectionResult(reference_channel=reference_channel)

        all_centers: dict[int, NDArray] = {}
        all_pairs: dict[int, tuple[NDArray, NDArray]] = {}

        ref_img = self._normalise(bead_stack[reference_channel])
        ref_centers = self._detect_beads(ref_img)
        all_centers[reference_channel] = ref_centers
        self._logger.info(
            "ch%d (reference): %d beads detected", reference_channel, len(ref_centers)
        )

        for ch in range(n_channels):
            if ch == reference_channel:
                continue

            mov_img = self._normalise(bead_stack[ch])
            mov_centers = self._detect_beads(mov_img)
            all_centers[ch] = mov_centers

            # Coarse shift from centroid difference: robust to rotation and scale
            # because the centroid of a symmetric bead distribution is unaffected
            # by rotation/scale — only the translation component shifts it.
            if len(ref_centers) > 0 and len(mov_centers) > 0:
                coarse_shift = ref_centers.mean(axis=0) - mov_centers.mean(axis=0)
            else:
                coarse_shift = np.zeros(self._spatial_ndim)
            self._logger.info(
                "ch%d: %d beads detected | coarse shift (array order) = %s",
                ch,
                len(mov_centers),
                np.round(coarse_shift, 2),
            )

            src, dst = self._match_beads(ref_centers, mov_centers, coarse_shift)
            self._logger.info(
                "ch%d: %d bead pairs matched (max_distance=%.1f)",
                ch,
                len(src),
                self.match_max_distance,
            )

            if len(src) < self.min_pairs:
                warnings.warn(
                    f"Channel {ch}: only {len(src)} bead pairs found "
                    f"(need {self.min_pairs}). Falling back to translation-only.",
                    stacklevel=3,
                )
                tform = self._fit_translation(coarse_shift)
                rms: float | None = None
                n_pairs = 0
                inlier_mask = np.zeros(len(src), dtype=bool)
            else:
                tform, rms, inlier_mask = self._fit_transform(src, dst, transform_type)
                n_pairs = len(src)
                self._logger.info(
                    "ch%d: fit RMS = %.3f  transform =\n%s", ch, rms, tform.params
                )
            # Store only RANSAC inlier pairs so validate() uses the same clean
            # subset (outlier pairs inflate mean_error when the transform is applied).
            all_pairs[ch] = (src[inlier_mask], dst[inlier_mask])

            result.transforms[ch] = ChannelTransform(
                channel=ch,
                transform=tform,
                rms_residual=rms,
                n_pairs=n_pairs,
            )

        self._result = result
        self._bead_stack = bead_stack
        self._matched_pairs = all_pairs

        result.detection_image = self._build_detection_image(bead_stack, all_centers)
        result.pairs_image = self._build_pairs_image(bead_stack, all_pairs)

        return result

    # ------------------------------------------------------------------
    # Validation (shared)
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        detection_threshold: float | None = None,
    ) -> dict[int, dict]:
        """
        Validate colocalization quality on the bead image used for calibration.

        Applies the stored correction to the bead stack that was passed to
        ``measure()``, re-detects beads in each channel of the corrected image,
        and reports the residual displacement between matched bead pairs.

        Parameters
        ----------
        detection_threshold : float or None
            Override ``threshold_rel`` for this validation pass only.

        Returns
        -------
        dict[int, dict]
            Per-channel dict with keys ``'mean_error'``, ``'median_error'``,
            ``'max_error'``, ``'std_error'``, ``'n_pairs'`` and ``'residuals'``
            (per-pair distances).  The 3-D corrector adds per-axis and (when a
            ``voxel_size`` is set) physical-unit residuals.
        """
        if self._result is None or self._bead_stack is None:
            raise RuntimeError("No calibration data available. Run measure() first.")

        ref_ch = self._result.reference_channel

        # Re-detect beads in the original (uncorrected) bead stack, apply the
        # fitted transform to the moving-channel *positions*, then match the
        # corrected positions against the reference with a tight threshold.
        # This avoids image-warp artifacts (cubic ringing inflates peak counts in
        # the warped volume) while remaining independent from the RANSAC inlier set.
        ref_img = self._normalise(self._bead_stack[ref_ch])
        ref_centers = self._detect_beads(ref_img, threshold_rel=detection_threshold)

        stats: dict[int, dict] = {}
        for ch in range(self._bead_stack.shape[0]):
            if ch == ref_ch or ch not in self._result.transforms:
                continue

            mov_img = self._normalise(self._bead_stack[ch])
            mov_centers = self._detect_beads(
                mov_img, threshold_rel=detection_threshold
            )
            if len(mov_centers) == 0:
                stats[ch] = self._residual_stats(
                    np.empty((0, self._spatial_ndim)),
                    np.empty((0, self._spatial_ndim)),
                )
                continue

            # Apply the fitted transform to every detected moving position so
            # that they land in the reference frame.
            tform = self._result.transforms[ch].transform
            mov_xy = mov_centers[:, ::-1]          # (z,y,x) → skimage (x,y,z)
            corrected_xy = tform(mov_xy)
            corrected_centers = corrected_xy[:, ::-1]   # back to (z,y,x)

            # Match corrected moving positions against reference positions using
            # a tight threshold (= RANSAC residual_threshold = 2 px) — after a
            # good correction every bead should be within ~0.5 px.
            src, dst = self._match_beads(
                ref_centers,
                corrected_centers,
                shift=np.zeros(self._spatial_ndim),
            )
            stats[ch] = self._residual_stats(src, dst)

        if self.verbose:
            self._print_validation(stats, ref_ch)

        return stats

    def _residual_stats(self, src: NDArray, dst: NDArray) -> dict:
        """Euclidean residual statistics for a set of matched bead pairs."""
        if len(src) == 0:
            return {
                "mean_error": np.nan,
                "median_error": np.nan,
                "max_error": np.nan,
                "std_error": np.nan,
                "n_pairs": 0,
                "residuals": np.array([]),
            }
        residuals = np.linalg.norm(src - dst, axis=1)
        return {
            "mean_error": float(np.mean(residuals)),
            "median_error": float(np.median(residuals)),
            "max_error": float(np.max(residuals)),
            "std_error": float(np.std(residuals)),
            "n_pairs": len(residuals),
            "residuals": residuals,
        }

    # ------------------------------------------------------------------
    # save / from_json (shared)
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """
        Save the calibration transforms to a JSON file.

        Only the transforms (homogeneous matrices, RMS residuals, pair counts),
        the reference channel and the detection parameters are serialised — the
        bead image is not.  The file is sufficient to reconstruct the corrector
        for :meth:`apply` via :meth:`from_json`.

        Parameters
        ----------
        path : str or path-like
            Destination file path (e.g. ``"calibration.json"``).
        """
        result = self._resolve_result(None)
        measure_params: dict = {
            "transform_type": self.transform_type,
            "smooth_sigma": self.smooth_sigma,
            "min_distance": self.min_distance,
            "threshold_rel": self.threshold_rel,
            "match_max_distance": self.match_max_distance,
            "min_pairs": self.min_pairs,
            "subpixel_refine": self.subpixel_refine,
            "refine_radius": self.refine_radius,
            "verbose": self.verbose,
        }
        if self.voxel_size is not None:
            measure_params["voxel_size"] = list(self.voxel_size)
        data: dict = {
            "ndim": self._spatial_ndim,
            "reference_channel": result.reference_channel,
            "measure_params": measure_params,
            "transforms": {
                str(ch): {
                    "channel": ct.channel,
                    "matrix": ct.transform.params.tolist(),
                    "rms_residual": ct.rms_residual,
                    "n_pairs": ct.n_pairs,
                }
                for ch, ct in result.transforms.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: str | os.PathLike) -> _BaseChromaticShiftCorrector:
        """
        Load a calibration saved by :meth:`save` and return a ready corrector.

        The returned instance can immediately call :meth:`apply`.
        :meth:`validate` is not available (the bead image is not stored in the
        file); :meth:`measure` can still be called to re-calibrate.

        Parameters
        ----------
        path : str or path-like
            Path to a JSON file previously written by :meth:`save`.

        Returns
        -------
        _BaseChromaticShiftCorrector
            Instance (of the calling class) with the stored transforms loaded.

        Raises
        ------
        ValueError
            If the file holds a calibration of a different dimensionality than
            the class it is being loaded into.
        """
        with open(path) as f:
            data = json.load(f)

        file_ndim = int(data.get("ndim", 2))
        if file_ndim != cls._spatial_ndim:
            raise ValueError(
                f"This file holds a {file_ndim}-D calibration but "
                f"{cls.__name__} expects {cls._spatial_ndim}-D. Load it with the "
                f"matching corrector class."
            )

        corrector = cls()
        mp = data.get("measure_params", {})
        corrector.transform_type = mp.get("transform_type", corrector.transform_type)
        corrector.smooth_sigma = mp.get("smooth_sigma", corrector.smooth_sigma)
        corrector.min_distance = mp.get("min_distance", corrector.min_distance)
        corrector.threshold_rel = mp.get("threshold_rel", corrector.threshold_rel)
        corrector.match_max_distance = mp.get(
            "match_max_distance", corrector.match_max_distance
        )
        corrector.min_pairs = mp.get("min_pairs", corrector.min_pairs)
        corrector.subpixel_refine = mp.get("subpixel_refine", corrector.subpixel_refine)
        corrector.refine_radius = mp.get("refine_radius", corrector.refine_radius)
        corrector.verbose = mp.get("verbose", corrector.verbose)
        voxel_size = mp.get("voxel_size", None)
        corrector.voxel_size = tuple(voxel_size) if voxel_size is not None else None

        transforms: dict[int, ChannelTransform] = {}
        for ch_str, td in data["transforms"].items():
            transforms[int(ch_str)] = ChannelTransform(
                channel=td["channel"],
                transform=AffineTransform(matrix=np.array(td["matrix"])),
                rms_residual=td["rms_residual"],
                n_pairs=td["n_pairs"],
            )
        corrector._result = CorrectionResult(
            reference_channel=data["reference_channel"],
            transforms=transforms,
        )
        return corrector

    # ------------------------------------------------------------------
    # Shared private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(img: NDArray) -> NDArray:
        img = img.astype(np.float64)
        # Percentile clipping prevents hot pixels / saturated spots from
        # compressing all bead peaks below the detection threshold.
        lo = float(np.percentile(img, 1))
        hi = float(np.percentile(img, 99.9))
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0.0, 1.0)
        return img

    def _detect_beads(
        self, img: NDArray, threshold_rel: float | None = None
    ) -> NDArray:
        """Return an ``(N, ndim)`` array of bead centres in array (z, y, x) order.

        Works for any dimensionality; optionally sub-pixel refined.
        """
        thr = threshold_rel if threshold_rel is not None else self.threshold_rel
        smoothed = gaussian_filter(img, sigma=self.smooth_sigma)
        peaks = self._find_peaks(smoothed, thr)
        if peaks.size == 0:
            return np.empty((0, self._spatial_ndim), dtype=np.float64)
        centers = peaks.astype(np.float64)
        if self.subpixel_refine:
            centers = self._refine_centers(img, centers, self.refine_radius)
        return centers

    def _find_peaks(self, smoothed: NDArray, thr: float) -> NDArray:
        """Local-maxima search supporting scalar or per-axis ``min_distance``."""
        md = self.min_distance
        if isinstance(md, (int, np.integer)):
            return peak_local_max(smoothed, min_distance=int(md), threshold_rel=thr)
        # Per-axis exclusion via an anisotropic footprint (e.g. for z-step >> xy).
        footprint = np.ones(tuple(2 * int(m) + 1 for m in md), dtype=bool)
        return peak_local_max(smoothed, footprint=footprint, threshold_rel=thr)

    @staticmethod
    def _refine_centers(
        img: NDArray, centers: NDArray, radius: int | tuple[int, ...]
    ) -> NDArray:
        """Intensity-weighted centroid refinement for sub-pixel localisation.

        Dimension-agnostic: works on 2-D and 3-D images; ``radius`` may be a
        scalar or one value per axis.
        """
        ndim = centers.shape[1]
        if isinstance(radius, (int, np.integer)):
            radii = [int(radius)] * ndim
        else:
            radii = [int(r) for r in radius]
        shape = img.shape
        refined = np.empty_like(centers)
        for i in range(centers.shape[0]):
            c = centers[i]
            lo = [max(0, int(c[k]) - radii[k]) for k in range(ndim)]
            hi = [min(shape[k], int(c[k]) + radii[k] + 1) for k in range(ndim)]
            sl = tuple(slice(lo[k], hi[k]) for k in range(ndim))
            patch = img[sl]
            total = float(patch.sum())
            if total == 0:  # pragma: no cover
                refined[i] = c
                continue
            grids = np.indices(patch.shape, dtype=np.float64)
            for k in range(ndim):
                refined[i, k] = float((patch * (grids[k] + lo[k])).sum()) / total
        return refined

    def _match_beads(
        self,
        ref_centers: NDArray,
        mov_centers: NDArray,
        shift: tuple | NDArray | None = None,
    ) -> tuple[NDArray, NDArray]:
        """
        Mutual nearest-neighbour matching after applying a coarse shift.

        All distances are in voxels; ``match_max_distance`` is always in voxels.
        Returns ``(src, dst)`` arrays of matched centres in array (z, y, x) order.
        """
        d = self._spatial_ndim
        if shift is None:
            shift = np.zeros(d)
        if ref_centers.shape[0] == 0 or mov_centers.shape[0] == 0:
            return np.empty((0, d)), np.empty((0, d))

        shifted_mov = mov_centers + np.asarray(shift)

        fwd_tree = cKDTree(shifted_mov)
        fwd_dists, fwd_idx = fwd_tree.query(ref_centers, k=1)

        rev_tree = cKDTree(ref_centers)
        _, rev_idx = rev_tree.query(shifted_mov, k=1)

        src_list, dst_list = [], []
        for i, (dist, j) in enumerate(zip(fwd_dists, fwd_idx, strict=False)):
            if dist <= self.match_max_distance and rev_idx[j] == i:
                src_list.append(ref_centers[i])
                dst_list.append(mov_centers[j])

        if not src_list:
            return np.empty((0, d)), np.empty((0, d))
        return np.array(src_list), np.array(dst_list)

    @classmethod
    def _fit_transform(
        cls, src: NDArray, dst: NDArray, transform_type: str
    ) -> tuple[AffineTransform, float]:
        """
        Fit a transform mapping dst (moving/ch) → src (reference) using RANSAC.

        Coordinates are reversed from array order (z, y, x) to (x, y, z) before
        fitting because skimage's transforms use (x, y[, z]) internally.
        """
        tform_classes: dict[str, type] = {
            "affine": AffineTransform,
            "similarity": SimilarityTransform,
            "euclidean": EuclideanTransform,
            "translation": AffineTransform,
        }
        tform_cls = tform_classes.get(transform_type, AffineTransform)

        src_xy = src[:, ::-1]  # (z, y, x) → (x, y, z)
        dst_xy = dst[:, ::-1]

        tform, inliers = ransac(
            (dst_xy, src_xy),
            tform_cls,
            min_samples=cls._ransac_min_samples,
            residual_threshold=2.0,
            max_trials=1000,
        )
        if (
            tform is None
            or inliers is None
            or np.sum(inliers) < cls._ransac_min_samples
        ):  # pragma: no cover
            fallback = tform_cls()
            tform = fallback if fallback.estimate(dst_xy, src_xy) else tform_cls()
            inliers = np.ones(len(src), dtype=bool)

        # Guard against a degenerate (singular) matrix — can occur when bead
        # positions are nearly coplanar in z (e.g. thin z-stacks) or when
        # match_max_distance is too large and produces chaotic correspondences.
        if abs(float(np.linalg.det(tform.params))) < 1e-6:
            cls._logger.warning(
                "Fitted transform is singular (det ≈ 0); bead positions may be "
                "nearly coplanar or match_max_distance too large. "
                "Falling back to identity (no correction). Check your parameters."
            )
            d = src_xy.shape[1]
            tform = AffineTransform(matrix=np.eye(d + 1))
            inliers = np.ones(len(src), dtype=bool)

        predicted = tform(dst_xy[inliers])
        rms = float(
            np.sqrt(np.mean(np.sum((predicted - src_xy[inliers]) ** 2, axis=1)))
        )
        return tform, rms, inliers

    def _fit_translation(self, shift: NDArray) -> AffineTransform:
        """Build a pure-translation homogeneous transform of the right size.

        ``shift`` is in array (z, y, x) order; the homogeneous matrix is in
        (x, y, z) order, hence the reversal.
        """
        shift = np.asarray(shift, dtype=np.float64)
        d = self._spatial_ndim
        matrix = np.eye(d + 1)
        matrix[:d, d] = shift[::-1]
        return AffineTransform(matrix=matrix)

    def _coerce_stack(self, image_or_stack: NDArray | list) -> NDArray:
        """Normalise any accepted input form into a ``(C, *spatial)`` ndarray."""
        if isinstance(image_or_stack, np.ndarray):
            return image_or_stack
        if not isinstance(image_or_stack, list) or len(image_or_stack) == 0:
            raise TypeError("image_or_stack must be an ndarray or a non-empty list")
        frames: list[NDArray] = []
        for item in image_or_stack:
            if isinstance(item, str):
                frames.append(self._read_file(item))
            elif isinstance(item, np.ndarray):
                if item.ndim != self._spatial_ndim:
                    raise ValueError(
                        f"Each array in the list must be {self._spatial_ndim}-D, "
                        f"got shape {item.shape}"
                    )
                frames.append(item)
            else:
                raise TypeError(f"List items must be str or ndarray, got {type(item)}")
        return np.stack(frames, axis=0)

    @staticmethod
    def _read_file(path: str) -> NDArray:
        import tifffile

        return np.asarray(tifffile.imread(path))

    def _resolve_result(self, result: CorrectionResult | None) -> CorrectionResult:
        if result is not None:
            return result
        if self._result is None:
            raise RuntimeError("No correction result available. Run measure() first.")
        return self._result

    def _print_validation(self, stats: dict[int, dict], ref_ch: int) -> None:
        header = (
            f"\nValidation report  (reference = channel {ref_ch})\n"
            f"{'Channel':>8}  {'N pairs':>8}  {'Mean err':>14}  "
            f"{'Median':>12}  {'Max':>10}  {'Std':>9}\n" + "-" * 70
        )
        rows = "\n".join(
            f"{ch:>8}  {s['n_pairs']:>8}  {s['mean_error']:>14.3f}  "
            f"{s['median_error']:>12.3f}  {s['max_error']:>10.3f}  "
            f"{s['std_error']:>9.3f}"
            for ch, s in stats.items()
        )
        self._logger.info("%s\n%s", header, rows)

    # ------------------------------------------------------------------
    # Dimension-specific hooks (implemented by subclasses)
    # ------------------------------------------------------------------

    def apply(
        self,
        image_or_stack: NDArray | list[NDArray] | list[str],
        result: CorrectionResult | None = None,
        *,
        crop: bool = True,
    ) -> NDArray:
        """Apply the estimated correction to an image or stack (see subclasses)."""
        raise NotImplementedError

    def _build_detection_image(
        self, bead_stack: NDArray, centers_per_ch: dict[int, NDArray]
    ) -> NDArray:
        raise NotImplementedError

    def _build_pairs_image(
        self, bead_stack: NDArray, pairs_per_ch: dict[int, tuple[NDArray, NDArray]]
    ) -> NDArray:
        raise NotImplementedError
