"""
Chromatic shift correction for multi-channel fluorescence microscopy.

Workflow
--------
1. Acquire a bead image with ChromaticShiftCorrector.measure()  → stores transforms
2. Validate the result with ChromaticShiftCorrector.validate()
3. Apply to any sample image with ChromaticShiftCorrector.apply()
4. Save / load the calibration with ChromaticShiftCorrector.save() / from_json()
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree
from skimage.feature import peak_local_max
from skimage.measure import ransac
from skimage.transform import (
    AffineTransform,
    EuclideanTransform,
    SimilarityTransform,
    warp,
)

if TYPE_CHECKING:
    import os

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data container for a single channel-to-reference transform
# ---------------------------------------------------------------------------


@dataclass
class ChannelTransform:
    """Affine transform that maps a channel into the reference frame."""

    channel: int
    # skimage AffineTransform (3x3 homogeneous matrix)
    transform: AffineTransform
    # residual RMS in pixels after fitting (None if not computed)
    rms_residual: float | None = None
    # number of bead pairs used to fit the transform
    n_pairs: int = 0


@dataclass
class CorrectionResult:
    """Return value of ChromaticShiftCorrector.measure()."""

    reference_channel: int
    transforms: dict[int, ChannelTransform] = field(default_factory=dict)
    # Set by measure(): (2*C, H, W) float32 — interleaved normalised
    # images and filled-disk bead masks for every channel.
    detection_image: NDArray | None = None
    # Set by measure(): (H, W) int32 label image — each matched bead
    # group shares the same non-zero integer; use a Glasbey LUT to visualise.
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
# Main class
# ---------------------------------------------------------------------------


class ChromaticShiftCorrector:
    """
    Estimate and apply chromatic shift corrections from multi-channel bead images.

    Pipeline
    --------
    This follows the standard approach used in TrackMate, FIJI, Imaris, and
    most published chromatic-aberration correction workflows:

    1. **Bead detection** via Gaussian smoothing + local-maxima search
       (`peak_local_max`). An intensity-weighted centroid refinement step
       then gives sub-pixel accuracy (~0.1 px) at negligible cost.

    2. **Coarse alignment** via centroid-difference between channels.  Robust
       to rotation and scale because the centroid of a symmetric bead
       distribution is only affected by translation.

    3. **Transform fitting** via RANSAC, which rejects mismatched bead pairs
       and fits the requested transform type (affine, similarity, euclidean,
       or translation) to the matched bead-centre pairs.

    Accuracy is typically < 0.3 px RMS for SNR > 10 with >= 20 matched pairs.
    """

    def __init__(self) -> None:
        self._result: CorrectionResult | None = None
        self._bead_stack: NDArray | None = None
        # Detection params stored after measure() so validate() can reuse them.
        self.transform_type: str = "affine"
        self.smooth_sigma: float = 2.0
        self.min_distance: int = 10
        self.threshold_rel: float = 0.1
        self.match_max_distance: float = 10.0
        self.min_pairs: int = 4
        self.subpixel_refine: bool = True
        self.refine_radius: int = 5
        self.verbose: bool = False

    # ------------------------------------------------------------------
    # Public method 1 — measure
    # ------------------------------------------------------------------

    def measure(
        self,
        bead_stack: NDArray,
        *,
        reference_channel: int = 0,
        transform_type: str = "affine",
        smooth_sigma: float = 2.0,
        min_distance: int = 10,
        threshold_rel: float = 0.1,
        match_max_distance: float = 10.0,
        min_pairs: int = 4,
        subpixel_refine: bool = True,
        refine_radius: int = 5,
        verbose: bool = False,
    ) -> CorrectionResult:
        """
        Estimate the chromatic shift from a multi-channel bead image.

        Parameters
        ----------
        bead_stack : NDArray, shape (C, H, W)
            Multi-channel bead image (float or uint).
        reference_channel : int
            Channel index used as the geometric reference.  All other channels
            are registered to it.  Default 0.
        transform_type : str
            Type of transform to fit: 'affine' (default), 'similarity',
            'euclidean', or 'translation'.
        smooth_sigma : float
            Sigma (px) of the Gaussian pre-smoothing applied before peak
            finding.  Should match the apparent bead radius.  Default 2.
        min_distance : int
            Minimum pixel distance between two accepted peaks.  Default 10 px.
        threshold_rel : float
            Minimum peak intensity as a fraction of the image maximum, after
            Gaussian smoothing.  Default 0.1.
        match_max_distance : float
            Maximum distance (px) for a valid bead pair.  Default 10 px.
        min_pairs : int
            Minimum matched pairs required before fitting the transform.  Below
            this threshold the corrector falls back to translation-only.
            Default 4.
        subpixel_refine : bool
            If True (default), refine each pixel-level peak to sub-pixel
            accuracy using an intensity-weighted centroid.
        refine_radius : int
            Half-width (px) of the patch used for centroid refinement.
            Default 5.
        verbose : bool
            If True, emit progress logs.  Default False.

        Returns
        -------
        CorrectionResult
            Contains one ChannelTransform per non-reference channel, plus
            visualisation arrays always populated:

            * ``result.detection_image`` — ``(2*C, H, W)`` float32 with
              normalised channel images and filled-disk bead masks interleaved:
              ``[img_ch0, beads_ch0, img_ch1, beads_ch1, ...]``.
            * ``result.pairs_image`` — ``(H, W)`` int32 label image where
              every bead in a matched group shares the same non-zero integer.
              Display with a Glasbey LUT (e.g. ``ndv.imshow(result.pairs_image)``).
        """
        # Store params so validate(), save(), and private helpers can read them.
        self.transform_type = transform_type
        self.smooth_sigma = smooth_sigma
        self.min_distance = min_distance
        self.threshold_rel = threshold_rel
        self.match_max_distance = match_max_distance
        self.min_pairs = min_pairs
        self.subpixel_refine = subpixel_refine
        self.refine_radius = refine_radius
        self.verbose = verbose

        if verbose and not logger.handlers:
            _handler = logging.StreamHandler()
            _handler.setFormatter(
                logging.Formatter("%(name)s | %(levelname)s | %(message)s")
            )
            logger.addHandler(_handler)
            logger.setLevel(logging.INFO)

        if bead_stack.ndim != 3:
            raise ValueError("bead_stack must be 3-D (C, H, W)")

        n_channels = bead_stack.shape[0]
        result = CorrectionResult(reference_channel=reference_channel)

        # Accumulate per-channel data for optional visualisation
        all_centers: dict[int, NDArray] = {}
        all_pairs: dict[int, tuple[NDArray, NDArray]] = {}

        ref_img = self._normalise(bead_stack[reference_channel])
        ref_centers = self._detect_beads(ref_img)
        all_centers[reference_channel] = ref_centers
        logger.info(
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
                coarse_shift = np.zeros(2)
            logger.info(
                "ch%d: %d beads detected | coarse shift (row, col) = %s",
                ch,
                len(mov_centers),
                np.round(coarse_shift, 2),
            )

            src, dst = self._match_beads(ref_centers, mov_centers, coarse_shift)
            all_pairs[ch] = (src, dst)
            logger.info(
                "ch%d: %d bead pairs matched (max_distance=%.1fpx)",
                ch,
                len(src),
                self.match_max_distance,
            )

            if len(src) < self.min_pairs:
                warnings.warn(
                    f"Channel {ch}: only {len(src)} bead pairs found "
                    f"(need {self.min_pairs}). Falling back to translation-only.",
                    stacklevel=2,
                )
                tform = self._fit_translation(coarse_shift)
                rms = None
                n_pairs = 0
            else:
                tform, rms = self._fit_transform(src, dst, transform_type)
                n_pairs = len(src)
                logger.info(
                    "ch%d: fit RMS = %.3fpx  transform =\n%s",
                    ch,
                    rms,
                    tform.params,
                )

            result.transforms[ch] = ChannelTransform(
                channel=ch,
                transform=tform,
                rms_residual=rms,
                n_pairs=n_pairs,
            )

        self._result = result
        self._bead_stack = bead_stack

        result.detection_image = self._build_detection_image(bead_stack, all_centers)
        result.pairs_image = self._build_pairs_image(bead_stack, all_pairs)

        return result

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
        Apply the estimated chromatic shift correction to an image or stack.

        Each non-reference channel is remapped into the reference frame using
        bicubic interpolation (order=3).  Pixels that fall outside the source
        frame after shifting are filled with 0 — no data is invented.

        Parameters
        ----------
        image_or_stack : NDArray (C, H, W) | list[NDArray] | list[str]
            The multi-channel image to correct.  Accepted forms:

            * ``NDArray`` shape ``(C, H, W)`` — channel-first stack.
            * ``list[NDArray]`` — one 2-D ``(H, W)`` array per channel; stacked
              in list order.  Useful when channels come from separate
              ``tifffile.imread`` calls.
            * ``list[str]`` — one file path per channel; each file is read with
              ``tifffile.imread``.

        result : CorrectionResult or None
            Use a previously computed result.  Defaults to the result stored
            by the last call to `measure()` or loaded via `from_json()`.
        crop : bool
            If True (default), the output is cropped to the largest rectangle
            containing valid data in every channel.
            If False, the output has the same spatial size as the input with
            zero-filled borders where channels have no source data.

        Returns
        -------
        NDArray
            Corrected stack, shape (C, H, W) or (C, H', W') if crop=True.
        """
        result = self._resolve_result(result)
        image_or_stack = self._coerce_stack(image_or_stack)
        orig_dtype = image_or_stack.dtype

        if image_or_stack.ndim != 3:
            raise ValueError("image_or_stack must be 3-D (C, H, W)")

        n_channels, H, W = image_or_stack.shape
        corrected = image_or_stack.copy().astype(np.float64)

        # valid[r, c] will be False wherever any channel has a zero-filled border
        valid_mask = np.ones((H, W), dtype=bool)
        ones = np.ones((H, W), dtype=np.float64)

        for ch in range(n_channels):
            if ch == result.reference_channel:
                continue
            if ch not in result.transforms:
                continue

            tform = result.transforms[ch].transform
            # tform maps ch(x,y) → ref(x,y).  warp() needs the inverse mapping
            # (ref → ch) to know where to sample the channel image for each
            # output pixel.  tform.inverse is that mapping.
            corrected[ch] = warp(
                image_or_stack[ch].astype(np.float64),
                tform.inverse,
                order=3,
                mode="constant",
                cval=0.0,
                preserve_range=True,
            )

            if crop:
                ch_valid = warp(
                    ones,
                    tform.inverse,
                    order=0,
                    mode="constant",
                    cval=0.0,
                    preserve_range=True,
                )
                valid_mask &= ch_valid > 0.5

        if crop:
            rows = np.where(valid_mask.any(axis=1))[0]
            cols = np.where(valid_mask.any(axis=0))[0]
            if rows.size and cols.size:
                corrected = corrected[:, rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

        return corrected.astype(orig_dtype)

    # ------------------------------------------------------------------
    # Public method 3 — validate
    # ------------------------------------------------------------------

    def validate(
        self,
        *,
        detection_threshold: float | None = None,
    ) -> dict[int, dict]:
        """
        Validate colocalization quality on the bead image used for calibration.

        Applies the stored correction to the bead stack that was passed to
        `measure()`, re-detects beads in each channel of the corrected image,
        and reports the residual displacement between matched bead pairs.

        Call this immediately after `measure()` to confirm the correction
        works before applying it to sample images.

        Parameters
        ----------
        detection_threshold : float or None
            Override `threshold_rel` for this validation pass only.

        Returns
        -------
        dict[int, dict]
            Per-channel dict with keys: 'mean_error', 'median_error',
            'max_error', 'std_error', 'n_pairs', 'residuals' (array of
            per-pair distances in pixels).
        """
        if self._result is None or self._bead_stack is None:
            raise RuntimeError("No calibration data available. Run measure() first.")

        corrected_stack = self.apply(self._bead_stack, crop=False)
        result = self._result
        ref_ch = result.reference_channel

        ref_img = self._normalise(corrected_stack[ref_ch])
        ref_centers = self._detect_beads(ref_img, threshold_rel=detection_threshold)

        stats: dict[int, dict] = {}
        for ch in range(corrected_stack.shape[0]):
            if ch == ref_ch:
                continue

            mov_img = self._normalise(corrected_stack[ch])
            mov_centers = self._detect_beads(mov_img, threshold_rel=detection_threshold)
            src, dst = self._match_beads(ref_centers, mov_centers, shift=(0.0, 0.0))

            if len(src) == 0:
                stats[ch] = {
                    "mean_error": np.nan,
                    "median_error": np.nan,
                    "max_error": np.nan,
                    "std_error": np.nan,
                    "n_pairs": 0,
                    "residuals": np.array([]),
                }
                continue

            residuals = np.linalg.norm(src - dst, axis=1)
            stats[ch] = {
                "mean_error": float(np.mean(residuals)),
                "median_error": float(np.median(residuals)),
                "max_error": float(np.max(residuals)),
                "std_error": float(np.std(residuals)),
                "n_pairs": len(residuals),
                "residuals": residuals,
            }

        if self.verbose:
            self._print_validation(stats, ref_ch)

        return stats

    # ------------------------------------------------------------------
    # Public method 4 — save / from_json
    # ------------------------------------------------------------------

    def save(self, path: str | os.PathLike) -> None:
        """
        Save the calibration transforms to a JSON file.

        Only the transforms (3x3 affine matrices, RMS residuals, pair counts)
        and the reference channel are serialised — the bead image is not.
        The saved file is sufficient to reconstruct the corrector for
        :meth:`apply` via :meth:`from_json`.

        Parameters
        ----------
        path : str or path-like
            Destination file path (e.g. ``"calibration.json"``).
        """
        result = self._resolve_result(None)
        data: dict = {
            "reference_channel": result.reference_channel,
            "measure_params": {
                "transform_type": self.transform_type,
                "smooth_sigma": self.smooth_sigma,
                "min_distance": self.min_distance,
                "threshold_rel": self.threshold_rel,
                "match_max_distance": self.match_max_distance,
                "min_pairs": self.min_pairs,
                "subpixel_refine": self.subpixel_refine,
                "refine_radius": self.refine_radius,
                "verbose": self.verbose,
            },
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
    def from_json(cls, path: str | os.PathLike) -> ChromaticShiftCorrector:
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
        ChromaticShiftCorrector
            Instance with the stored transforms loaded.

        Examples
        --------
        >>> csc = ChromaticShiftCorrector.from_json("calibration.json")
        >>> corrected = csc.apply(sample_image)
        """
        with open(path) as f:
            data = json.load(f)

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
        transforms: dict[int, ChannelTransform] = {}
        for ch_str, td in data["transforms"].items():
            ch = int(ch_str)
            transforms[ch] = ChannelTransform(
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
    # Private helpers
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
        """Return an (N, 2) array of (row, col) bead centres.

        Optionally sub-pixel refined.
        """
        thr = threshold_rel if threshold_rel is not None else self.threshold_rel
        smoothed = gaussian_filter(img, sigma=self.smooth_sigma)
        peaks = peak_local_max(
            smoothed,
            min_distance=self.min_distance,
            threshold_rel=thr,
        )
        if peaks.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        centers = peaks.astype(np.float64)
        if self.subpixel_refine:
            centers = self._refine_centers(img, centers, self.refine_radius)
        return centers

    @staticmethod
    def _refine_centers(img: NDArray, centers: NDArray, radius: int = 5) -> NDArray:
        """Intensity-weighted centroid refinement for sub-pixel bead localisation."""
        H, W = img.shape
        refined = np.empty_like(centers)
        for i, (cy, cx) in enumerate(centers):
            r0 = max(0, int(cy) - radius)
            r1 = min(H, int(cy) + radius + 1)
            c0 = max(0, int(cx) - radius)
            c1 = min(W, int(cx) + radius + 1)
            patch = img[r0:r1, c0:c1]
            total = patch.sum()
            if total == 0:  # pragma: no cover
                refined[i] = centers[i]
                continue
            rows = np.arange(r0, r1, dtype=np.float64)
            cols = np.arange(c0, c1, dtype=np.float64)
            refined[i, 0] = (patch.sum(axis=1) @ rows) / total
            refined[i, 1] = (patch.sum(axis=0) @ cols) / total
        return refined

    def _match_beads(
        self,
        ref_centers: NDArray,
        mov_centers: NDArray,
        shift: tuple | NDArray = (0.0, 0.0),
    ) -> tuple[NDArray, NDArray]:
        """
        Mutual nearest-neighbour matching after applying a coarse shift.

        Returns (src, dst) arrays of matched (row, col) pairs.
        """
        if ref_centers.shape[0] == 0 or mov_centers.shape[0] == 0:
            return np.empty((0, 2)), np.empty((0, 2))

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
            return np.empty((0, 2)), np.empty((0, 2))
        return np.array(src_list), np.array(dst_list)

    @staticmethod
    def _fit_transform(
        src: NDArray, dst: NDArray, transform_type: str
    ) -> tuple[AffineTransform, float]:
        """
        Fit a transform mapping dst (moving/ch) → src (reference) using RANSAC.

        Coordinates are flipped from (row, col) to (col, row) = (x, y) before
        fitting because skimage's transforms and warp() use (x, y) internally.
        """
        _TFORM_CLASSES: dict[str, type] = {
            "affine": AffineTransform,
            "similarity": SimilarityTransform,
            "euclidean": EuclideanTransform,
            "translation": AffineTransform,
        }
        cls = _TFORM_CLASSES.get(transform_type, AffineTransform)

        src_xy = src[:, ::-1]  # (row, col) → (col, row) = (x, y)
        dst_xy = dst[:, ::-1]

        tform, inliers = ransac(
            (dst_xy, src_xy),
            cls,
            min_samples=3,
            residual_threshold=2.0,
            max_trials=1000,
        )
        if tform is None or inliers is None or np.sum(inliers) < 3:  # pragma: no cover
            tform = cls()
            tform.estimate(dst_xy, src_xy)
            inliers = np.ones(len(src), dtype=bool)

        predicted = tform(dst_xy[inliers])
        rms = float(
            np.sqrt(np.mean(np.sum((predicted - src_xy[inliers]) ** 2, axis=1)))
        )
        return tform, rms

    @staticmethod
    def _fit_translation(shift: NDArray) -> AffineTransform:
        # shift is (delta_row, delta_col); convert to (x,y) = (col, row).
        return AffineTransform(translation=(float(shift[1]), float(shift[0])))

    @staticmethod
    def _coerce_stack(image_or_stack: NDArray | list) -> NDArray:
        """Normalise any accepted input form into a (C, H, W) ndarray."""
        if isinstance(image_or_stack, np.ndarray):
            return image_or_stack
        if not isinstance(image_or_stack, list) or len(image_or_stack) == 0:
            raise TypeError("image_or_stack must be an ndarray or a non-empty list")
        frames: list[NDArray] = []
        for item in image_or_stack:
            if isinstance(item, str):
                frames.append(ChromaticShiftCorrector._read_file(item))
            elif isinstance(item, np.ndarray):
                if item.ndim != 2:
                    raise ValueError(
                        "Each array in the list must be 2-D (H, W), "
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

    # ------------------------------------------------------------------
    # Visualisation array builders (no display dependency)
    # ------------------------------------------------------------------

    def _build_detection_image(
        self,
        bead_stack: NDArray,
        centers_per_ch: dict[int, NDArray],
    ) -> NDArray:
        """Return (2*C, H, W) float32: interleaved normalised images and bead masks."""
        C, H, W = bead_stack.shape
        vis = np.zeros((2 * C, H, W), dtype=np.float32)
        for ch in range(C):
            vis[2 * ch] = self._normalise(bead_stack[ch]).astype(np.float32)
            if ch in centers_per_ch and len(centers_per_ch[ch]) > 0:
                vis[2 * ch + 1] = self._make_bead_mask(H, W, centers_per_ch[ch])
        return vis

    def _build_pairs_image(
        self,
        bead_stack: NDArray,
        pairs_per_ch: dict[int, tuple[NDArray, NDArray]],
    ) -> NDArray:
        """Return (H, W) int32: matched bead groups share a non-zero integer label."""
        _, H, W = bead_stack.shape
        labels = np.zeros((H, W), dtype=np.int32)
        radius = max(2, round(self.smooth_sigma))
        ref_to_label: dict[tuple[int, int], int] = {}
        next_label = 1
        for _ch, (src, dst) in pairs_per_ch.items():
            for s, d in zip(src, dst, strict=False):
                key = (round(s[0]), round(s[1]))
                if key not in ref_to_label:
                    ref_to_label[key] = next_label
                    next_label += 1
                lbl = ref_to_label[key]
                self._paint_disk(labels, round(s[0]), round(s[1]), radius, lbl)
                self._paint_disk(labels, round(d[0]), round(d[1]), radius, lbl)
        return labels

    def _make_bead_mask(self, H: int, W: int, centers: NDArray) -> NDArray:
        mask = np.zeros((H, W), dtype=np.float32)
        radius = max(2, round(self.smooth_sigma))
        for cy, cx in centers:
            self._paint_disk(mask, round(cy), round(cx), radius, 1.0)
        return mask

    @staticmethod
    def _paint_disk(
        img: NDArray, row: int, col: int, radius: int, value: float
    ) -> None:
        H, W = img.shape
        for r in range(max(0, row - radius), min(H, row + radius + 1)):
            for c in range(max(0, col - radius), min(W, col + radius + 1)):
                if (r - row) ** 2 + (c - col) ** 2 <= radius * radius:
                    img[r, c] = value

    @staticmethod
    def _print_validation(stats: dict[int, dict], ref_ch: int) -> None:
        header = (
            f"\nValidation report  (reference = channel {ref_ch})\n"
            f"{'Channel':>8}  {'N pairs':>8}  {'Mean err (px)':>14}  "
            f"{'Median (px)':>12}  {'Max (px)':>10}  {'Std (px)':>9}\n" + "-" * 70
        )
        rows = "\n".join(
            f"{ch:>8}  {s['n_pairs']:>8}  {s['mean_error']:>14.3f}  "
            f"{s['median_error']:>12.3f}  {s['max_error']:>10.3f}  "
            f"{s['std_error']:>9.3f}"
            for ch, s in stats.items()
        )
        logger.info("%s\n%s", header, rows)
