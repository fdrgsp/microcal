"""
Chromatic shift correction for 2-D multi-channel fluorescence microscopy.

Workflow
--------
1. Acquire a bead image with ChromaticShiftCorrector.measure()  → stores transforms
2. Validate the result with ChromaticShiftCorrector.validate()
3. Apply to any sample image with ChromaticShiftCorrector.apply()
4. Save / load the calibration with ChromaticShiftCorrector.save() / from_json()

For volumetric (z-stack) data see
:class:`~microcal.ChromaticShiftCorrector3D`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from skimage.transform import warp

from ._base_corrector import (
    ChannelTransform,
    CorrectionResult,
    _BaseChromaticShiftCorrector,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Re-exported for backward compatibility (these used to live in this module).
__all__ = ["ChannelTransform", "ChromaticShiftCorrector", "CorrectionResult"]

logger = logging.getLogger(__name__)


class ChromaticShiftCorrector(_BaseChromaticShiftCorrector):
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

    _spatial_ndim: ClassVar[int] = 2
    _ransac_min_samples: ClassVar[int] = 3
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
        method: str = "beads",
        smooth_sigma: float = 2.0,
        min_distance: int = 10,
        threshold_rel: float = 0.1,
        match_max_distance: float = 10.0,
        min_pairs: int = 4,
        subpixel_refine: bool = True,
        refine_radius: int = 5,
        block_size: int | tuple[int, int] = 128,
        block_overlap: float = 0.5,
        correlation_upsample: int = 10,
        min_correlation: float = 0.1,
        verbose: bool = False,
    ) -> CorrectionResult:
        """
        Estimate the chromatic shift from a multi-channel calibration image.

        Two correspondence strategies are available via `method`:

        * `"beads"` (default) — detect discrete sub-resolution beads and match
          them between channels.  Requires a sparse bead sample.
        * `"correlation"` — marker-free block phase correlation.  Tiles the
          image and measures each tile's sub-pixel shift by FFT cross-correlation,
          so it works on **continuous structures** stained in several colours
          (e.g. mitochondria), where bead detection finds nothing.

        Both strategies feed the same RANSAC transform fit, so the output (one
        affine matrix per channel) and `apply` / `validate` / `save` are
        identical.

        Parameters
        ----------
        bead_stack : NDArray, shape (C, H, W)
            Multi-channel calibration image (float or uint).
        reference_channel : int
            Channel index used as the geometric reference.  All other channels
            are registered to it.  Default 0.
        transform_type : str
            Type of transform to fit: 'affine' (default), 'similarity',
            'euclidean', or 'translation'.
        method : str
            Correspondence strategy: `"beads"` (default) or `"correlation"`.
        smooth_sigma : float
            (`method="beads"`) Sigma (px) of the Gaussian pre-smoothing applied
            before peak finding.  Should match the apparent bead radius.  Default 2.
        min_distance : int
            (`method="beads"`) Minimum pixel distance between two accepted
            peaks.  Default 10 px.
        threshold_rel : float
            (`method="beads"`) Minimum peak intensity as a fraction of the
            image maximum, after Gaussian smoothing.  Default 0.1.
        match_max_distance : float
            (`method="beads"`) Maximum distance (px) for a valid bead pair.
            Default 10 px.
        min_pairs : int
            Minimum matched pairs (beads or correlation blocks) required before
            fitting the transform.  Below this the corrector falls back to
            translation-only.  Default 4.
        subpixel_refine : bool
            (`method="beads"`) If True (default), refine each pixel-level peak
            to sub-pixel accuracy using an intensity-weighted centroid.
        refine_radius : int
            (`method="beads"`) Half-width (px) of the centroid-refinement
            patch.  Default 5.
        block_size : int or (by, bx)
            (`method="correlation"`) Edge length (px) of each correlation
            block; a tuple sets a different size per axis.  Default 128.
        block_overlap : float
            (`method="correlation"`) Fractional overlap of adjacent blocks, in
            `[0, 1)`.  Default 0.5 (50 %).
        correlation_upsample : int
            (`method="correlation"`) Up-sampling factor for sub-pixel phase
            correlation (skimage `upsample_factor`).  Default 10.
        min_correlation : float
            (`method="correlation"`) Minimum normalised cross-correlation
            quality (`1 - error`, in `[0, 1]`) for a block to be used; flat
            or poorly-correlated blocks below this are skipped.  Default 0.1.
        verbose : bool
            If True, emit progress logs.  Default False.

        Returns
        -------
        CorrectionResult
            Contains one ChannelTransform per non-reference channel, plus
            visualisation arrays always populated:

            * `result.detection_image` — `(2*C, H, W)` float32 with
              normalised channel images and filled-disk bead masks interleaved:
              `[img_ch0, beads_ch0, img_ch1, beads_ch1, ...]`.
            * `result.pairs_image` — `(H, W)` int32 label image where
              every bead in a matched group shares the same non-zero integer.
              Display with a Glasbey LUT (e.g. `ndv.imshow(result.pairs_image)`).
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
        self.voxel_size = None
        self.verbose = verbose
        self._store_correlation_params(
            method, block_size, block_overlap, correlation_upsample, min_correlation
        )

        self._setup_logger(verbose)

        if bead_stack.ndim != 3:
            raise ValueError("bead_stack must be 3-D (C, H, W)")

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
        Apply the estimated chromatic shift correction to an image or stack.

        Each non-reference channel is remapped into the reference frame using
        bicubic interpolation (order=3).  Pixels that fall outside the source
        frame after shifting are filled with 0 — no data is invented.

        Parameters
        ----------
        image_or_stack : NDArray (C, H, W) | list[NDArray] | list[str]
            The multi-channel image to correct.  Accepted forms:

            * `NDArray` shape `(C, H, W)` — channel-first stack.
            * `list[NDArray]` — one 2-D `(H, W)` array per channel; stacked
              in list order.  Useful when channels come from separate
              `tifffile.imread` calls.
            * `list[str]` — one file path per channel; each file is read with
              `tifffile.imread`.

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
            warped = warp(
                image_or_stack[ch].astype(np.float64),
                tform.inverse,
                order=3,
                mode="constant",
                cval=0.0,
                preserve_range=True,
            )
            corrected[ch] = np.clip(warped, 0.0, float(image_or_stack[ch].max()))

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
            corrected = self._crop_to_valid(corrected, valid_mask)

        return corrected.astype(orig_dtype)

    # ------------------------------------------------------------------
    # Visualisation array builders (2-D)
    # ------------------------------------------------------------------

    def _build_detection_image(
        self,
        bead_stack: NDArray,
        centers_per_ch: dict[int, NDArray],
    ) -> NDArray:
        """Return (2*C, H, W) float32: interleaved normalised images and masks."""
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
        radius = max(2, round(float(np.min(np.atleast_1d(self.smooth_sigma)))))
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
        radius = max(2, round(float(np.min(np.atleast_1d(self.smooth_sigma)))))
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
