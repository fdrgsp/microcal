# API reference — `microcal`

---

## `ChromaticShiftCorrector`

```python
from microcal import ChromaticShiftCorrector
```

### Constructor

```python
ChromaticShiftCorrector(
    reference_channel: int    = 0,
    smooth_sigma: float       = 2.0,
    min_distance: int         = 10,
    threshold_rel: float      = 0.1,
    match_max_distance: float = 10.0,
    min_pairs: int            = 4,
    subpixel_refine: bool     = True,
    refine_radius: int        = 5,
    verbose: bool             = False,
)
```

| Parameter | Default | Meaning |
| --- | --- | --- |
| `reference_channel` | `0` | Channel index used as the geometric reference. All other channels are registered to it. |
| `smooth_sigma` | `2.0` | Gaussian blur σ (pixels) applied before peak detection. Set to match the apparent bead PSF radius. Typical: 1–2 px for 100 nm beads at 100×, 2–3 px for 200 nm beads at 60×. |
| `min_distance` | `10` | Minimum centre-to-centre distance (pixels) between two accepted bead peaks. Peaks closer than this are merged (only the brightest survives). |
| `threshold_rel` | `0.1` | Minimum peak intensity as a fraction of the image maximum (after smoothing). Too high → dim beads missed. Too low → noise spikes counted as beads. |
| `match_max_distance` | `10.0` | Maximum distance (pixels) for two bead centres to be paired. Must be larger than the residual displacement after the coarse shift, and smaller than the minimum inter-bead spacing. |
| `min_pairs` | `4` | Minimum number of matched pairs required before fitting a full transform. If fewer are found, a translation-only fallback is used. |
| `subpixel_refine` | `True` | Refine pixel-level peak positions to sub-pixel accuracy using intensity-weighted centroid. Recommended; improves accuracy ~5–10×. |
| `refine_radius` | `5` | Half-width (pixels) of the patch used for sub-pixel centroid refinement. Should be ≥ `smooth_sigma`. |
| `verbose` | `False` | If `True`, show progress logs. |

---

### `measure`

```python
ChromaticShiftCorrector.measure(
    bead_stack: NDArray,
    *,
    transform_type: str = "affine",
) -> CorrectionResult
```

Estimates the chromatic shift transform for each channel relative to the reference channel. **Always call this on a bead image**, never on a sample image. The result is stored internally and also returned — calling `apply()` or `validate()` afterwards does not require passing it explicitly.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `bead_stack` | — | `NDArray` shape `(C, H, W)`, any integer or float dtype. |
| `transform_type` | `"affine"` | Type of geometric transform to fit. See options below. |

**`transform_type` options:**

| Value | DOF | Captures | When to use |
| --- | --- | --- | --- |
| `"affine"` | 6 | translation + rotation + anisotropic scale + shear | Default; handles all common chromatic shift types |
| `"similarity"` | 4 | translation + rotation + uniform scale | More stable with fewer beads; use when scale is isotropic |
| `"euclidean"` | 3 | translation + rotation only | Use when scale difference is negligible |
| `"translation"` | 2 | translation only | Use when only a lateral offset is expected |

---

### `apply`

```python
ChromaticShiftCorrector.apply(
    image_or_stack: NDArray | list[NDArray] | list[str],
    result: CorrectionResult | None = None,
    *,
    crop: bool = True,
) -> NDArray
```

Applies the measured correction to any image stack. Returns the corrected stack with the same dtype as the input.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `image_or_stack` | — | Input to correct. Accepted forms: `NDArray (C, H, W)`; `list[NDArray]` — one 2-D array per channel; `list[str]` — one file path per channel (read with `tifffile`). |
| `result` | `None` | `CorrectionResult` from a previous `measure()` call. If `None`, the result from the last `measure()` call is used automatically. |
| `crop` | `True` | If `True`, crop the output to the largest rectangle containing valid data in all channels (removes zero-filled borders). If `False`, keep the original spatial size with zeros at the borders. |

---

### `validate`

```python
ChromaticShiftCorrector.validate(
    *,
    detection_threshold: float | None = None,
) -> dict[int, dict]
```

Re-detects beads in the corrected bead calibration image and measures the residual displacement between channels. Call this after `measure()` to confirm the correction worked before applying it to sample data. No arguments are required — the corrector applies the correction to the stored bead stack internally. If `verbose=True` was passed to the constructor, a summary table is logged automatically.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `detection_threshold` | `None` | Override `threshold_rel` for this validation pass only. |

**Returns:** `dict[int, dict]` keyed by channel index, each entry containing:

| Key | Meaning |
| --- | --- |
| `mean_error` | Mean bead-pair displacement in pixels |
| `median_error` | Median displacement in pixels |
| `max_error` | Worst-case displacement in pixels |
| `std_error` | Standard deviation of displacements |
| `n_pairs` | Number of bead pairs used |
| `residuals` | Array of per-pair displacements |

**Interpretation:**

| `mean_error` | Quality |
| --- | --- |
| < 0.3 px | Excellent — diffraction-limited colocalization |
| 0.3–1.0 px | Acceptable for most applications |
| > 1.0 px | Something went wrong; revisit detection parameters |

---

## `CorrectionResult`

```python
from microcal import CorrectionResult
```

Returned by `measure()`. All fields are readable directly.

```python
result.reference_channel   # int — index of the reference channel
result.transforms          # dict[int, ChannelTransform] — one entry per non-reference channel
result.detection_image     # NDArray (2*C, H, W) float32 — interleaved normalised channel
                           #   images and filled-disk bead masks: [img_ch0, beads_ch0, ...]
result.pairs_image         # NDArray (H, W) int32 — label image; beads in the same matched
                           #   group share the same non-zero integer (display with Glasbey LUT)
```

Printing a `CorrectionResult` shows the RMS and 3×3 transform matrix for each channel:

```python
print(result)
# CorrectionResult(reference=0)
#   ch1: rms=0.082px  n_pairs=48
#        matrix=
# [[ 1.002  0.003 -1.521]
#  [-0.003  0.998  2.489]
#  [ 0.     0.     1.   ]]
```

---

## `ChannelTransform`

```python
from microcal import ChannelTransform

ct = result.transforms[1]
ct.channel           # int — channel index
ct.reference         # int — reference channel index
ct.transform         # skimage.transform.AffineTransform — 3×3 matrix in (x, y) space
ct.transform.params  # NDArray — the 3×3 homogeneous matrix
ct.rms_residual      # float | None — RMS of inlier bead pairs after fitting, in pixels
ct.n_pairs           # int — number of bead pairs used
```

The 3×3 matrix maps `[x_ch, y_ch, 1]ᵀ → [x_ref, y_ref, 1]ᵀ` where `(x, y) = (col, row)`.

---

## `generate_beads_image`

```python
from microcal import generate_beads_image

image, metadata = generate_beads_image(
    n_channels: int,
    shape: tuple[int, int],
    n_beads: int,
    bead_sigma: float,
    bead_intensity: float,
    bit_depth: int,
    offset: int,
    shifts: list[tuple[float, float]],
    rotations: list[float],
    scales: list[tuple[float, float]],
    snr: float,
    seed: int | None = None,
) -> tuple[NDArray, dict]
```

Generates a synthetic multi-channel bead image for testing and development. Returns a `(C, H, W)` uint array and a metadata dict.

| Parameter | Meaning |
| --- | --- |
| `n_channels` | Number of channels |
| `shape` | Image size `(H, W)` in pixels |
| `n_beads` | Number of beads to place |
| `bead_sigma` | PSF radius (pixels) of each bead |
| `bead_intensity` | Peak bead intensity before noise |
| `bit_depth` | Output bit depth (`8` or `16`) |
| `offset` | Constant background offset added to all pixels |
| `shifts` | Per-channel `(row, col)` translation applied to bead positions |
| `rotations` | Per-channel rotation in degrees |
| `scales` | Per-channel `(scale_row, scale_col)` anisotropic scale |
| `snr` | Signal-to-noise ratio (controls Gaussian noise level) |
| `seed` | Random seed for reproducibility |
