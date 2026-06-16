# API reference — `microcal`

---

## `ChromaticShiftCorrector`

```python
from microcal import ChromaticShiftCorrector
```

### Constructor

```python
ChromaticShiftCorrector()
```

No arguments. Detection and fitting parameters are passed to `measure()`.
To load an existing calibration, use `ChromaticShiftCorrector.from_json()`.

---

### `measure`

```python
ChromaticShiftCorrector.measure(
    bead_stack: NDArray,
    *,
    reference_channel: int    = 0,
    transform_type: str       = "affine",
    smooth_sigma: float       = 2.0,
    min_distance: int         = 10,
    threshold_rel: float      = 0.1,
    match_max_distance: float = 10.0,
    min_pairs: int            = 4,
    subpixel_refine: bool     = True,
    refine_radius: int        = 5,
    verbose: bool             = False,
) -> CorrectionResult
```

Estimates the chromatic shift transform for each channel relative to the reference channel. **Always call this on a bead image**, never on a sample image. The result is stored internally and also returned — calling `apply()` or `validate()` afterwards does not require passing it explicitly.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `bead_stack` | — | `NDArray` shape `(C, H, W)`, any integer or float dtype. |
| `reference_channel` | `0` | Channel index used as the geometric reference. All other channels are registered to it. |
| `transform_type` | `"affine"` | Type of geometric transform to fit. See options below. |
| `smooth_sigma` | `2.0` | Gaussian blur σ (pixels) applied before peak detection. Set to match the apparent bead PSF radius. Typical: 1–2 px for 100 nm beads at 100×, 2–3 px for 200 nm beads at 60×. |
| `min_distance` | `10` | Minimum centre-to-centre distance (pixels) between two accepted bead peaks. Peaks closer than this are merged (only the brightest survives). |
| `threshold_rel` | `0.1` | Minimum peak intensity as a fraction of the image maximum (after smoothing). Too high → dim beads missed. Too low → noise spikes counted as beads. |
| `match_max_distance` | `10.0` | Maximum distance **in pixels** for two bead centres to be paired. Must be larger than the residual displacement after the coarse shift, and smaller than the minimum inter-bead spacing. Always interpreted in pixels (voxels for 3-D), regardless of whether `voxel_size` is set. |
| `min_pairs` | `4` | Minimum number of matched pairs required before fitting a full transform. If fewer are found, a translation-only fallback is used. |
| `subpixel_refine` | `True` | Refine pixel-level peak positions to sub-pixel accuracy using intensity-weighted centroid. Recommended; improves accuracy ~5–10×. |
| `refine_radius` | `5` | Half-width (pixels) of the patch used for sub-pixel centroid refinement. Should be ≥ `smooth_sigma`. |
| `verbose` | `False` | If `True`, emit progress logs. |

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
| `result` | `None` | `CorrectionResult` from a previous `measure()` call. If `None`, the result from the last `measure()` call (or `from_json()`) is used automatically. |
| `crop` | `True` | If `True`, crop the output to the largest rectangle containing valid data in all channels (removes zero-filled borders). If `False`, keep the original spatial size with zeros at the borders. |

---

### `validate`

```python
ChromaticShiftCorrector.validate(
    *,
    detection_threshold: float | None = None,
) -> dict[int, dict]
```

Re-detects beads in the corrected bead calibration image and measures the residual displacement between channels. Call this after `measure()` to confirm the correction worked before applying it to sample data. No arguments are required — the corrector applies the correction to the stored bead stack internally. If `verbose=True` was passed to `measure()`, a summary table is logged automatically.

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

### `save`

```python
ChromaticShiftCorrector.save(path: str | os.PathLike) -> None
```

Saves the calibration to a JSON file. The file contains the reference channel, all transform matrices (3×3 affine, RMS residuals, pair counts), and the full set of `measure()` parameters used to produce them. The bead image is not stored. The saved file is sufficient to reconstruct the corrector for `apply()` via `from_json()`.

| Parameter | Meaning |
| --- | --- |
| `path` | Destination file path (e.g. `"calibration.json"`). |

---

### `from_json`

```python
ChromaticShiftCorrector.from_json(path: str | os.PathLike) -> ChromaticShiftCorrector
```

Class method. Loads a calibration saved by `save()` and returns a corrector ready to call `apply()`. All `measure()` parameters from the original calibration run are restored on the instance (e.g. `csc.smooth_sigma`, `csc.transform_type`). `validate()` is not available on a loaded instance (the bead image is not stored in the file); `measure()` can still be called to re-calibrate.

| Parameter | Meaning |
| --- | --- |
| `path` | Path to a JSON file previously written by `save()`. |

**Example:**

```python
csc = ChromaticShiftCorrector.from_json("calibration.json")
corrected = csc.apply(sample_image)
```

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
    n_channels: int = 3,
    shape: tuple[int, int] = (512, 512),                    # (H, W)
    n_beads: int = 50,
    bead_sigma: float = 2.0,
    bead_intensity: float = 60.0,
    bit_depth: int = 8,
    shifts: list[tuple[float, float]] | None = None,        # (row, col)
    rotations: list[float] | None = None,
    scales: list[tuple[float, float]] | None = None,        # (scale_row, scale_col)
    offset: int = 10,
    snr: float = 20.0,
    seed: int | None = 42,
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
| `shifts` | Per-channel `(row, col)` translation applied to bead positions |
| `rotations` | Per-channel rotation in degrees |
| `scales` | Per-channel `(scale_row, scale_col)` anisotropic scale |
| `offset` | Constant background offset added to all pixels |
| `snr` | Signal-to-noise ratio (controls Gaussian noise level) |
| `seed` | Random seed for reproducibility |

---

## `ChromaticShiftCorrector3D`

```python
from microcal import ChromaticShiftCorrector3D
```

The volumetric counterpart of `ChromaticShiftCorrector` for z-stacks shaped `(C, Z, Y, X)`. The pipeline, return types and method names are identical — `measure()`, `apply()`, `validate()`, `save()`, `from_json()` — so the tables above apply, with the following 3-D differences:

- **Input** is `(C, Z, Y, X)` (and `apply` also accepts `list[NDArray]` of `(Z, Y, X)` volumes or `list[str]` paths to 3-D TIFFs).
- **Transforms are 4×4** homogeneous matrices (12-DOF affine). `apply()` warps with `scipy.ndimage.affine_transform` (tricubic).
- `detection_image` is `(2*C, Z, Y, X)` (volume + filled-sphere mask per channel); `pairs_image` is `(Z, Y, X)`.
- `save()` writes the 4×4 matrices plus an `"ndim": 3` marker and the `voxel_size`. `from_json()` raises if a file's `ndim` does not match the class.

### `measure` — additional / changed parameters

```python
ChromaticShiftCorrector3D.measure(
    bead_stack: NDArray,                 # (C, Z, Y, X)
    *,
    reference_channel: int = 0,
    transform_type: str = "affine",
    smooth_sigma: float | tuple[float, float, float] = 2.0,   # scalar or (sz, sy, sx)
    min_distance: int | tuple[int, int, int] = 10,            # scalar or (mz, my, mx)
    threshold_rel: float = 0.1,
    match_max_distance: float = 10.0,
    min_pairs: int = 4,
    subpixel_refine: bool = True,
    refine_radius: int | tuple[int, int, int] = 5,            # scalar or (rz, ry, rx)
    voxel_size: tuple[float, float, float] | None = None,     # (z, y, x)
    verbose: bool = False,
) -> CorrectionResult
```

| Parameter | Meaning (3-D specifics) |
| --- | --- |
| `smooth_sigma` | Scalar or per-axis `(sz, sy, sx)`. Use a per-axis value for anisotropic stacks where the PSF is axially elongated. |
| `min_distance` | Scalar (isotropic in voxels) or per-axis `(mz, my, mx)`, which builds an anisotropic exclusion footprint (e.g. a smaller axial separation). |
| `refine_radius` | Scalar or per-axis `(rz, ry, rx)` half-width of the centroid-refinement patch. |
| `voxel_size` | Optional physical voxel size `(z, y, x)` in any consistent unit (e.g. µm). When given, `validate()` additionally reports residuals in physical units (`mean_error_physical` etc.). Does **not** change the unit of `match_max_distance`, which is always in voxels. `None` ⇒ voxel-unit residuals only. |

The fitted transform always lives in **voxel space**, so the correction is correct regardless of `voxel_size`; that argument only affects matching robustness and residual reporting.

`validate()` returns the same keys as the 2-D version (in voxels) and additionally:

| Key | Meaning |
| --- | --- |
| `mean_error_per_axis` | Mean absolute residual per axis `(z, y, x)`, in voxels |
| `mean_error_physical` / `median_error_physical` / `max_error_physical` | Residual norm in physical units (only when `voxel_size` is set) |

---

## `generate_beads_image_3d`

```python
from microcal import generate_beads_image_3d

volume, metadata = generate_beads_image_3d(
    n_channels: int = 3,
    shape: tuple[int, int, int] = (32, 256, 256),          # (Z, Y, X)
    n_beads: int = 50,
    bead_sigma: float | tuple[float, float, float] = (1.5, 2.0, 2.0),
    bead_intensity: float = 60.0,
    bit_depth: int = 8,
    shifts: list[tuple[float, float, float]] | None = None,        # (dz, dy, dx)
    rotations: list[float | tuple[float, float, float]] | None = None,
    scales: list[tuple[float, float, float]] | None = None,        # (sz, sy, sx)
    offset: int = 10,
    snr: float = 20.0,
    seed: int | None = 42,
) -> tuple[NDArray, dict]
```

The 3-D counterpart of `generate_beads_image`. Renders beads with an anisotropic Gaussian PSF and warps each channel by a per-channel 3-D affine about the volume centre. Returns a `(C, Z, Y, X)` uint array and a metadata dict (`centers` is `(N, 3)` in `(z, y, x)`).

| Parameter | Meaning (differences from 2-D) |
| --- | --- |
| `shape` | Volume size `(Z, Y, X)` in voxels |
| `bead_sigma` | Scalar or per-axis `(sz, sy, sx)` PSF radius (real PSFs have `sz > sxy`) |
| `shifts` | Per-channel `(dz, dy, dx)` translation in voxels |
| `rotations` | Per-channel rotation in degrees: a scalar (about the optical/z axis) or a `(rz, ry, rx)` Euler triple |
| `scales` | Per-channel `(sz, sy, sx)` anisotropic scale |
