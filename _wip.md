# ChromaticShiftCorrector

A Python class for measuring and correcting chromatic shift in multi-channel fluorescence microscopy images, using fluorescent bead calibration images.

---

## What is chromatic shift?

Fluorescence microscopes use different optical paths for each emission wavelength channel. Imperfections in the optics (lenses, dichroics, filters) cause each channel to be slightly magnified, rotated, and translated differently. A bead that truly sits at position (x, y) may appear at a slightly different position in channel 1 compared to channel 0. This is **chromatic shift** (also called chromatic aberration or channel misregistration).

The correction is always calibrated using fluorescent beads that are visible in all channels at once. Because the beads are point-like (sub-diffraction), their apparent centres can be localised with sub-pixel precision, and the geometric transform that maps one channel onto another can be estimated from those centre positions.

---

## Typical workflow

```python
import tifffile
from chromatic_shift_correction import ChromaticShiftCorrector

# 1. Load a multi-channel bead image (C, H, W)
bead_img = tifffile.imread("beads.tiff")   # shape (2, 512, 512), uint16

# 2. Create the corrector (tune parameters to your images)
sc = ChromaticShiftCorrector(
    reference_channel=0,
    smooth_sigma=2.0,
    min_distance=10,
    threshold_rel=0.1,
    match_max_distance=15,
    min_pairs=4,
    subpixel_refine=True,
    refine_radius=5,
)

# 3. Measure the chromatic shift from the bead image
result = sc.measure(bead_img)
print(result)

# 4. Validate — apply the transform to the bead image and measure residual error.
#    This tells you how accurate the correction is before using it on real data.
#    No arguments needed: the corrector applies the transform to the stored bead
#    stack internally, re-detects beads in each channel, and reports the
#    residual displacement between matched pairs.
stats = sc.validate()
# Typical output: mean error < 0.3 px for good calibration data
# If mean_error is high (> 1 px), revisit detection parameters before proceeding.

# 5. Apply the correction to any sample image (only once you are happy with the
#    validation result above)
sample_img = tifffile.imread("sample.tiff")   # same shape (2, 512, 512)
corrected = sc.apply(sample_img, crop=True)
tifffile.imwrite("sample_corrected.tiff", corrected)
```

---

## Constructor parameters

```python
ChromaticShiftCorrector(
    reference_channel = 0,
    smooth_sigma      = 2.0,
    min_distance      = 10,
    threshold_rel     = 0.1,
    match_max_distance = 10.0,
    min_pairs         = 4,
    subpixel_refine   = True,
    refine_radius     = 5,
)
```

| Parameter | Default | Meaning |
|---|---|---|
| `reference_channel` | `0` | Channel index to use as the geometric reference. All other channels are registered to it. |
| `smooth_sigma` | `2.0` | Gaussian blur σ (pixels) applied before peak detection. Set to match the apparent bead PSF radius. Typical: 1–2 px for 100 nm beads at 100×, 2–3 px for 200 nm beads at 60×. |
| `min_distance` | `10` | Minimum centre-to-centre distance (pixels) between two accepted bead peaks. Peaks closer than this are merged (only the brightest survives). Set to slightly less than the minimum expected bead spacing. |
| `threshold_rel` | `0.1` | Minimum peak intensity as a fraction of the image maximum (after smoothing). Peaks below this are rejected. Too high → dim beads missed. Too low → noise spikes counted as beads. |
| `match_max_distance` | `10.0` | Maximum distance (pixels) for two bead centres (one from each channel) to be paired as a corresponding pair. Must be larger than the residual displacement after the coarse shift, and smaller than the minimum inter-bead spacing. |
| `min_pairs` | `4` | Minimum number of matched pairs required before fitting a full transform. If fewer are found, a translation-only fallback is used. |
| `subpixel_refine` | `True` | Refine pixel-level peak positions to sub-pixel accuracy using intensity-weighted centroid. Recommended; improves accuracy ~5–10×. |
| `refine_radius` | `5` | Half-width (pixels) of the patch used for sub-pixel centroid refinement. Should be ≥ `smooth_sigma`. |

---

## Public methods

### `measure(bead_stack, *, transform_type="affine") → CorrectionResult`

Estimates the chromatic shift transform for each channel relative to the reference channel. **Always call this on a bead image**, never on a sample image.

**Input:** `bead_stack` — NumPy array of shape `(C, H, W)`, any integer or float dtype.

**Output:** `CorrectionResult` — a dataclass containing one `ChannelTransform` per non-reference channel. The result is also stored internally so you can call `apply()` without passing it explicitly.

**`transform_type` options:**

| Type | DOF | Captures | When to use |
|---|---|---|---|
| `"affine"` | 6 | translation + rotation + anisotropic scale + shear | Default; handles all common chromatic shift types |
| `"similarity"` | 4 | translation + rotation + uniform scale | More stable with fewer beads; use when scale is isotropic |
| `"euclidean"` | 3 | translation + rotation only | Pure rigid registration; use when scale is negligible |

---

### `apply(image_or_stack, result=None, *, crop=True) → NDArray`

Applies the measured correction to any image stack (bead image or sample image).

**Input:** `image_or_stack` — NumPy array of shape `(C, H, W)`, same dtype as the input is preserved in the output.

**`result`:** A `CorrectionResult` from a previous `measure()` call. If `None`, the result from the last `measure()` call is used automatically.

**`crop`:**
- `True` (default) — the output is cropped to the largest rectangular region that contains valid (non-zero-filled) data in all channels. This removes the thin black borders that appear at the edges after transformation, and ensures all channels cover exactly the same physical area.
- `False` — the output has the same spatial dimensions as the input. Pixels that fall outside the source frame after transformation are filled with 0.

---

### `validate(*, detection_threshold=None, verbose=True) → dict`

Re-detects beads in the corrected bead calibration image and measures the residual displacement between channels. Call this after `measure()` to confirm the correction worked and to quantify its accuracy. No arguments are needed — the corrector stores the bead stack internally during `measure()` and applies the correction automatically before re-detecting.

```python
stats = sc.validate()
```

**Output:** a dict keyed by channel index, each containing:

| Key | Meaning |
|---|---|
| `mean_error` | Mean bead-pair displacement in pixels |
| `median_error` | Median displacement in pixels |
| `max_error` | Worst-case displacement in pixels |
| `std_error` | Standard deviation of displacements |
| `n_pairs` | Number of bead pairs used |
| `residuals` | Array of per-pair displacements |

**Interpretation:**
- `mean_error < 0.3 px` → excellent correction; diffraction-limited colocalization
- `mean_error 0.3–1.0 px` → acceptable for most applications
- `mean_error > 1.0 px` → something went wrong; check bead detection parameters

---

## Step-by-step internal pipeline

### Step 1 — Normalisation (`_normalise`)

Each channel image is converted to float64 and rescaled to [0, 1] using **percentile clipping**:
- The 1st percentile is used as the low clip point (background level)
- The 99.9th percentile is used as the high clip point (bright bead peak)

This is critical because a single saturated pixel would make all bead peaks appear at the same very small value under min–max normalisation, causing them to fall below the detection threshold. Percentile clipping is robust to hot pixels and saturated spots.

### Step 2 — Bead detection (`_detect_beads`)

1. **Gaussian smoothing:** the normalised image is blurred with a Gaussian of σ = `smooth_sigma`. This suppresses pixel-level readout noise while preserving bead peaks (whose PSF σ is comparable to `smooth_sigma`).

2. **Local maxima search (`peak_local_max`):** finds all pixels that are a local maximum within a neighbourhood of radius `min_distance` AND exceed `threshold_rel` × image maximum. Each such pixel is a candidate bead centre. Returns integer (row, col) positions.

3. **Sub-pixel refinement (`_refine_centers`, if `subpixel_refine=True`):** for each integer peak, a patch of `(2×refine_radius+1)²` pixels is extracted from the *original* (unsmoothed) normalised image. The intensity-weighted centroid of the patch is computed:

   ```
   row_refined = Σ(intensity[r,c] × r) / Σ(intensity[r,c])
   col_refined = Σ(intensity[r,c] × c) / Σ(intensity[r,c])
   ```

   This removes the quantisation error of the integer peak position and achieves ~0.1 px localisation accuracy for well-sampled beads. All subsequent steps use these sub-pixel centres.

### Step 3 — Coarse shift estimation

For each non-reference channel, a **centroid-difference** coarse shift is computed:

```
coarse_shift = mean(ref_centers) - mean(mov_centers)
```

This gives an approximate translation between the two channels. It is **rotation- and scale-invariant**: for a bead distribution that is roughly symmetric around the image centre, rotation and scale do not move the centroid — only translation does. This makes it more robust than phase cross-correlation, which fails when the channels differ by rotation or scale in addition to translation.

The coarse shift is used only to pre-align bead centres before the nearest-neighbour matching step. It does not need to be precise.

### Step 4 — Mutual nearest-neighbour matching (`_match_beads`)

Corresponding bead pairs between the reference and moving channel are found using a **mutual nearest-neighbour (MNN)** criterion with a `cKDTree`:

1. **Forward query:** for each reference bead, find its nearest moving bead (after applying the coarse shift). The `cKDTree` data structure makes this O(N log N) instead of O(N²).

2. **Reverse query:** for each moving bead (coarse-shifted), find its nearest reference bead.

3. **Mutual consistency check:** a pair `(ref_i, mov_j)` is only accepted if:
   - The distance (after coarse shift) is ≤ `match_max_distance`, **and**
   - The reverse query also assigns `mov_j` to `ref_i` (i.e., it is a mutual nearest neighbour)

   The mutual check prevents two reference beads from being matched to the same moving bead (one-to-one correspondence). This is important when bead density is high relative to the displacement.

Pairs that pass the check are returned as two arrays `src` (ref positions) and `dst` (original, unshifted mov positions).

### Step 5 — Transform fitting with RANSAC (`_fit_transform`)

**Coordinate system conversion:**
Before fitting, bead centres are flipped from NumPy's `(row, col)` order to `(col, row) = (x, y)` order. This is required because `skimage.transform` functions (`estimate_transform`, `warp`) use `(x, y)` convention internally — passing `(row, col)` directly would transpose the rotation and scale components.

```python
src_xy = src[:, ::-1]   # (row, col) → (col, row) = (x, y)
dst_xy = dst[:, ::-1]
```

**RANSAC fitting:**
[RANSAC (Random Sample Consensus)](https://en.wikipedia.org/wiki/Random_sample_consensus) is used instead of plain least squares. The algorithm:

1. Randomly samples the minimum number of pairs needed to fit the model (3 for affine, 2 for similarity/euclidean).
2. Fits a candidate transform to those pairs.
3. Counts how many of all matched pairs are **inliers** — pairs where `|T(mov_xy) - ref_xy| < 2.0 px`.
4. Repeats up to 1000 times.
5. Returns the transform with the most inliers, then refits it using all inliers.

RANSAC makes the fitting **robust to wrong bead pairings**. Even if 30–40% of the nearest-neighbour matches are incorrect (possible when `match_max_distance` is large relative to inter-bead spacing), RANSAC can still recover the correct transform as long as the majority of pairs are correct.

**RMS residual** is computed on the RANSAC inliers only:
```
RMS = sqrt(mean(|T(mov_xy_inliers) - ref_xy_inliers|²))
```

### Step 6 — Storing the result

The fitted transform (a `skimage.transform.AffineTransform` object with a 3×3 homogeneous matrix in `(x, y)` space) is stored in a `ChannelTransform` dataclass along with the RMS and number of inlier pairs. All channel transforms are collected in a `CorrectionResult` object, which is returned to the caller and also stored internally on the corrector.

The transform encodes the mapping **moving channel → reference channel**: given a pixel location `(x, y)` in the moving channel, `T(x, y)` gives the corresponding location in the reference channel.

### Step 7 — Applying the correction (`apply`)

For each non-reference channel:

1. **Retrieve the stored transform** `T` (moving → reference, in `(x, y)` space).

2. **Warp the channel image** using `skimage.transform.warp`:
   ```python
   corrected[ch] = warp(channel_image, T.inverse, order=3, mode="constant", cval=0.0)
   ```
   `warp(image, inverse_map)` fills each output pixel by sampling the input. For output pixel at position `p_ref`, it calls `inverse_map(p_ref)` to find where to sample in the input. The inverse map needed is `T.inverse` (reference → moving), so that output pixel at `p_ref` samples from `channel_image[T⁻¹(p_ref)]`.

   - **Bicubic interpolation** (`order=3`) for the main image — sub-pixel accuracy without ringing.
   - **Border fill = 0** (`cval=0.0`) — pixels that would require sampling outside the source image are set to zero. No data is invented.

3. **Crop (if `crop=True`):** a unit mask (all ones) is warped through the same transform with nearest-neighbour interpolation (`order=0`) to find which output pixels have valid data. The intersection of all channel valid masks is computed, and the stack is sliced to the bounding box of that intersection. This removes the zero-filled border strips and ensures all channels cover exactly the same physical region.

4. **Dtype restoration:** the output array is cast back to the original input dtype (`uint8`, `uint16`, etc.) so downstream code receives the same format.

### Step 8 — Validation (`validate`)

1. Re-detects beads in the corrected stack using the same detection pipeline as `measure()`.
2. Matches them between reference and corrected non-reference channels using `_match_beads` with `shift=(0, 0)` — after correct correction the beads should already be aligned so no pre-shift is needed.
3. Computes the Euclidean distance between each matched pair: `residual = |ref_center - mov_center|`.
4. Reports per-channel statistics (mean, median, max, std, N).

---

## Parameter tuning guide

### Too few beads detected

- Lower `threshold_rel` (e.g. 0.05)
- Increase `smooth_sigma` to match your bead PSF size
- Lower `min_distance` if beads are densely packed

### Too many false positives (noise spikes detected as beads)

- Increase `threshold_rel` (e.g. 0.2–0.3)
- Increase `smooth_sigma`
- Increase `min_distance`

### Too few matched pairs

- Increase `match_max_distance` — but keep it well below the minimum inter-bead spacing
- Check if the coarse shift estimate is reasonable (printed during `measure()`)
- If chromatic shift is very large (> 20 px), the centroid-based coarse shift may be inaccurate; try increasing `match_max_distance` to compensate

### High RMS after fitting (> 1 px)

- Too many wrong matches — try decreasing `match_max_distance`
- Too few inliers for RANSAC — increase bead density in the calibration image
- Consider switching from `"affine"` to `"similarity"` if you have fewer than 10 matched pairs

### High validation error (> 0.5 px)

- Check bead density and SNR in the calibration image
- Ensure the calibration bead image was acquired under the same conditions as the sample (same objective, immersion, temperature)
- Verify `smooth_sigma` and `refine_radius` match the bead size in your images

---

## Data classes

### `CorrectionResult`

```python
result.reference_channel          # int: index of the reference channel
result.transforms                 # dict[int, ChannelTransform]: one entry per non-reference channel
```

Printing a `CorrectionResult` shows the RMS and 3×3 transform matrix for each channel.

### `ChannelTransform`

```python
ct = result.transforms[1]
ct.channel         # int: channel index
ct.reference       # int: reference channel index
ct.transform       # skimage.transform.AffineTransform — 3×3 matrix in (x,y) space
ct.transform.params  # numpy array, the 3×3 homogeneous matrix
ct.rms_residual    # float: RMS of inlier bead pairs after fitting, in pixels
ct.n_pairs         # int: number of bead pairs used
```

The 3×3 matrix maps `[x_ch, y_ch, 1]ᵀ → [x_ref, y_ref, 1]ᵀ` in `(x, y) = (col, row)` space.

---

## Dependencies

```
numpy
scipy        (gaussian_filter, cKDTree)
scikit-image (peak_local_max, ransac, AffineTransform, warp)
```
