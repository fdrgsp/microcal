# Internal pipeline — step by step

This document walks through exactly what `ChromaticShiftCorrector.measure()` does
internally, in order, with small worked numerical examples for each step. It
mirrors the implementation in
[`_chromatic_shift_corrector.py`](src/microcal/_chromatic_shift_corrector.py).

---

## Step 1 — Normalisation

Each channel is independently rescaled to the `[0, 1]` range using **percentile
clipping** (1st and 99.9th percentile), instead of the raw min/max:

```python
lo = percentile(img, 1)
hi = percentile(img, 99.9)
img = clip((img - lo) / (hi - lo), 0, 1)
```

**Why percentiles and not min/max?** A single hot or saturated pixel can throw
the whole scale off.

**Example:** a 16-bit image where real beads peak around 2000–5000, but one dead
pixel reads 65535. Using `min`/`max` would rescale so that the hot pixel becomes
`1.0` and every real bead collapses to roughly `0.03–0.08` — almost
indistinguishable from background noise. The 99.9th percentile instead sits near
the top of the real bead intensities, so beads are stretched across nearly the
full `[0, 1]` range and the hot pixel is simply clipped to `1.0`.

---

## Step 2 — Bead detection

Bead detection happens independently for every channel and produces a list of
`(row, col)` bead centres. It has three sub-steps.

### 2a — Gaussian smoothing

The normalised image is blurred with `gaussian_filter(img, sigma=smooth_sigma)`.
This merges the handful of bright pixels that make up one bead's point-spread
function into a single smooth hill, and suppresses single-pixel noise spikes
that would otherwise be picked up as fake beads.

### 2b — Local maxima search

`peak_local_max` scans the smoothed image and returns **integer pixel**
coordinates of local intensity maxima, subject to two filters:

- `min_distance` — two peaks closer than this are merged (only the brightest survives)
- `threshold_rel` — peaks below `threshold_rel × image_max` are discarded

At this point each bead is located to the nearest whole pixel, e.g. `(10, 15)`.

### 2c — Sub-pixel refinement

A whole-pixel position is rarely the *true* centre of a bead — the real centre
usually falls between pixels. `_refine_centers` improves each detection to
sub-pixel accuracy using an **intensity-weighted centroid**, computed
independently for each bead over a small `(2 × refine_radius + 1)` square patch
around its integer peak:

```text
centroid_row = Σ (intensity[r, c] × r) / Σ intensity[r, c]
centroid_col = Σ (intensity[r, c] × c) / Σ intensity[r, c]
```

The sums run over every **pixel** in that one bead's small patch — not over
multiple beads. Each bead is refined on its own, from its own patch, completely
independently of every other bead.

**Worked example** — a bead whose integer peak landed at `(10, 15)`, with
`refine_radius = 1` (a 3×3 patch, 9 pixels, all belonging to this *one* bead):

```text
              col 14   col 15   col 16
   row  9        20       80       30
   row 10        60      200 ←     150     (200 = brightest pixel = integer peak)
   row 11        10       40       20
```

Row sums: `9 → 130`, `10 → 410`, `11 → 70`. Column sums: `14 → 90`, `15 → 320`,
`16 → 200`. Total intensity = `610`.

```text
centroid_row = (9×130 + 10×410 + 11×70) / 610 = 6040 / 610 ≈ 9.90
centroid_col = (14×90 + 15×320 + 16×200) / 610 = 9260 / 610 ≈ 15.18
```

**Result for this one bead:** integer peak `(10, 15)` → refined centre
`(9.90, 15.18)`. One input bead, one output position.

This is repeated independently for every detected bead, so for `N` detected
beads you end up with `N` refined `(row, col)` centres, e.g.:

```text
bead 0:  integer (10, 15)  →  refined (9.90, 15.18)
bead 1:  integer (40, 88)  →  refined (40.32, 87.74)
bead 2:  integer (12, 203) →  refined (11.88, 203.41)
...
```

---

## Step 3 — Coarse shift estimation

After Step 2, both the reference channel and the moving channel each have their
own list of sub-pixel bead centres (one entry per detected bead). Step 3 turns
those two lists into a **single translation vector** for the whole channel pair:

```python
coarse_shift = ref_centers.mean(axis=0) - mov_centers.mean(axis=0)
```

This is **not** per bead — it collapses every bead in the reference channel into
one average position, collapses every bead in the moving channel into one
average position, and subtracts the two. The result is one `(Δrow, Δcol)` vector
for the entire channel, used only to bring the two bead clouds roughly into
register before the precise matching in Step 4.

**Why the centroid?** The centroid of a roughly-symmetric cloud of bead
positions is (to first order) unaffected by rotation or scaling — only by
translation. So even if channel 1 is also rotated/scaled relative to the
reference, the centroid difference still gives a good estimate of the
*translational* part of the misalignment.

**Worked example:** suppose the reference channel's beads average to
`(255.3, 248.7)` and channel 1's beads average to `(251.8, 251.2)`. Then:

```text
coarse_shift = (255.3 − 251.8, 248.7 − 251.2) = (3.5, −2.5)
```

This single vector `(3.5, −2.5)` is added to **every** channel-1 bead position
before the matching step — not recomputed per bead.

---

## Step 4 — Mutual nearest-neighbour matching

Now each moving-channel bead is shifted by `coarse_shift`, and the corrector
must decide which reference bead corresponds to which moving bead.
`_match_beads` does this with **mutual nearest-neighbour (MNN)** matching using
a `cKDTree` for fast spatial lookups:

```python
shifted_mov = mov_centers + coarse_shift

fwd_idx = nearest shifted-moving bead for each reference bead   (cKDTree query)
rev_idx = nearest reference bead for each shifted-moving bead   (cKDTree query)

keep pair (i, j) only if:
    distance(ref[i], shifted_mov[j]) <= match_max_distance
    AND rev_idx[j] == i        # the match is mutual / symmetric
```

A pair survives only if each bead is the *other's* closest neighbour — not just
one-directional closest. This rejects many-to-one matches such as two reference
beads both being closest to the same moving bead (common near image borders or
in dense bead fields).

**Worked example:** four reference beads `R0…R3` and four shifted moving beads
`M0…M3`.

- `R0`'s nearest neighbour is `M0`, and `M0`'s nearest neighbour is `R0` →
  **mutual → kept**.
- `R2`'s nearest neighbour is `M3` (distance `4.1 px`), but `M3`'s nearest
  neighbour is actually `R3` (distance `1.8 px`, smaller) → **not mutual →
  discarded**. `R2` and `M3` are left unmatched (likely `R2`'s true partner
  wasn't detected, e.g. it was too dim in channel 1).

**Why `cKDTree` / MNN?** With hundreds of beads, brute-force all-pairs distance
checks are `O(N²)`; a k-d tree nearest-neighbour query is `O(N log N)`. Mutual
nearest-neighbour is the standard, lightweight way to get a clean
one-to-one correspondence from two roughly-aligned point clouds — a full
optimal assignment (e.g. the Hungarian algorithm, `O(N³)`) would be overkill
here since the coarse shift has already brought the clouds close together, and
MNN + a distance cutoff is what tools such as TrackMate/FIJI use for the same
bead-registration problem.

---

## Step 5 — Transform fitting with RANSAC

The matched `(reference, moving)` centre pairs are now used to fit a geometric
transform that maps the moving channel onto the reference frame.

**Coordinate flip:** skimage's transform classes and `warp()` work in `(x, y) =
(col, row)` order, while the corrector works in `(row, col)` order everywhere
else. So just before fitting, both arrays are flipped:

```python
src_xy = src[:, ::-1]   # (row, col) → (col, row)
dst_xy = dst[:, ::-1]
```

**RANSAC** (`skimage.measure.ransac`) then repeatedly: samples a minimal subset
of pairs (`min_samples=3`), fits a candidate transform of the requested type
(`affine`/`similarity`/`euclidean`/`translation`), and counts how many of the
*other* pairs agree with it within `residual_threshold=2.0 px`. The candidate
with the most agreeing pairs ("inliers") wins, and the final transform is
re-estimated from the inliers only. The RMS reported in the log/result is
computed over inliers only.

**Why RANSAC?** A few of the MNN matches from Step 4 can still be wrong (e.g. a
spurious detection near the edge that happened to have a mutual nearest
neighbour). A plain least-squares fit would let those few bad pairs pull the
whole transform off. RANSAC instead identifies and ignores them.

**Worked example:** 50 pairs are matched in Step 4, but 2 of them are spurious
— one bead was a noise spike near the image border, and its "match" sits about
`8 px` away from where the fitted transform predicts it should be (versus
`< 0.5 px` for the genuine pairs). RANSAC's inlier test (`residual_threshold =
2.0 px`) flags those 2 pairs as outliers, fits the final transform from the
remaining 48, and reports `n_pairs = 48`, `rms ≈ 0.32 px`.

---

## Step 6 — Applying the correction

`apply()` uses `skimage.transform.warp` to resample each non-reference channel
into the reference frame:

```python
corrected[ch] = warp(image[ch], tform.inverse, order=3, mode="constant", cval=0.0)
```

**Why `tform.inverse`?** `tform` was fitted to map *moving → reference*
coordinates. But `warp` needs the **inverse** mapping (*reference → moving*):
for every pixel `(r, c)` in the *output* (reference-frame) image, it must know
*where in the input (moving-channel) image* to sample from. `tform.inverse`
gives exactly that reference→moving mapping. Sampling uses bicubic
interpolation (`order=3`) for sub-pixel accuracy, and any output pixel that maps
outside the input frame is filled with `0` (`mode="constant", cval=0.0`) — no
data is invented.

**Worked example:** suppose channel 1 is shifted `(+2, −3) px` relative to the
reference (i.e. a bead truly at reference position `(100, 100)` appears at
`(102, 97)` in the raw channel-1 image). To undo this, output pixel `(100,
100)` must be filled by *sampling* channel 1 at `(102, 97)` — exactly what
`tform.inverse` encodes. After warping, that bead lands back at `(100, 100)` in
the corrected image, aligned with the reference channel.

If `crop=True`, the corrector also warps an all-ones mask through the same
transform for every channel, intersects the "valid" regions (`> 0.5`), and
crops the output to the largest rectangle that contains valid data in *every*
channel — removing the zero-filled borders that appear after shifting/rotating.

---

## Step 7 — Validation

`validate()` checks how good the measured correction actually is, using the
*same* bead calibration image (not new data):

1. Apply the stored correction to the calibration bead stack (`crop=False`).
2. Re-detect beads in every corrected channel (Steps 1–2 again).
3. Match corrected reference beads against corrected moving beads — this time
   with `shift = (0, 0)`, since the images should already be aligned.
4. Compute the residual distance for every matched pair and summarise it as
   `mean_error`, `median_error`, `max_error`, `std_error`, and `n_pairs`.

If the correction worked, matched beads should now sit almost exactly on top of
each other, so the residuals should be small:

| `mean_error`  | Quality                                         |
| ------------- | ----------------------------------------------- |
| < 0.3 px      | Excellent — diffraction-limited colocalization  |
| 0.3 – 1.0 px  | Acceptable for most applications                |
| > 1.0 px      | Something went wrong; revisit detection parameters |

A high residual after validation usually means too few/too noisy bead detections
going into Step 5 (check `smooth_sigma`, `threshold_rel`, `min_distance`) rather
than a problem with the fitting itself.
