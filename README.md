🚧 WIP 🚧

# microcal — microscopy calibration tools

[![CI](https://github.com/fdrgsp/microcal/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/microcal/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/microcal/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/microcal)

## Table of contents

- [Installation](#installation)
- [Chromatic Shift Correction](#chromatic-shift-correction)
  - [Method 1 — multi-colour beads](#method-1--multi-colour-beads)
  - [Method 2 — continuous structures (marker-free)](#method-2--continuous-structures-marker-free)
- [API Reference](#api-reference)

## Installation

### ▶ Using (uv) pip

Install `microcal` directly without cloning (Python 3.10 or greater):

```bash
(uv) pip install "git+https://github.com/fdrgsp/microcal"
```

To include the optional [ndv](https://github.com/pyapp-kit/ndv) viewer for displaying images:

```bash
(uv) pip install "git+https://github.com/fdrgsp/microcal[ndv]"
```

Or in a Jupyter notebook:

```bash
(uv) pip install "git+https://github.com/fdrgsp/microcal[ndv-jup]"
```

### ▶ Using uv (from source)

Clone the repository and install with `uv`:

```bash
git clone https://github.com/fdrgsp/microcal
cd microcal
uv sync
```

With the optional ndv viewer:

```bash
uv sync --extra ndv
```

Or in a Jupyter notebook:

```bash
uv sync --extra ndv-jup
```

## Chromatic shift correction

`microcal` provides **two interchangeable methods** to estimate the per-channel
transform. Both feed the same RANSAC fit and the same
`measure` → `validate` → `apply` → `save` / `from_json` workflow, and each has a
2-D (`ChromaticShiftCorrector`) and a 3-D (`ChromaticShiftCorrector3D`) variant.
Pick the one that matches your calibration sample:

| Method | `method=` | Best for | How correspondences are found |
| --- | --- | --- | --- |
| **1 — Beads** | `"beads"` (default) | A good-quality, sparse **multi-colour bead** sample | Detects isolated bead maxima and matches them between channels |
| **2 — Correlation** | `"correlation"` | A **continuous structure stained in several colours** (e.g. the same mitochondria imaged in two channels) — also works on beads | Tiles the image into blocks and measures each block's sub-pixel shift by FFT phase correlation |

For a detailed explanation of both pipelines and the underlying algorithms, see
the chromatic shift correction [documentation](chromatic_shift_correction.md).

### Method 1 — multi-colour beads

Use `method="beads"` (the default) with a sparse bead calibration image.

```python
import tifffile
from microcal import ChromaticShiftCorrector

# 1. Load a multi-channel bead image (C, H, W)
bead_img = tifffile.imread("beads.tiff")   # e.g. shape (2, 512, 512), uint16

# 2. Measure the chromatic shift — tune detection parameters here
csc = ChromaticShiftCorrector()
result = csc.measure(
    bead_img,
    reference_channel=0,
    smooth_sigma=3,
    min_distance=2,
    threshold_rel=0.5,
    match_max_distance=20,
    min_pairs=2,
    subpixel_refine=True,
    refine_radius=2,
    verbose=True,
)

# 3. Validate — re-detects beads in the corrected bead image and reports
# the residual displacement. Mean error < 0.3 px is excellent.
val = csc.validate()

# 4. Apply the correction to any sample image
sample_img = tifffile.imread("sample.tiff")   # same number of channels
corrected = csc.apply(sample_img, crop=True)
tifffile.imwrite("sample_corrected.tiff", corrected)

# ── Save & Load ──────────────────────────────────────────────────────
# Save the calibration for later use (transforms only, no bead image)
csc.save("calibration.json")

# Load a saved calibration and apply it without re-running measure().
csc2 = ChromaticShiftCorrector.from_json("calibration.json")
corrected = csc2.apply(sample_img, crop=True)
```

Runnable 2-D bead example (script or notebook):

```bash
uv run examples/example_correction.py
uvx juv run examples/example_correction.ipynb
```

**Volumetric (3-D).** For z-stacks shaped `(C, Z, Y, X)` use `ChromaticShiftCorrector3D`. The API mirrors
the 2-D corrector — `measure()` / `validate()` / `apply()` / `save()` / `from_json()` —
but fits a 4×4 (3-D) affine and warps with `scipy.ndimage.affine_transform`.
Because z-stacks are anisotropic (axial step ≫ lateral pixel), `smooth_sigma` and
`refine_radius` accept a per-axis `(z, y, x)` tuple, and an optional `voxel_size`
makes bead matching physically isotropic and reports residuals in physical units.

```python
import tifffile
from microcal import ChromaticShiftCorrector3D

# 1. Load a multi-channel bead volume (C, Z, Y, X)
bead_vol = tifffile.imread("beads_zstack.tiff")   # e.g. (2, 32, 512, 512)

# 2. Measure — per-axis sigma + optional voxel_size handle anisotropy
csc = ChromaticShiftCorrector3D()
csc.measure(
    bead_vol,
    smooth_sigma=(1.5, 2.0, 2.0),   # (sz, sy, sx)
    min_distance=4,
    threshold_rel=0.3,
    match_max_distance=15,
    refine_radius=(2, 3, 3),
    voxel_size=(0.5, 0.1, 0.1),     # optional (z, y, x), e.g. microns
)

# 3. Validate, then apply to any sample volume
csc.validate()
sample_vol = tifffile.imread("sample_zstack.tiff")
corrected = csc.apply(sample_vol, crop=True)

# 4. Save / load (JSON records the 4x4 transforms, ndim and voxel_size)
csc.save("calibration_3d.json")
csc2 = ChromaticShiftCorrector3D.from_json("calibration_3d.json")
```

Runnable 3-D bead example:

```bash
uv run examples/example_correction_3d.py
uvx juv run examples/example_correction_3d.ipynb
```

### Method 2 — continuous structures (marker-free)

When the sample has no isolated point sources — for example a **continuous
structure stained in several colours** (the same mitochondrial/membrane construct
imaged in two channels) — bead detection has nothing to localise. Pass
`method="correlation"` to either corrector to estimate the transform with
**marker-free block phase correlation** instead: the image is tiled into
overlapping blocks, each block's sub-pixel shift between channels is measured by
FFT phase correlation, and the field of local shifts is fit to the same affine
transform with RANSAC. The output (one matrix per channel) and the
`validate` / `apply` / `save` / `from_json` workflow are identical to the bead
method — only the correspondence step changes. It also works on bead samples.

```python
import tifffile
from microcal import ChromaticShiftCorrector

# A multi-channel image of a continuous structure (no beads needed)
img = tifffile.imread("structure.tiff")   # e.g. (2, 512, 512)

csc = ChromaticShiftCorrector()
csc.measure(
    img,
    method="correlation",
    block_size=128,        # block edge in px (tuple for per-axis sizes)
    block_overlap=0.5,     # 50 % overlap between adjacent blocks
    correlation_upsample=10,   # sub-pixel up-sampling factor
    min_correlation=0.1,   # drop flat / poorly-correlated blocks
)
csc.validate()                       # residual block shift after correction
corrected = csc.apply(img, crop=True)
```

The same `method="correlation"` option works for `ChromaticShiftCorrector3D`
(blocks become sub-volumes; `block_size` defaults to the anisotropic
`(16, 64, 64)`). The package ships `generate_structures_image` /
`generate_structures_image_3d` to create synthetic continuous-structure samples
for experimentation:

```bash
uv run examples/example_correction_correlation.py        # 2-D
uv run examples/example_correction_correlation_3d.py     # 3-D
```

## API reference

See the full [API reference](API.md) for all parameters and return types.
