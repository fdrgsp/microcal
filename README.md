🚧 WIP 🚧

# microcal — microscopy calibration tools

[![CI](https://github.com/fdrgsp/microcal/actions/workflows/ci.yml/badge.svg)](https://github.com/fdrgsp/microcal/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/fdrgsp/microcal/branch/main/graph/badge.svg)](https://codecov.io/gh/fdrgsp/microcal)

## Table of contents

- [Installation](#installation)
- [Chromatic Shift Correction](#chromatic-shift-correction)
  - [Example Usage](#example-usage)
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

For a detail explanation of the chromatic shift correction workflow and the underlying algorithms, see the chromatic shift correction [documentation](chromatic_shift_correction.md).

### Example usage

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

For a complete runnable example see the [examples/](examples/) folder:

```bash
# Python script
uv run examples/example_correction.py

# Jupyter notebook
uvx juv run examples/example_correction.ipynb
```

### Volumetric (3-D) correction

For z-stacks shaped `(C, Z, Y, X)` use `ChromaticShiftCorrector3D`. The API mirrors
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

Runnable 3-D example:

```bash
uv run examples/example_correction_3d.py
uvx juv run examples/example_correction_3d.ipynb
```

## API reference

See the full [API reference](API.md) for all parameters and return types.
