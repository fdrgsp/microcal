# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "microcal[ndv] @ git+https://github.com/fdrgsp/microcal",
# ]
# ///

"""Marker-free 3-D chromatic shift correction on a continuous-structure volume.

The volumetric counterpart of ``example_correction_correlation.py``: no beads
are required — ``method="correlation"`` estimates the per-channel 4x4 transform
from a continuous 3-D structure (e.g. the same construct stained in two colours)
using block phase correlation over sub-volumes.
"""

import ndv

from microcal import ChromaticShiftCorrector3D, generate_structures_image_3d

# generate a 2-channel synthetic continuous-structure volume (C, Z, Y, X).
# The signal is a membrane-like network (the bright level-set surfaces of a
# smoothed 3-D random field) with a per-channel chromatic shift, rotation and
# slight anisotropic scale — the case where bead detection has nothing to
# localise but block correlation works directly on the image content.
scale_z, scale_y, scale_x = (0.3, 0.1, 0.1)

struct_vol, _ = generate_structures_image_3d(
    n_channels=2,
    shape=(48, 256, 256),
    # coarser, thinner structures keep the volume sparse (smaller axial sigma
    # for anisotropic stacks). Too-dense structures hurt phase correlation.
    structure_scale=(3.0, 10.0, 10.0),
    contour_width=0.15,
    bit_depth=16,
    offset=100,
    shifts=[(0, 0, 0), (1.5, 3.0, -2.0)],  # (dz, dy, dx)
    rotations=[0, 1.5],
    scales=[(1, 1, 1), (1.0, 1.01, 0.99)],
    snr=20,
    seed=7,
    background=0.05,
)

# visualize the synthetic structure volume with ndv (z is a slider)
ndv.imshow(
    struct_vol,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)

# measure the chromatic shift with the marker-free correlation method —
# all correlation parameters go here (no bead-detection parameters needed).
# block_size is per-axis (bz, by, bx); the default (16, 64, 64) already uses a
# thinner axial block for anisotropic stacks. An optional voxel_size makes
# validate() additionally report residuals in physical units.
csc = ChromaticShiftCorrector3D()
results = csc.measure(
    struct_vol,
    reference_channel=0,
    method="correlation",
    block_size=(16, 64, 64),  # (bz, by, bx)
    block_overlap=0.5,        # 50 % overlap between adjacent blocks
    correlation_upsample=10,
    min_correlation=0.1,      # drop flat / poorly-correlated blocks
    voxel_size=(scale_z, scale_y, scale_x),  # optional (z, y, x) in microns
    verbose=True,
)

# pairs_image (Z, Y, X) is still populated — for the correlation method the
# "points" are the block centres used for the per-channel correspondences.
assert results.pairs_image is not None
ndv.imshow(
    results.pairs_image.astype("uint16"),
    default_lut={"cmap": "glasbey"},
    scales={0: scale_z, 1: scale_y, 2: scale_x},  # (Z, Y, X)
)

# validate: re-runs block correlation on the corrected volume and reports the
# residual per-block shift. With voxel_size set, per-axis (voxel) and
# physical-unit errors are reported too.
val = csc.validate()

# apply the correction to a sample volume (here we reuse the structure volume)
vol_corr = csc.apply(image_or_stack=struct_vol, crop=True)
ndv.imshow(
    vol_corr,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)

# the calibration (including method and correlation parameters) can be saved and
# reloaded without re-running measure(), exactly like the bead workflow:
# csc.save("calibration_correlation_3d.json")
# csc2 = ChromaticShiftCorrector3D.from_json("calibration_correlation_3d.json")
# vol_corr2 = csc2.apply(image_or_stack=struct_vol, crop=True)
