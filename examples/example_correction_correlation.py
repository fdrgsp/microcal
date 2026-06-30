# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "microcal[ndv] @ git+https://github.com/fdrgsp/microcal",
# ]
# ///

"""Marker-free chromatic shift correction on a continuous-structure sample.

Beads are not required: ``method="correlation"`` estimates the per-channel
transform from a continuous structure (e.g. the same construct stained in two
colours) using block phase correlation.
"""

import ndv
import tifffile

from microcal import ChromaticShiftCorrector, generate_structures_image

# generate a 2-channel synthetic continuous-structure image (no beads).
# The signal is a filament/membrane-like network with a per-channel chromatic
# shift, rotation and slight scale — exactly where bead detection has nothing to
# localise but block correlation works directly on the image content.
struct_img, _ = generate_structures_image(
    n_channels=2,
    shape=(512, 512),
    structure_scale=16.0,
    contour_width=0.3,
    bit_depth=16,
    offset=100,
    shifts=[(0, 0), (3.5, -2.5)],
    rotations=[0, 1.5],
    scales=[(1, 1), (1.01, 0.99)],
    snr=15,
    seed=3,
    background=0.05,
)

# visualize the synthetic structure image with ndv
ndv.imshow(
    struct_img,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
)

# measure the chromatic shift with the marker-free correlation method —
# all correlation parameters go here (no bead-detection parameters needed)
csc = ChromaticShiftCorrector()
results = csc.measure(
    struct_img,
    reference_channel=0,
    method="correlation",
    block_size=128,        # block edge in px (tuple for per-axis sizes)
    block_overlap=0.5,     # 50 % overlap between adjacent blocks
    correlation_upsample=10,
    min_correlation=0.1,   # drop flat / poorly-correlated blocks
    verbose=True,
)

# detection_image / pairs_image are still populated — for the correlation method
# the "points" are the block centres used for the per-channel correspondences.
assert results.pairs_image is not None
ndv.imshow(results.pairs_image.astype("uint16"), default_lut={"cmap": "glasbey"})

# validate: re-runs block correlation on the corrected image and reports the
# residual per-block shift (should be well under a pixel)
val = csc.validate()

# apply the correction to a sample image (here we reuse the structure image)
image_corr = csc.apply(image_or_stack=struct_img, crop=True)

# save and visualize the corrected image
tifffile.imwrite("corrected_structure.tif", image_corr)
ndv.imshow(
    image_corr,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
)

# the calibration (including method and correlation parameters) can be saved and
# reloaded without re-running measure(), exactly like the bead workflow:
# csc.save("calibration_correlation.json")
# csc2 = ChromaticShiftCorrector.from_json("calibration_correlation.json")
# image_corr2 = csc2.apply(image_or_stack=struct_img, crop=True)
