# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "microcal[ndv] @ git+https://github.com/fdrgsp/microcal",
# ]
# ///

"""Measure, validate, save, load, and apply chromatic shift correction."""

import ndv

from microcal import ChromaticShiftCorrector, generate_beads_image

# generate a 2-channel synthetic beads image
beads_img, _ = generate_beads_image(
    n_channels=2,
    shape=(512, 512),
    n_beads=50,
    bead_sigma=2,
    bead_intensity=60.0,
    bit_depth=16,
    offset=100,
    shifts=[(0, 0), (1.5, -2.5)],
    rotations=[0, 5],
    scales=[(1, 1), (1.05, 0.95)],
    snr=8,
    seed=42,
)

# visualize the synthetic beads image with ndv
ndv.imshow(
    beads_img,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
)

# measure the chromatic shift — all detection parameters go here
csc = ChromaticShiftCorrector()
results = csc.measure(
    beads_img,
    reference_channel=0,
    smooth_sigma=3,
    min_distance=2,
    threshold_rel=0.5,
    match_max_distance=50,
    min_pairs=2,
    subpixel_refine=True,
    refine_radius=2,
    verbose=True,
)

# detection_image and pairs_image are always populated by measure()
assert results.detection_image is not None
assert results.pairs_image is not None

# visualize the detected beads in the first (reference) channel (beads + masks)
ch1_det = results.detection_image[:2, :, :]
# in this image, 0 is the reference channel, and 1 is the beads mask
ndv.imshow(
    ch1_det,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "gray"}},
)

# visualize the detected beads in the second channel (beads + masks)
ch2_det = results.detection_image[2:4, :, :]
# in this image, 2 is the second channel, and 3 is the beads mask
ndv.imshow(
    ch2_det,
    channel_mode="composite",
    luts={0: {"cmap": "magenta"}, 1: {"cmap": "gray"}},
)

# visualize the matched bead pairs between the two channels
ndv.imshow(results.pairs_image.astype("uint16"), default_lut={"cmap": "glasbey"})

# validate: re-detects beads on the corrected bead image and reports residuals
val = csc.validate()

# apply the correction to a sample image
# (here we reuse the bead image for demonstration)
image_corr = csc.apply(image_or_stack=beads_img, crop=True)
ndv.imshow(
    image_corr,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
)

# you can also save the calibration parameters and transform to a JSON file that can be
# loaded for later use without needing to re-run the measurement step
# csc.save("calibration.json")

# load a previously saved calibration and apply it without re-running measure().
# csc2 = ChromaticShiftCorrector.from_json("calibration.json")
# image_corr2 = csc2.apply(image_or_stack=beads_img, crop=True)
# ndv.imshow(
#     image_corr2,
#     channel_mode="composite",
#     luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
# )
