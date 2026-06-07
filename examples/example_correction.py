# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "microcal[ndv] @ git+https://github.com/fdrgsp/microcal",
# ]
# ///

"""Measure, validate, and apply chromatic shift correction, visualizing each step."""

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

# initialize the chromatic shift corrector with appropriate parameters
csc = ChromaticShiftCorrector(
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

# measure the chromatic shift on the synthetic beads image
results = csc.measure(beads_img)
# detection_image and pairs_image are always populated by measure()
assert results.detection_image is not None
assert results.pairs_image is not None

# visualize the detected beads in the first (reference) channel
ch1_det = results.detection_image[:2, :, :]
# in this image, 0 is the reference channel, and 1 is the beads mask
ndv.imshow(
    ch1_det,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "gray"}},
)

# visualize the detected beads in the second channel
ch2_det = results.detection_image[2:4, :, :]
# in this image, 2 is the second channel, and 3 is the beads mask
ndv.imshow(
    ch2_det,
    channel_mode="composite",
    luts={0: {"cmap": "magenta"}, 1: {"cmap": "gray"}},
)

# visualize the matched bead pairs between the two channels
ndv.imshow(results.pairs_image.astype("uint16"), default_lut={"cmap": "glasbey"})

# run the validation
val = csc.validate()

# apply the measured chromatic shift correction to another image (in this case,
# the same synthetic beads image)
image_corr = csc.apply(image_or_stack=beads_img, result=results, crop=True)
ndv.imshow(
    image_corr,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
)
