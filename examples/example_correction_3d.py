# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "microcal[ndv] @ git+https://github.com/fdrgsp/microcal",
# ]
# ///

"""Measure, validate, save, load, and apply 3-D chromatic shift correction."""

import ndv

from microcal import ChromaticShiftCorrector3D, generate_beads_image_3d

# generate a 2-channel synthetic 3-D beads volume (C, Z, Y, X) with an
# anisotropic PSF (axially elongated, as in a real z-stack)
scale_z, scale_y, scale_x = (0.5, 0.1, 0.1)

beads_vol, _ = generate_beads_image_3d(
    n_channels=2,
    shape=(64, 256, 256),
    n_beads=60,
    bead_sigma=(1.5, 2.0, 2.0),  # (sz, sy, sx)
    bead_intensity=60.0,
    bit_depth=16,
    offset=100,
    shifts=[(0, 0, 0), (2.0, 1.5, -2.5)],  # (dz, dy, dx)
    rotations=[0, 3],
    scales=[(1, 1, 1), (1.0, 1.02, 0.98)],
    snr=10,
    seed=42,
)

# visualize the synthetic beads volume with ndv (z is a slider)
ndv.imshow(
    beads_vol,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)

# measure the chromatic shift — all detection parameters go here.
# smooth_sigma / refine_radius accept a per-axis (z, y, x) tuple for anisotropy,
# and an optional voxel_size makes bead matching physically isotropic.
csc = ChromaticShiftCorrector3D()
results = csc.measure(
    beads_vol,
    reference_channel=0,
    smooth_sigma=(1.5, 2.0, 2.0),
    min_distance=4,
    threshold_rel=0.3,
    match_max_distance=15,
    min_pairs=4,
    subpixel_refine=True,
    refine_radius=(2, 3, 3),
    voxel_size=(scale_z, scale_y, scale_x),  # optional (z, y, x) in microns
    verbose=True,
)

# detection_image (2*C, Z, Y, X) and pairs_image (Z, Y, X) are always populated
assert results.detection_image is not None
assert results.pairs_image is not None

# visualize the detected beads in the reference channel (volume + sphere masks)
ch1_det = results.detection_image[:2]
ndv.imshow(
    ch1_det,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "gray"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)

# visualize the detected beads in the shifted channel (volume + sphere masks)
ch2_det = results.detection_image[2:4]
ndv.imshow(
    ch2_det,
    channel_mode="composite",
    luts={0: {"cmap": "magenta"}, 1: {"cmap": "gray"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)

# visualize the matched bead pairs between the two channels (3-D label volume)
ndv.imshow(
    results.pairs_image.astype("uint16"),
    default_lut={"cmap": "glasbey"},
    scales={0: scale_z, 1: scale_y, 2: scale_x},  # (Z, Y, X)
)

# validate: re-detects beads on the corrected volume and reports residuals.
# With voxel_size set, per-axis (voxel) and physical-unit errors are reported.
val = csc.validate()

# apply the correction to a sample volume (here we reuse the bead volume)
vol_corr = csc.apply(image_or_stack=beads_vol, crop=True)
ndv.imshow(
    vol_corr,
    channel_mode="composite",
    luts={0: {"cmap": "green"}, 1: {"cmap": "magenta"}},
    scales={0: 1.0, 1: scale_z, 2: scale_y, 3: scale_x},  # (C, Z, Y, X)
)
# you can also save the calibration parameters and transform to a JSON file that
# can be loaded later without re-running the measurement step
# csc.save("calibration_3d.json")

# load a previously saved calibration and apply it without re-running measure().
# csc2 = ChromaticShiftCorrector3D.from_json("calibration_3d.json")
# vol_corr2 = csc2.apply(image_or_stack=beads_vol, crop=True)
