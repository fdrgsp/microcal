from microcal import ChromaticShiftCorrector
from rich import print
import ndv
from microcal._sample_generator import generate_beads_image

img, meta = generate_beads_image(
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
ndv.imshow(img)

sc = ChromaticShiftCorrector(
    reference_channel=0,
    smooth_sigma=3,
    min_distance=2,
    threshold_rel=0.5,
    match_max_distance=20,
    min_pairs=2,
    subpixel_refine=True,
    refine_radius=2,
)

results = sc.measure(img)
print(results)

val = sc.validate()
print(val)

image_corr = sc.apply(image_or_stack=img, result=results, crop=True)
ndv.imshow(image_corr)
