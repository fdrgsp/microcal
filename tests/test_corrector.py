"""
pytest tests for ChromaticShiftCorrector.

Synthetic bead images from _sample_generator are used as ground-truth
calibration data so tests are self-contained and reproducible.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import tifffile
from skimage.transform import AffineTransform

from microcal import ChannelTransform, ChromaticShiftCorrector, CorrectionResult
from microcal._sample_generator import generate_beads_image

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Use scope="module" so expensive measure() calls happen once per test session.


@pytest.fixture(scope="module")  #  type: ignore
def bead_2ch() -> tuple[np.ndarray, dict]:
    """2-channel, translation-only shift, high SNR — fast and deterministic."""
    img, meta = generate_beads_image(
        n_channels=2,
        shape=(256, 256),
        n_beads=30,
        bead_sigma=2,
        bead_intensity=60,
        bit_depth=16,
        offset=100,
        shifts=[(0, 0), (4.0, -3.0)],
        rotations=[0, 0],
        scales=[(1, 1), (1, 1)],
        snr=20,
        seed=42,
    )
    return img, meta


@pytest.fixture(scope="module")  #  type: ignore
def bead_3ch() -> tuple[np.ndarray, dict]:
    """3-channel with rotation and anisotropic scale."""
    img, meta = generate_beads_image(
        n_channels=3,
        shape=(256, 256),
        n_beads=30,
        bead_sigma=2,
        bead_intensity=60,
        bit_depth=16,
        offset=100,
        shifts=[(0, 0), (4.0, -3.0), (-2.0, 2.0)],
        rotations=[0, 3, -2],
        scales=[(1, 1), (1.02, 0.98), (0.98, 1.02)],
        snr=20,
        seed=42,
    )
    return img, meta


@pytest.fixture(scope="module")  #  type: ignore
def sc_2ch(bead_2ch: tuple[np.ndarray, dict]) -> ChromaticShiftCorrector:
    """Corrector already calibrated on the 2-channel bead stack."""
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector()
    sc.measure(
        img,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=15,
        min_pairs=3,
        refine_radius=3,
    )
    return sc


# ---------------------------------------------------------------------------
# measure() tests
# ---------------------------------------------------------------------------


def test_measure_returns_correction_result(bead_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector()
    result = sc.measure(
        img,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=15,
        min_pairs=3,
        refine_radius=3,
    )
    assert isinstance(result, CorrectionResult)
    assert result.reference_channel == 0
    assert 1 in result.transforms
    assert isinstance(result.transforms[1], ChannelTransform)


def test_measure_rms_below_one_pixel(sc_2ch: ChromaticShiftCorrector) -> None:
    assert sc_2ch._result is not None
    rms = sc_2ch._result.transforms[1].rms_residual
    assert rms is not None
    assert rms < 1.0


def test_measure_sufficient_pairs(sc_2ch: ChromaticShiftCorrector) -> None:
    assert sc_2ch._result is not None
    assert sc_2ch._result.transforms[1].n_pairs >= 10


def test_measure_stores_bead_stack(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    assert sc_2ch._bead_stack is not None
    assert sc_2ch._bead_stack.shape == img.shape


# ---------------------------------------------------------------------------
# apply() — input form tests
# ---------------------------------------------------------------------------


def test_apply_ndarray_shape_preserved(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=False)
    assert out.shape == img.shape


def test_apply_dtype_preserved(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=False)
    assert out.dtype == img.dtype


def test_apply_crop_reduces_spatial_size(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=True)
    assert out.shape[0] == img.shape[0]  # channels unchanged
    assert out.shape[1] <= img.shape[1]
    assert out.shape[2] <= img.shape[2]


def test_apply_list_of_ndarrays(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    planes = [img[i] for i in range(img.shape[0])]
    out = sc_2ch.apply(planes, crop=False)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_apply_list_of_paths(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(img.shape[0]):
            p = os.path.join(d, f"ch{i}.tiff")
            tifffile.imwrite(p, img[i])
            paths.append(p)
        out = sc_2ch.apply(paths, crop=False)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_apply_list_of_paths_with_crop(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(img.shape[0]):
            p = os.path.join(d, f"ch{i}.tiff")
            tifffile.imwrite(p, img[i])
            paths.append(p)
        out = sc_2ch.apply(paths, crop=True)
    assert out.shape[0] == img.shape[0]
    assert out.shape[1] <= img.shape[1]


def test_apply_bad_list_item_raises() -> None:
    sc = ChromaticShiftCorrector()
    sc._result = CorrectionResult(reference_channel=0)
    with pytest.raises(TypeError):
        sc.apply([42, 43])  # not str or ndarray


def test_apply_wrong_ndim_raises() -> None:
    sc = ChromaticShiftCorrector()
    sc._result = CorrectionResult(reference_channel=0)
    with pytest.raises(ValueError):
        sc.apply([np.zeros((3, 4, 5))])  # 3-D item instead of 2-D


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------


def test_validate_mean_error_below_half_pixel(sc_2ch: ChromaticShiftCorrector) -> None:
    stats = sc_2ch.validate()
    assert stats[1]["mean_error"] < 0.5


def test_validate_has_sufficient_pairs(sc_2ch: ChromaticShiftCorrector) -> None:
    stats = sc_2ch.validate()
    assert stats[1]["n_pairs"] >= 10


def test_validate_before_measure_raises() -> None:
    sc = ChromaticShiftCorrector()
    with pytest.raises(RuntimeError):
        sc.validate()


# ---------------------------------------------------------------------------
# save() / from_json() tests
# ---------------------------------------------------------------------------


def test_save_creates_file(sc_2ch: ChromaticShiftCorrector) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc_2ch.save(path)
        assert os.path.isfile(path)


def test_from_json_apply_matches_original(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead_2ch
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc_2ch.save(path)
        sc2 = ChromaticShiftCorrector.from_json(path)

    out_original = sc_2ch.apply(img, crop=False)
    out_loaded = sc2.apply(img, crop=False)
    np.testing.assert_array_equal(out_original, out_loaded)


def test_from_json_reference_channel_preserved(sc_2ch: ChromaticShiftCorrector) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc_2ch.save(path)
        sc2 = ChromaticShiftCorrector.from_json(path)
    assert sc2._result is not None
    assert sc2._result.reference_channel == sc_2ch._result.reference_channel  # type: ignore[union-attr]


def test_from_json_validate_raises() -> None:
    """from_json does not restore the bead stack, so validate() must fail."""
    sc = ChromaticShiftCorrector()
    sc._result = CorrectionResult(
        reference_channel=0,
        transforms={
            1: ChannelTransform(
                channel=1,
                transform=AffineTransform(),
                rms_residual=0.1,
                n_pairs=20,
            )
        },
    )
    with pytest.raises(RuntimeError):
        sc.validate()


def test_from_json_measure_params_restored(sc_2ch: ChromaticShiftCorrector) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc_2ch.save(path)
        sc2 = ChromaticShiftCorrector.from_json(path)
    assert sc2.smooth_sigma == sc_2ch.smooth_sigma
    assert sc2.min_distance == sc_2ch.min_distance
    assert sc2.threshold_rel == sc_2ch.threshold_rel
    assert sc2.match_max_distance == sc_2ch.match_max_distance
    assert sc2.transform_type == sc_2ch.transform_type


def test_save_before_measure_raises() -> None:
    sc = ChromaticShiftCorrector()
    with pytest.raises(RuntimeError):
        sc.save("/tmp/should_not_exist.json")


# ---------------------------------------------------------------------------
# Multi-channel (3-channel) tests
# ---------------------------------------------------------------------------


def test_three_channel_measure_and_validate(bead_3ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead_3ch
    sc = ChromaticShiftCorrector()
    sc.measure(
        img,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=20,
        min_pairs=3,
        refine_radius=3,
    )
    assert sc._result is not None
    assert 1 in sc._result.transforms
    assert 2 in sc._result.transforms

    stats = sc.validate()
    assert stats[1]["mean_error"] < 1.0
    assert stats[2]["mean_error"] < 1.0


def test_three_channel_apply_shape(bead_3ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead_3ch
    sc = ChromaticShiftCorrector()
    sc.measure(
        img,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=20,
        min_pairs=3,
        refine_radius=3,
    )
    out = sc.apply(img, crop=False)
    assert out.shape == img.shape
