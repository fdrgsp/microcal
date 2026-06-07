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

from microcal import ChannelTransform, ChromaticShiftCorrector, CorrectionResult
from microcal._sample_generator import generate_beads_image

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

# Use scope="module" so expensive measure() calls happen once per test session.

@pytest.fixture(scope="module")
def bead_2ch():
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


@pytest.fixture(scope="module")
def bead_3ch():
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


@pytest.fixture(scope="module")
def sc_2ch(bead_2ch):
    """Corrector already calibrated on the 2-channel bead stack."""
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector(
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=15,
        min_pairs=3,
        refine_radius=3,
    )
    sc.measure(img)
    return sc


# ---------------------------------------------------------------------------
# measure() tests
# ---------------------------------------------------------------------------

def test_measure_returns_correction_result(bead_2ch):
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector(
        smooth_sigma=2, min_distance=5, threshold_rel=0.3,
        match_max_distance=15, min_pairs=3, refine_radius=3,
    )
    result = sc.measure(img)
    assert isinstance(result, CorrectionResult)
    assert result.reference_channel == 0
    assert 1 in result.transforms
    assert isinstance(result.transforms[1], ChannelTransform)


def test_measure_rms_below_one_pixel(sc_2ch):
    rms = sc_2ch._result.transforms[1].rms_residual
    assert rms is not None
    assert rms < 1.0


def test_measure_sufficient_pairs(sc_2ch):
    assert sc_2ch._result.transforms[1].n_pairs >= 10


def test_measure_stores_bead_stack(sc_2ch, bead_2ch):
    img, _ = bead_2ch
    assert sc_2ch._bead_stack is not None
    assert sc_2ch._bead_stack.shape == img.shape


# ---------------------------------------------------------------------------
# apply() — input form tests
# ---------------------------------------------------------------------------

def test_apply_ndarray_shape_preserved(sc_2ch, bead_2ch):
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=False)
    assert out.shape == img.shape


def test_apply_dtype_preserved(sc_2ch, bead_2ch):
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=False)
    assert out.dtype == img.dtype


def test_apply_crop_reduces_spatial_size(sc_2ch, bead_2ch):
    img, _ = bead_2ch
    out = sc_2ch.apply(img, crop=True)
    assert out.shape[0] == img.shape[0]       # channels unchanged
    assert out.shape[1] <= img.shape[1]
    assert out.shape[2] <= img.shape[2]


def test_apply_list_of_ndarrays(sc_2ch, bead_2ch):
    img, _ = bead_2ch
    planes = [img[i] for i in range(img.shape[0])]
    out = sc_2ch.apply(planes, crop=False)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_apply_list_of_paths(sc_2ch, bead_2ch):
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


def test_apply_list_of_paths_with_crop(sc_2ch, bead_2ch):
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


def test_apply_bad_list_item_raises():
    sc = ChromaticShiftCorrector()
    sc._result = CorrectionResult(reference_channel=0)
    with pytest.raises(TypeError):
        sc.apply([42, 43])  # not str or ndarray


def test_apply_wrong_ndim_raises():
    sc = ChromaticShiftCorrector()
    sc._result = CorrectionResult(reference_channel=0)
    with pytest.raises(ValueError):
        sc.apply([np.zeros((3, 4, 5))])  # 3-D item instead of 2-D


# ---------------------------------------------------------------------------
# validate() tests
# ---------------------------------------------------------------------------

def test_validate_mean_error_below_half_pixel(sc_2ch):
    stats = sc_2ch.validate()
    assert stats[1]["mean_error"] < 0.5


def test_validate_has_sufficient_pairs(sc_2ch):
    stats = sc_2ch.validate()
    assert stats[1]["n_pairs"] >= 10


def test_validate_before_measure_raises():
    sc = ChromaticShiftCorrector()
    with pytest.raises(RuntimeError):
        sc.validate()


# ---------------------------------------------------------------------------
# Multi-channel (3-channel) tests
# ---------------------------------------------------------------------------

def test_three_channel_measure_and_validate(bead_3ch):
    img, _ = bead_3ch
    sc = ChromaticShiftCorrector(
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=20,
        min_pairs=3,
        refine_radius=3,
    )
    sc.measure(img)
    assert 1 in sc._result.transforms
    assert 2 in sc._result.transforms

    stats = sc.validate()
    assert stats[1]["mean_error"] < 1.0
    assert stats[2]["mean_error"] < 1.0


def test_three_channel_apply_shape(bead_3ch):
    img, _ = bead_3ch
    sc = ChromaticShiftCorrector(
        smooth_sigma=2, min_distance=5, threshold_rel=0.3,
        match_max_distance=20, min_pairs=3, refine_radius=3,
    )
    sc.measure(img)
    out = sc.apply(img, crop=False)
    assert out.shape == img.shape
