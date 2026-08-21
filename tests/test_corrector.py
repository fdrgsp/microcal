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


# ---------------------------------------------------------------------------
# CorrectionResult.__repr__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "rms,expected",
    [
        (0.123, "0.123px"),
        (None, "translation fallback"),
    ],
)
def test_correction_result_repr(rms: float | None, expected: str) -> None:
    result = CorrectionResult(
        reference_channel=0,
        transforms={
            1: ChannelTransform(
                channel=1,
                transform=AffineTransform(),
                rms_residual=rms,
                n_pairs=5 if rms is not None else 0,
            )
        },
    )
    assert expected in repr(result)


# ---------------------------------------------------------------------------
# measure() — additional edge cases
# ---------------------------------------------------------------------------


def test_measure_wrong_ndim_raises(bead_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector()
    with pytest.raises(ValueError, match="3-D"):
        sc.measure(img[0])  # 2-D slice


def test_measure_verbose_configures_logger(bead_2ch: tuple[np.ndarray, dict]) -> None:
    import logging

    log = logging.getLogger("microcal._chromatic_shift_corrector")
    log.handlers.clear()
    sc = ChromaticShiftCorrector()
    sc.measure(
        bead_2ch[0],
        verbose=True,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=15,
        min_pairs=3,
        refine_radius=3,
    )
    assert log.level == logging.INFO
    log.handlers.clear()


def test_measure_translation_fallback(bead_2ch: tuple[np.ndarray, dict]) -> None:
    """min_pairs >> actual pairs triggers the translation-only fallback."""
    img, _ = bead_2ch
    sc = ChromaticShiftCorrector()
    with pytest.warns(UserWarning, match="Falling back to translation"):
        result = sc.measure(
            img,
            smooth_sigma=2,
            min_distance=5,
            threshold_rel=0.3,
            match_max_distance=15,
            min_pairs=9999,
            refine_radius=3,
        )
    ct = result.transforms[1]
    assert ct.rms_residual is None
    assert ct.n_pairs == 0


def test_measure_blank_image_no_beads() -> None:
    """Blank image → no peaks → coarse-shift zeros → translation fallback."""
    blank = np.zeros((2, 64, 64), dtype=np.uint16)
    sc = ChromaticShiftCorrector()
    with pytest.warns(UserWarning, match="Falling back to translation"):
        result = sc.measure(blank, threshold_rel=0.01, min_pairs=1)
    assert isinstance(result, CorrectionResult)
    assert result.transforms[1].rms_residual is None


# ---------------------------------------------------------------------------
# apply() — additional edge cases
# ---------------------------------------------------------------------------


def test_apply_with_explicit_result(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    """Passing result explicitly exercises the _resolve_result early-return branch."""
    img, _ = bead_2ch
    assert sc_2ch._result is not None
    out_implicit = sc_2ch.apply(img, crop=False)
    out_explicit = sc_2ch.apply(img, sc_2ch._result, crop=False)
    np.testing.assert_array_equal(out_implicit, out_explicit)


def test_apply_2d_ndarray_raises(sc_2ch: ChromaticShiftCorrector) -> None:
    with pytest.raises(ValueError, match="3-D"):
        sc_2ch.apply(np.zeros((64, 64)))


def test_apply_extra_channel_not_in_transforms(
    sc_2ch: ChromaticShiftCorrector, bead_2ch: tuple[np.ndarray, dict]
) -> None:
    """A 3-channel image against a 2-channel calibration: ch2 has no transform."""
    img, _ = bead_2ch
    three_ch = np.concatenate([img, img[:1]], axis=0)  # (3, H, W)
    out = sc_2ch.apply(three_ch, crop=False)
    assert out.shape == three_ch.shape
    np.testing.assert_array_equal(out[2], three_ch[2])  # ch2 untouched


def test_apply_empty_list_raises(sc_2ch: ChromaticShiftCorrector) -> None:
    with pytest.raises(TypeError, match="non-empty"):
        sc_2ch.apply([])


# ---------------------------------------------------------------------------
# validate() — additional edge cases
# ---------------------------------------------------------------------------


def test_validate_verbose(bead_2ch: tuple[np.ndarray, dict]) -> None:
    """verbose=True should reach _print_validation without error."""
    import logging

    img, _ = bead_2ch
    log = logging.getLogger("microcal._chromatic_shift_corrector")
    log.handlers.clear()
    sc = ChromaticShiftCorrector()
    sc.measure(
        img,
        verbose=True,
        smooth_sigma=2,
        min_distance=5,
        threshold_rel=0.3,
        match_max_distance=15,
        min_pairs=3,
        refine_radius=3,
    )
    stats = sc.validate()
    assert 1 in stats
    log.handlers.clear()


def test_validate_zero_pairs() -> None:
    """Blank image → no beads after correction → zero-pairs branch in validate."""
    blank = np.zeros((2, 64, 64), dtype=np.uint16)
    sc = ChromaticShiftCorrector()
    with pytest.warns(UserWarning):
        sc.measure(blank, threshold_rel=0.01, min_pairs=1)
    stats = sc.validate()
    assert stats[1]["n_pairs"] == 0
    assert np.isnan(stats[1]["mean_error"])


# ---------------------------------------------------------------------------
# _match_beads — internal edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "ref,mov",
    [
        (np.empty((0, 2)), np.array([[10.0, 20.0]])),
        (np.array([[10.0, 20.0]]), np.empty((0, 2))),
    ],
)
def test_match_beads_empty_input(
    sc_2ch: ChromaticShiftCorrector, ref: np.ndarray, mov: np.ndarray
) -> None:
    src, dst = sc_2ch._match_beads(ref, mov)
    assert src.shape == (0, 2)
    assert dst.shape == (0, 2)


def test_match_beads_all_rejected_by_distance(
    sc_2ch: ChromaticShiftCorrector,
) -> None:
    """Pairs exist but all exceed match_max_distance → empty result."""
    ref = np.array([[0.0, 0.0]])
    mov = np.array([[1000.0, 1000.0]])  # distance >> match_max_distance (15)
    src, dst = sc_2ch._match_beads(ref, mov, shift=(0.0, 0.0))
    assert src.shape == (0, 2)
    assert dst.shape == (0, 2)


# ---------------------------------------------------------------------------
# generate_beads_image — validation errors and None defaults
# ---------------------------------------------------------------------------

_BASE_GEN_KWARGS: dict = {
    "n_channels": 2,
    "shape": (64, 64),
    "n_beads": 5,
    "bead_sigma": 2,
    "bead_intensity": 60,
    "bit_depth": 16,
    "offset": 10,
    "shifts": None,
    "rotations": None,
    "scales": None,
    "snr": 10,
    "seed": 0,
}


@pytest.mark.parametrize(  # type: ignore[untyped-decorator]
    "override,exc_type,match",
    [
        ({"bead_intensity": 0.0}, ValueError, "bead_intensity"),
        ({"bead_intensity": 101.0}, ValueError, "bead_intensity"),
        ({"bit_depth": 32}, ValueError, "bit_depth"),
        ({"offset": 1.5}, TypeError, "offset"),
        ({"offset": -1}, ValueError, "offset"),
    ],
)
def test_generate_beads_invalid_params(
    override: dict, exc_type: type, match: str
) -> None:
    kwargs = {**_BASE_GEN_KWARGS, **override}
    with pytest.raises(exc_type, match=match):
        generate_beads_image(**kwargs)


def test_generate_beads_none_defaults() -> None:
    """shifts/rotations/scales=None should use identity defaults."""
    img, _ = generate_beads_image(**_BASE_GEN_KWARGS)
    assert img.shape == (2, 64, 64)
    assert img.dtype == np.uint16
