"""
pytest tests for the marker-free correlation method of ChromaticShiftCorrector.

Synthetic continuous-structure images (no isolated beads) from
`generate_structures_image` are the target use-case for
`measure(method="correlation")`.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from skimage.registration import phase_cross_correlation

from microcal import (
    ChromaticShiftCorrector,
    CorrectionResult,
    generate_beads_image,
    generate_structures_image,
)


def _channel_misalignment(stack: np.ndarray) -> float:
    """Residual shift (px) between channel 1 and the reference channel 0."""
    shift, _, _ = phase_cross_correlation(
        stack[0].astype(float), stack[1].astype(float), upsample_factor=10
    )
    return float(np.linalg.norm(shift))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")  # type: ignore
def struct_2ch() -> tuple[np.ndarray, dict]:
    """2-channel continuous structure: translation + small rotation."""
    return generate_structures_image(
        n_channels=2,
        shape=(256, 256),
        structure_scale=6.0,
        shifts=[(0, 0), (3.5, -2.0)],
        rotations=[0, 1.0],
        bit_depth=16,
        snr=15,
        seed=3,
    )


@pytest.fixture(scope="module")  # type: ignore
def csc_2ch(struct_2ch: tuple[np.ndarray, dict]) -> ChromaticShiftCorrector:
    """Corrector calibrated with the correlation method on the structure stack."""
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    sc.measure(img, method="correlation", block_size=64, block_overlap=0.5)
    return sc


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------


def test_measure_returns_correction_result(csc_2ch: ChromaticShiftCorrector) -> None:
    assert csc_2ch._result is not None
    assert isinstance(csc_2ch._result, CorrectionResult)
    assert 1 in csc_2ch._result.transforms


def test_measure_method_stored(csc_2ch: ChromaticShiftCorrector) -> None:
    assert csc_2ch.method == "correlation"


def test_measure_rms_below_one_pixel(csc_2ch: ChromaticShiftCorrector) -> None:
    rms = csc_2ch._result.transforms[1].rms_residual  # type: ignore[union-attr]
    assert rms is not None
    assert rms < 1.0


def test_measure_sufficient_blocks(csc_2ch: ChromaticShiftCorrector) -> None:
    assert csc_2ch._result.transforms[1].n_pairs >= 10  # type: ignore[union-attr]


def test_measure_builds_viz_images(csc_2ch: ChromaticShiftCorrector) -> None:
    res = csc_2ch._result
    assert res is not None
    assert res.detection_image is not None
    assert res.detection_image.shape == (4, 256, 256)
    assert res.pairs_image is not None and res.pairs_image.max() > 0


def test_measure_block_size_tuple(struct_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    res = sc.measure(img, method="correlation", block_size=(64, 96), block_overlap=0.5)
    assert res.transforms[1].n_pairs >= 4


# ---------------------------------------------------------------------------
# validate() / apply()
# ---------------------------------------------------------------------------


def test_validate_mean_error_below_half_pixel(csc_2ch: ChromaticShiftCorrector) -> None:
    stats = csc_2ch.validate()
    assert stats[1]["mean_error"] < 0.5
    assert stats[1]["n_pairs"] >= 10


def test_apply_aligns_channels(
    csc_2ch: ChromaticShiftCorrector, struct_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct_2ch
    before = _channel_misalignment(img)
    after = _channel_misalignment(csc_2ch.apply(img, crop=True))
    assert before > 2.0
    assert after < 1.0
    assert after < before / 3


def test_apply_shape_and_dtype_preserved(
    csc_2ch: ChromaticShiftCorrector, struct_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct_2ch
    out = csc_2ch.apply(img, crop=False)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


def test_apply_crop_reduces_size(
    csc_2ch: ChromaticShiftCorrector, struct_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct_2ch
    out = csc_2ch.apply(img, crop=True)
    assert out.shape[0] == img.shape[0]
    assert out.shape[1] <= img.shape[1]


# ---------------------------------------------------------------------------
# Correlation also works on a bead sample (it is sample-agnostic)
# ---------------------------------------------------------------------------


def test_correlation_on_beads_recovers_shift() -> None:
    img, _ = generate_beads_image(
        n_channels=2,
        shape=(256, 256),
        n_beads=40,
        bead_intensity=60,
        bit_depth=16,
        offset=100,
        shifts=[(0, 0), (4.0, -3.0)],
        snr=20,
        seed=7,
    )
    sc = ChromaticShiftCorrector()
    sc.measure(img, method="correlation", block_size=96, block_overlap=0.5)
    assert _channel_misalignment(sc.apply(img, crop=True)) < 1.0


# ---------------------------------------------------------------------------
# save() / from_json()
# ---------------------------------------------------------------------------


def test_save_load_roundtrip_params_and_apply(
    csc_2ch: ChromaticShiftCorrector, struct_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct_2ch
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        csc_2ch.save(path)
        sc2 = ChromaticShiftCorrector.from_json(path)

    assert sc2.method == "correlation"
    assert sc2.block_size == csc_2ch.block_size
    assert sc2.block_overlap == csc_2ch.block_overlap
    assert sc2.correlation_upsample == csc_2ch.correlation_upsample
    assert sc2.min_correlation == csc_2ch.min_correlation
    np.testing.assert_array_equal(
        csc_2ch.apply(img, crop=False), sc2.apply(img, crop=False)
    )


def test_save_load_roundtrip_tuple_block_size(
    struct_2ch: tuple[np.ndarray, dict],
) -> None:
    """A tuple block_size must survive the JSON round-trip as a tuple."""
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    sc.measure(img, method="correlation", block_size=(64, 96), block_overlap=0.5)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc.save(path)
        sc2 = ChromaticShiftCorrector.from_json(path)
    assert sc2.block_size == (64, 96)


# ---------------------------------------------------------------------------
# Parameter validation & fallbacks
# ---------------------------------------------------------------------------


def test_measure_unknown_method_raises(struct_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    with pytest.raises(ValueError, match="method must be"):
        sc.measure(img, method="bogus")


@pytest.mark.parametrize("overlap", [-0.1, 1.0, 1.5])  # type: ignore[misc]
def test_measure_bad_block_overlap_raises(
    struct_2ch: tuple[np.ndarray, dict], overlap: float
) -> None:
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    with pytest.raises(ValueError, match="block_overlap"):
        sc.measure(img, method="correlation", block_overlap=overlap)


def test_measure_bad_upsample_raises(struct_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    with pytest.raises(ValueError, match="correlation_upsample"):
        sc.measure(img, method="correlation", correlation_upsample=0)


def test_high_min_correlation_falls_back_to_translation(
    struct_2ch: tuple[np.ndarray, dict],
) -> None:
    """An unreachable correlation threshold rejects every block → fallback."""
    img, _ = struct_2ch
    sc = ChromaticShiftCorrector()
    with pytest.warns(UserWarning, match="Falling back to translation"):
        result = sc.measure(
            img, method="correlation", block_size=64, min_correlation=0.999
        )
    assert result.transforms[1].rms_residual is None
    assert result.transforms[1].n_pairs == 0


# ---------------------------------------------------------------------------
# generate_structures_image — validation errors and defaults
# ---------------------------------------------------------------------------

_BASE_KWARGS: dict = {
    "n_channels": 2,
    "shape": (64, 64),
    "bit_depth": 16,
    "offset": 10,
    "snr": 10,
    "seed": 0,
}


@pytest.mark.parametrize(  # type: ignore[misc]
    "override,exc_type,match",
    [
        ({"structure_intensity": 0.0}, ValueError, "structure_intensity"),
        ({"structure_intensity": 101.0}, ValueError, "structure_intensity"),
        ({"bit_depth": 32}, ValueError, "bit_depth"),
        ({"contour_width": 0.0}, ValueError, "contour_width"),
        ({"offset": 1.5}, TypeError, "offset"),
        ({"offset": -1}, ValueError, "offset"),
    ],
)
def test_generate_structures_invalid_params(
    override: dict, exc_type: type, match: str
) -> None:
    kwargs = {**_BASE_KWARGS, **override}
    with pytest.raises(exc_type, match=match):
        generate_structures_image(**kwargs)


def test_generate_structures_none_defaults() -> None:
    img, gt = generate_structures_image(**_BASE_KWARGS)
    assert img.shape == (2, 64, 64)
    assert img.dtype == np.uint16
    assert "structure_scale" in gt
