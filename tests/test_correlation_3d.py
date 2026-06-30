"""
pytest tests for the marker-free correlation method of ChromaticShiftCorrector3D.

Synthetic continuous-structure volumes (membrane-like level sets, no isolated
beads) from `generate_structures_image_3d` exercise the volumetric
`measure(method="correlation")` path.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from skimage.registration import phase_cross_correlation

from microcal import (
    ChromaticShiftCorrector3D,
    CorrectionResult,
    generate_structures_image_3d,
)


def _channel_misalignment(stack: np.ndarray) -> float:
    """Residual shift (voxels) between channel 1 and the reference channel 0."""
    shift, _, _ = phase_cross_correlation(
        stack[0].astype(float), stack[1].astype(float), upsample_factor=10
    )
    return float(np.linalg.norm(shift))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")  # type: ignore
def struct3d_2ch() -> tuple[np.ndarray, dict]:
    """2-channel continuous-structure volume: 3-D shift + small rotation/scale."""
    return generate_structures_image_3d(
        n_channels=2,
        shape=(32, 128, 128),
        structure_scale=(2.5, 6.0, 6.0),
        shifts=[(0, 0, 0), (1.5, 3.0, -2.0)],
        rotations=[0, 1.5],
        scales=[(1, 1, 1), (1.0, 1.01, 0.99)],
        bit_depth=16,
        snr=20,
        seed=7,
    )


@pytest.fixture(scope="module")  # type: ignore
def csc3d_2ch(struct3d_2ch: tuple[np.ndarray, dict]) -> ChromaticShiftCorrector3D:
    """3-D corrector calibrated with the correlation method."""
    img, _ = struct3d_2ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, method="correlation", block_size=(16, 48, 48), block_overlap=0.5)
    return sc


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------


def test_measure_returns_result(csc3d_2ch: ChromaticShiftCorrector3D) -> None:
    assert csc3d_2ch._result is not None
    assert isinstance(csc3d_2ch._result, CorrectionResult)
    assert 1 in csc3d_2ch._result.transforms
    # 3-D transforms are 4x4 homogeneous matrices.
    assert csc3d_2ch._result.transforms[1].transform.params.shape == (4, 4)


def test_measure_rms_below_one_voxel(csc3d_2ch: ChromaticShiftCorrector3D) -> None:
    rms = csc3d_2ch._result.transforms[1].rms_residual  # type: ignore[union-attr]
    assert rms is not None
    assert rms < 1.0


def test_measure_sufficient_blocks(csc3d_2ch: ChromaticShiftCorrector3D) -> None:
    assert csc3d_2ch._result.transforms[1].n_pairs >= 10  # type: ignore[union-attr]


def test_default_block_size_is_anisotropic_tuple() -> None:
    sc = ChromaticShiftCorrector3D()
    # The documented default favours a thinner axial block.
    img, _ = generate_structures_image_3d(
        n_channels=2,
        shape=(24, 96, 96),
        shifts=[(0, 0, 0), (1.0, 2.0, -1.5)],
        bit_depth=16,
        snr=20,
        seed=1,
    )
    res = sc.measure(img, method="correlation", block_overlap=0.5)
    assert sc.block_size == (16, 64, 64)
    assert res.transforms[1].n_pairs >= 4


# ---------------------------------------------------------------------------
# validate() / apply()
# ---------------------------------------------------------------------------


def test_validate_mean_error_below_half_voxel(
    csc3d_2ch: ChromaticShiftCorrector3D,
) -> None:
    stats = csc3d_2ch.validate()
    assert stats[1]["mean_error"] < 0.5
    assert "mean_error_per_axis" in stats[1]
    assert stats[1]["mean_error_per_axis"].shape == (3,)


def test_apply_aligns_channels(
    csc3d_2ch: ChromaticShiftCorrector3D, struct3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct3d_2ch
    before = _channel_misalignment(img)
    after = _channel_misalignment(csc3d_2ch.apply(img, crop=True))
    assert before > 2.0
    assert after < 1.0
    assert after < before / 3


def test_apply_shape_and_dtype_preserved(
    csc3d_2ch: ChromaticShiftCorrector3D, struct3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct3d_2ch
    out = csc3d_2ch.apply(img, crop=False)
    assert out.shape == img.shape
    assert out.dtype == img.dtype


# ---------------------------------------------------------------------------
# voxel_size → physical-unit validation residuals
# ---------------------------------------------------------------------------


def test_validate_physical_residuals_with_voxel_size(
    struct3d_2ch: tuple[np.ndarray, dict],
) -> None:
    img, _ = struct3d_2ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(
        img,
        method="correlation",
        block_size=(16, 48, 48),
        voxel_size=(1.0, 0.1, 0.1),
    )
    stats = sc.validate()
    assert "mean_error_physical" in stats[1]
    assert np.isfinite(stats[1]["mean_error_physical"])


# ---------------------------------------------------------------------------
# save() / from_json()
# ---------------------------------------------------------------------------


def test_save_load_roundtrip(
    csc3d_2ch: ChromaticShiftCorrector3D, struct3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = struct3d_2ch
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        csc3d_2ch.save(path)
        sc2 = ChromaticShiftCorrector3D.from_json(path)
    assert sc2.method == "correlation"
    assert sc2.block_size == csc3d_2ch.block_size
    np.testing.assert_array_equal(
        csc3d_2ch.apply(img, crop=False), sc2.apply(img, crop=False)
    )


def test_load_2d_into_3d_raises(
    csc3d_2ch: ChromaticShiftCorrector3D,
) -> None:
    """A 3-D calibration file must not load into the 2-D corrector class."""
    from microcal import ChromaticShiftCorrector

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        csc3d_2ch.save(path)
        with pytest.raises(ValueError, match="3-D"):
            ChromaticShiftCorrector.from_json(path)


# ---------------------------------------------------------------------------
# generate_structures_image_3d
# ---------------------------------------------------------------------------


def test_generate_structures_3d_shape_dtype() -> None:
    img, gt = generate_structures_image_3d(
        n_channels=3, shape=(16, 64, 64), bit_depth=8, snr=10, seed=0
    )
    assert img.shape == (3, 16, 64, 64)
    assert img.dtype == np.uint8
    assert len(gt["structure_scale"]) == 3
