"""
pytest tests for ChromaticShiftCorrector3D.

Synthetic 3-D bead volumes from _sample_generator are used as ground-truth
calibration data so tests are self-contained and reproducible.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
import tifffile

from microcal import (
    ChannelTransform,
    ChromaticShiftCorrector,
    ChromaticShiftCorrector3D,
    CorrectionResult,
)
from microcal._sample_generator import generate_beads_image_3d

# Reusable detection params that reliably converge on the synthetic volumes.
_MEASURE_KW: dict = {
    "smooth_sigma": (1.5, 2.0, 2.0),
    "min_distance": 4,
    "threshold_rel": 0.3,
    "match_max_distance": 12,
    "min_pairs": 4,
    "refine_radius": (2, 3, 3),
}


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")  # type: ignore
def bead3d_2ch() -> tuple[np.ndarray, dict]:
    """2-channel, translation-only shift (with a clear axial component)."""
    return generate_beads_image_3d(
        n_channels=2,
        shape=(24, 96, 96),
        n_beads=40,
        bead_sigma=(1.5, 2.0, 2.0),
        bead_intensity=70,
        bit_depth=16,
        offset=100,
        shifts=[(0, 0, 0), (3.0, 2.0, -4.0)],
        snr=30,
        seed=1,
    )


@pytest.fixture(scope="module")  # type: ignore
def bead3d_3ch() -> tuple[np.ndarray, dict]:
    """3-channel with rotation (about z) and anisotropic scale."""
    return generate_beads_image_3d(
        n_channels=3,
        shape=(24, 96, 96),
        n_beads=40,
        bead_sigma=(1.5, 2.0, 2.0),
        bead_intensity=70,
        bit_depth=16,
        offset=100,
        shifts=[(0, 0, 0), (2.0, 3.0, -3.0), (-1.5, -2.0, 2.0)],
        rotations=[0.0, 2.0, -2.0],
        scales=[(1, 1, 1), (1.0, 1.02, 0.98), (1.0, 0.98, 1.02)],
        snr=30,
        seed=3,
    )


@pytest.fixture(scope="module")  # type: ignore
def sc3d_2ch(bead3d_2ch: tuple[np.ndarray, dict]) -> ChromaticShiftCorrector3D:
    """Corrector already calibrated on the 2-channel bead volume."""
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, **_MEASURE_KW)
    return sc


# ---------------------------------------------------------------------------
# measure()
# ---------------------------------------------------------------------------


def test_measure_returns_correction_result(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    result = sc.measure(img, **_MEASURE_KW)
    assert isinstance(result, CorrectionResult)
    assert result.reference_channel == 0
    assert 1 in result.transforms
    assert isinstance(result.transforms[1], ChannelTransform)
    # 3-D transforms are 4x4 homogeneous matrices.
    assert result.transforms[1].transform.params.shape == (4, 4)


def test_measure_rms_below_one_voxel(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    assert sc3d_2ch._result is not None
    rms = sc3d_2ch._result.transforms[1].rms_residual
    assert rms is not None
    assert rms < 1.0


def test_measure_sufficient_pairs(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    assert sc3d_2ch._result is not None
    assert sc3d_2ch._result.transforms[1].n_pairs >= 10


def test_measure_stores_bead_stack(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    assert sc3d_2ch._bead_stack is not None
    assert sc3d_2ch._bead_stack.shape == img.shape


def test_measure_recovers_translation_per_axis(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    """The fitted transform maps known moving bead centres back to the reference.

    Checked per axis so an axis-swap in the (z, y, x) → (x, y, z) conversion
    fails loudly (an isotropic norm would hide it).
    """
    _, gt = bead3d_2ch
    centers = gt["centers"]  # (N, 3) in (z, y, x), reference frame
    t_fwd = np.array([3.0, 2.0, -4.0])  # ch1 shift (dz, dy, dx)
    assert sc3d_2ch._result is not None
    tform = sc3d_2ch._result.transforms[1].transform
    # moving-frame bead positions, fed to the transform in (x, y, z) order
    mov_xyz = (centers + t_fwd)[:, ::-1]
    predicted_ref_xyz = tform(mov_xyz)
    ref_xyz = centers[:, ::-1]
    per_axis = np.abs(predicted_ref_xyz - ref_xyz).mean(axis=0)  # (x, y, z)
    assert np.all(per_axis < 0.5)


def test_measure_pure_z_shift(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    """A pure-z shift must land in z only — the key axis-conversion diagnostic."""
    img, _ = generate_beads_image_3d(
        n_channels=2,
        shape=(24, 96, 96),
        n_beads=40,
        bead_sigma=(1.5, 2.0, 2.0),
        bit_depth=16,
        offset=100,
        shifts=[(0, 0, 0), (3.0, 0.0, 0.0)],
        snr=30,
        seed=2,
    )
    sc = ChromaticShiftCorrector3D()
    res = sc.measure(img, **_MEASURE_KW)
    tx, ty, tz = res.transforms[1].transform.params[:3, 3]
    assert abs(tx) < 0.5  # no leak into x
    assert abs(ty) < 0.5  # no leak into y
    assert tz == pytest.approx(-3.0, abs=0.5)
    stats = sc.validate()
    assert stats[1]["mean_error"] < 0.5


def test_measure_wrong_ndim_raises(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    with pytest.raises(ValueError, match="4-D"):
        sc.measure(img[0])  # 3-D volume, not (C, Z, Y, X)


def test_measure_bad_voxel_size_raises(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    with pytest.raises(ValueError, match="voxel_size"):
        sc.measure(img, voxel_size=(0.2, 0.1), **_MEASURE_KW)


def test_measure_translation_fallback(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    kw = {**_MEASURE_KW, "min_pairs": 9999}
    with pytest.warns(UserWarning, match="Falling back to translation"):
        result = sc.measure(img, **kw)
    ct = result.transforms[1]
    assert ct.rms_residual is None
    assert ct.n_pairs == 0
    # the fallback must still produce a valid 4x4 transform
    assert ct.transform.params.shape == (4, 4)


def test_measure_blank_volume_no_beads() -> None:
    blank = np.zeros((2, 16, 48, 48), dtype=np.uint16)
    sc = ChromaticShiftCorrector3D()
    with pytest.warns(UserWarning, match="Falling back to translation"):
        result = sc.measure(blank, threshold_rel=0.01, min_pairs=1)
    assert result.transforms[1].rms_residual is None


# ---------------------------------------------------------------------------
# apply() — input forms
# ---------------------------------------------------------------------------


def test_apply_ndarray_shape_preserved(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    out = sc3d_2ch.apply(img, crop=False)
    assert out.shape == img.shape


def test_apply_dtype_preserved(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    out = sc3d_2ch.apply(img, crop=False)
    assert out.dtype == img.dtype


def test_apply_crop_reduces_all_spatial_dims(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    out = sc3d_2ch.apply(img, crop=True)
    assert out.shape[0] == img.shape[0]  # channels unchanged
    assert out.shape[1] <= img.shape[1]
    assert out.shape[2] <= img.shape[2]
    assert out.shape[3] <= img.shape[3]


def test_apply_list_of_ndarrays(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    volumes = [img[i] for i in range(img.shape[0])]
    out = sc3d_2ch.apply(volumes, crop=False)
    assert out.shape == img.shape


def test_apply_list_of_paths(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    with tempfile.TemporaryDirectory() as d:
        paths = []
        for i in range(img.shape[0]):
            p = os.path.join(d, f"ch{i}.tiff")
            tifffile.imwrite(p, img[i])
            paths.append(p)
        out = sc3d_2ch.apply(paths, crop=False)
    assert out.shape == img.shape


def test_apply_bad_list_item_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with pytest.raises(TypeError):
        sc3d_2ch.apply([42, 43])


def test_apply_wrong_ndim_item_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with pytest.raises(ValueError, match="3-D"):
        sc3d_2ch.apply([np.zeros((2, 3, 4, 5))])  # 4-D item instead of 3-D


def test_apply_3d_ndarray_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with pytest.raises(ValueError, match="4-D"):
        sc3d_2ch.apply(np.zeros((16, 48, 48)))  # single volume, not (C, Z, Y, X)


def test_apply_empty_list_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with pytest.raises(TypeError, match="non-empty"):
        sc3d_2ch.apply([])


def test_apply_extra_channel_passthrough(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    three = np.concatenate([img, img[:1]], axis=0)  # (3, Z, Y, X)
    out = sc3d_2ch.apply(three, crop=False)
    assert out.shape == three.shape
    np.testing.assert_array_equal(out[2], three[2])  # ch2 untouched


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------


def test_validate_mean_error_below_half_voxel(
    sc3d_2ch: ChromaticShiftCorrector3D,
) -> None:
    stats = sc3d_2ch.validate()
    assert stats[1]["mean_error"] < 0.5


def test_validate_per_axis_present(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    stats = sc3d_2ch.validate()
    per_axis = stats[1]["mean_error_per_axis"]
    assert per_axis.shape == (3,)
    assert np.all(per_axis < 0.5)


def test_validate_before_measure_raises() -> None:
    sc = ChromaticShiftCorrector3D()
    with pytest.raises(RuntimeError):
        sc.validate()


def test_validate_zero_pairs() -> None:
    blank = np.zeros((2, 16, 48, 48), dtype=np.uint16)
    sc = ChromaticShiftCorrector3D()
    with pytest.warns(UserWarning):
        sc.measure(blank, threshold_rel=0.01, min_pairs=1)
    stats = sc.validate()
    assert stats[1]["n_pairs"] == 0
    assert np.isnan(stats[1]["mean_error"])


def test_validate_physical_residual_with_voxel_size(
    bead3d_2ch: tuple[np.ndarray, dict],
) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, voxel_size=(0.5, 0.1, 0.1), **_MEASURE_KW)
    stats = sc.validate()
    assert "mean_error_physical" in stats[1]
    assert stats[1]["mean_error_physical"] >= 0.0


# ---------------------------------------------------------------------------
# save() / from_json()
# ---------------------------------------------------------------------------


def test_save_creates_file(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc3d_2ch.save(path)
        assert os.path.isfile(path)


def test_from_json_apply_matches_original(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc3d_2ch.save(path)
        sc2 = ChromaticShiftCorrector3D.from_json(path)
    np.testing.assert_array_equal(
        sc3d_2ch.apply(img, crop=False), sc2.apply(img, crop=False)
    )


def test_from_json_persists_voxel_size(bead3d_2ch: tuple[np.ndarray, dict]) -> None:
    img, _ = bead3d_2ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, voxel_size=(0.5, 0.1, 0.1), **_MEASURE_KW)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc.save(path)
        sc2 = ChromaticShiftCorrector3D.from_json(path)
    assert sc2.voxel_size == (0.5, 0.1, 0.1)


def test_from_json_measure_params_restored(
    sc3d_2ch: ChromaticShiftCorrector3D,
) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc3d_2ch.save(path)
        sc2 = ChromaticShiftCorrector3D.from_json(path)
    assert sc2.threshold_rel == sc3d_2ch.threshold_rel
    assert sc2.match_max_distance == sc3d_2ch.match_max_distance
    assert sc2.transform_type == sc3d_2ch.transform_type


def test_from_json_cross_dim_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    """A 3-D calibration loaded by the 2-D class must raise."""
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc3d_2ch.save(path)
        with pytest.raises(ValueError, match="-D"):
            ChromaticShiftCorrector.from_json(path)


def test_from_json_validate_raises(sc3d_2ch: ChromaticShiftCorrector3D) -> None:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "cal.json")
        sc3d_2ch.save(path)
        sc2 = ChromaticShiftCorrector3D.from_json(path)
    with pytest.raises(RuntimeError):
        sc2.validate()


def test_save_before_measure_raises() -> None:
    sc = ChromaticShiftCorrector3D()
    with pytest.raises(RuntimeError):
        sc.save("/tmp/should_not_exist_3d.json")


# ---------------------------------------------------------------------------
# Visualisation arrays
# ---------------------------------------------------------------------------


def test_detection_and_pairs_images(
    sc3d_2ch: ChromaticShiftCorrector3D, bead3d_2ch: tuple[np.ndarray, dict]
) -> None:
    img, _ = bead3d_2ch
    assert sc3d_2ch._result is not None
    det = sc3d_2ch._result.detection_image
    pairs = sc3d_2ch._result.pairs_image
    assert det is not None and det.shape == (2 * img.shape[0], *img.shape[1:])
    assert pairs is not None and pairs.shape == img.shape[1:]
    assert pairs.max() > 0  # at least one matched group labelled


# ---------------------------------------------------------------------------
# 3-channel / rotation / scale
# ---------------------------------------------------------------------------


def test_three_channel_measure_and_validate(
    bead3d_3ch: tuple[np.ndarray, dict],
) -> None:
    img, _ = bead3d_3ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, **{**_MEASURE_KW, "match_max_distance": 15})
    assert sc._result is not None
    assert 1 in sc._result.transforms
    assert 2 in sc._result.transforms
    stats = sc.validate()
    assert stats[1]["mean_error"] < 1.0
    assert stats[2]["mean_error"] < 1.0


@pytest.mark.parametrize("transform_type", ["affine", "similarity"])  # type: ignore
def test_rotation_scale_transform_types(
    bead3d_3ch: tuple[np.ndarray, dict], transform_type: str
) -> None:
    img, _ = bead3d_3ch
    sc = ChromaticShiftCorrector3D()
    sc.measure(
        img,
        transform_type=transform_type,
        **{**_MEASURE_KW, "match_max_distance": 15},
    )
    stats = sc.validate()
    assert stats[1]["mean_error"] < 1.0


def test_full_euler_rotation_recovers() -> None:
    """Generate with a 3-axis Euler rotation and confirm the affine fit works."""
    img, _ = generate_beads_image_3d(
        n_channels=2,
        shape=(32, 96, 96),
        n_beads=45,
        bead_sigma=(1.5, 2.0, 2.0),
        bit_depth=16,
        offset=100,
        shifts=[(0, 0, 0), (2.0, 2.0, -2.0)],
        rotations=[0.0, (2.0, 1.5, -1.5)],
        snr=30,
        seed=7,
    )
    sc = ChromaticShiftCorrector3D()
    sc.measure(img, **{**_MEASURE_KW, "match_max_distance": 15})
    stats = sc.validate()
    assert stats[1]["mean_error"] < 1.0


# ---------------------------------------------------------------------------
# _match_beads — internal edge cases (3-D)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(  # type: ignore[misc]
    "ref,mov",
    [
        (np.empty((0, 3)), np.array([[1.0, 2.0, 3.0]])),
        (np.array([[1.0, 2.0, 3.0]]), np.empty((0, 3))),
    ],
)
def test_match_beads_empty_input(
    sc3d_2ch: ChromaticShiftCorrector3D, ref: np.ndarray, mov: np.ndarray
) -> None:
    src, dst = sc3d_2ch._match_beads(ref, mov)
    assert src.shape == (0, 3)
    assert dst.shape == (0, 3)


def test_match_beads_all_rejected_by_distance(
    sc3d_2ch: ChromaticShiftCorrector3D,
) -> None:
    ref = np.array([[0.0, 0.0, 0.0]])
    mov = np.array([[500.0, 500.0, 500.0]])
    src, dst = sc3d_2ch._match_beads(ref, mov)
    assert src.shape == (0, 3)
    assert dst.shape == (0, 3)


# ---------------------------------------------------------------------------
# generate_beads_image_3d — validation and defaults
# ---------------------------------------------------------------------------

_BASE_GEN_KW: dict = {
    "n_channels": 2,
    "shape": (12, 48, 48),
    "n_beads": 5,
    "bead_sigma": (1.5, 2.0, 2.0),
    "bead_intensity": 60,
    "bit_depth": 16,
    "offset": 10,
    "snr": 10,
    "seed": 0,
}


@pytest.mark.parametrize(  # type: ignore[misc]
    "override,exc_type,match",
    [
        ({"bead_intensity": 0.0}, ValueError, "bead_intensity"),
        ({"bit_depth": 32}, ValueError, "bit_depth"),
        ({"offset": 1.5}, TypeError, "offset"),
        ({"offset": -1}, ValueError, "offset"),
        ({"bead_sigma": (1.0, 2.0)}, ValueError, "sigma"),
    ],
)
def test_generate_beads_3d_invalid_params(
    override: dict, exc_type: type, match: str
) -> None:
    kwargs = {**_BASE_GEN_KW, **override}
    with pytest.raises(exc_type, match=match):
        generate_beads_image_3d(**kwargs)


def test_generate_beads_3d_none_defaults() -> None:
    img, gt = generate_beads_image_3d(**_BASE_GEN_KW)
    assert img.shape == (2, 12, 48, 48)
    assert img.dtype == np.uint16
    assert gt["centers"].shape == (5, 3)
