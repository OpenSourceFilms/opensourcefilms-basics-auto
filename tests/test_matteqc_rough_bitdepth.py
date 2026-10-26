"""Synthetic post-matting rough-mask bit-depth regression.

All PNGs are disposable. The tests run the report pipeline, not just its
imported loader, and never contact VideoMaMa or production storage.
"""
from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from basics_auto import matteqc


def _save_mask(path, values, depth):
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth == 1:
        image = Image.fromarray((values >= 0.5).astype(np.uint8) * 255, "L").convert("1")
    elif depth == 8:
        image = Image.fromarray(np.rint(values * 255).astype(np.uint8), "L")
    elif depth == 16:
        image = Image.fromarray(np.rint(values * 65535).astype(np.uint16))
    else:
        raise ValueError(depth)
    image.save(path)


def _save_alpha(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(values * 65535).astype(np.uint16)).save(path)


@pytest.mark.parametrize("depth", [1, 8, 16])
def test_faint_rough_mask_is_not_an_opaque_body_core(tmp_path, depth):
    h, w = 96, 128
    rough = np.zeros((h, w), np.float32)
    rough[16:80, 16:64] = 1.0
    rough[16:80, 64:112] = 1000.0 / 65535.0
    matte = np.zeros((h, w), np.float32)
    matte[16:80, 16:64] = 1.0
    matte[16:80, 64:112] = 0.65

    root = tmp_path / str(depth)
    matte_path = root / "alpha" / "000007.png"
    rough_path = root / "rough" / "000007.png"
    _save_alpha(matte_path, matte)
    _save_mask(rough_path, rough, depth)
    before = (matte_path.read_bytes(), rough_path.read_bytes())
    report = matteqc.run(
        root / "alpha", root / "review.json", root / "rough",
        matteqc.MatteQCParams(core_erode_px=3)
    )
    frame = report["frames"][7]
    assert frame["partial_core"] == 0.0
    assert frame["core_mean"] > 0.99
    assert report["partial_core_frames"] == []
    assert report["review_required"] is False
    assert (matte_path.read_bytes(), rough_path.read_bytes()) == before


def test_uint16_half_intensity_boundary_selects_correct_core(tmp_path):
    # A pixel value of 32767 is below the 16-bit half-range threshold;
    # 32768 belongs to the opaque mask. Converting I;16 -> L first
    # incorrectly saturates both values to 255.
    h, w = 80, 128
    rough = np.zeros((h, w), np.uint16)
    rough[8:72, 8:64] = 32767
    rough[8:72, 64:120] = 32768
    matte = np.zeros((h, w), np.float32)
    matte[8:72, 8:64] = 0.65
    matte[8:72, 64:120] = 1.0
    rough_path = tmp_path / "rough" / "000001.png"
    rough_path.parent.mkdir(parents=True)
    Image.fromarray(rough).save(rough_path)
    _save_alpha(tmp_path / "alpha" / "000001.png", matte)

    report = matteqc.run(
        tmp_path / "alpha", rough_dir=tmp_path / "rough",
        p=matteqc.MatteQCParams(core_erode_px=3),
    )
    frame = report["frames"][1]
    assert frame["partial_core"] == 0.0
    assert frame["core_mean"] > 0.99
    assert report["partial_core_frames"] == []


def test_16bit_rough_resizing_matches_nearest_neighbor_8bit(tmp_path):
    # The rough SAM mask can differ in shape from the compositor alpha.
    # Binary thresholding must happen before resizing with INTER_NEAREST.
    half_h, half_w = 48, 64
    rough = np.zeros((half_h, half_w), np.float32)
    rough[8:40, 8:32] = 1.0
    rough[8:40, 32:56] = 0.015
    matte = np.zeros((96, 128), np.float32)
    matte[16:80, 16:64] = 1.0
    matte[16:80, 64:112] = 0.65

    outputs = []
    for depth in (8, 16):
        root = tmp_path / str(depth)
        _save_mask(root / "rough" / "000003.png", rough, depth)
        _save_alpha(root / "alpha" / "000003.png", matte)
        report = matteqc.run(
            root / "alpha", rough_dir=root / "rough",
            p=matteqc.MatteQCParams(core_erode_px=3),
        )
        outputs.append(report)
    for field in ("partial_core", "core_mean"):
        assert outputs[0]["frames"][3][field] == outputs[1]["frames"][3][field]
    assert outputs[1]["frames"][3]["partial_core"] == 0.0
    assert outputs[1]["frames"][3]["core_mean"] > 0.99
    assert outputs[1]["review_required"] is False
