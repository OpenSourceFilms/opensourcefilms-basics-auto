"""Synthetic 1-, 8- and 16-bit mask loader regression tests.

No production footage or model calls. Only temporary PNG fixtures.
"""
import numpy as np
import pytest
from PIL import Image

from basics_auto import maskgate


def _write(path, pixels, depth):
    path.parent.mkdir(parents=True, exist_ok=True)
    if depth == 1:
        Image.fromarray(pixels.astype(bool)).convert("1").save(path)
    elif depth == 8:
        Image.fromarray(pixels.astype(np.uint8), "L").save(path)
    elif depth == 16:
        Image.fromarray(pixels.astype(np.uint16)).save(path)
    else:
        raise ValueError(depth)


@pytest.mark.parametrize("lowres", [1, 2, 4])
def test_equivalent_binary_masks_use_same_ownership_at_every_depth(tmp_path, lowres):
    # 4-pixel alignment gives equivalent area downsampling for all depths.
    actor = np.zeros((32, 48), dtype=np.uint8)
    actor[4:28, 8:40] = 1
    expected = actor.astype(bool)
    outputs = []
    for depth, level in ((1, 1), (8, 255), (16, 65535)):
        path = tmp_path / f"mask_{depth}.png"
        _write(path, actor.astype(np.uint16) * level, depth)
        mask = maskgate._load(path, lowres)
        assert mask.dtype == np.dtype("bool")
        assert mask.shape == (32 // lowres, 48 // lowres)
        outputs.append(mask)
    assert all(np.array_equal(outputs[0], mask) for mask in outputs[1:])
    assert np.array_equal(outputs[0], expected[::lowres, ::lowres])


def test_low_16bit_alpha_values_are_not_mistaken_for_occupied_pixels(tmp_path):
    values = np.zeros((8, 8), dtype=np.uint16)
    values[0:4, 0:4] = 1000        # 1.5% opacity: must be background
    values[4:8, 0:4] = 32767       # below 50%: background
    values[0:4, 4:8] = 32768       # above 50%: foreground
    values[4:8, 4:8] = 65535       # opaque foreground
    png = tmp_path / "uint16.png"
    _write(png, values, 16)
    output = maskgate._load(png, lowres=1)
    assert not output[:, :4].any()
    assert output[:, 4:].all()
    # This is the original defect: even 1000 previously passed '>127'.
    assert not output[1, 1]


def test_fractional_16bit_mask_matches_equivalent_8bit_after_downsampling(tmp_path):
    block = np.zeros((8, 8), dtype=np.uint8)
    block[:, :4] = 64              # 25% should be unoccupied
    block[:, 4:] = 192             # 75% should be occupied
    u8, u16 = tmp_path / "a.png", tmp_path / "b.png"
    _write(u8, block, 8)
    _write(u16, block.astype(np.uint16) * 257, 16)
    for scale in (1, 2, 4):
        expected = maskgate._load(u8, scale)
        observed = maskgate._load(u16, scale)
        assert np.array_equal(observed, expected)
        assert not observed[:, 0].any()
        assert observed[:, -1].all()


def test_full_maskgate_verdict_unchanged_across_uint8_and_uint16(tmp_path):
    reports = []
    for depth, level in ((8, 255), (16, 65535)):
        root = tmp_path / f"depth_{depth}"
        for frame in range(12):
            a = np.zeros((48, 80), dtype=np.uint16)
            b = np.zeros_like(a)
            a[8:40, 8:40] = level
            b[8:40, 48:72] = level
            if frame == 6:
                # Synthetic SAM ownership mistake: b suddenly takes a's half.
                b[8:40, 24:40] = level
                a[8:40, 24:40] = 0
            for actor, pixels in (("actor_001", a), ("actor_002", b)):
                _write(root / actor / f"{frame:06d}.png", pixels, depth)
        report = maskgate.run(
            root, p=maskgate.GateParams(lowres=2, win=2, contact_px=2,
                                        core_erode_px=2, area_tol=0.2)
        )
        reports.append(report)
    for field in ("review_required", "flagged_frames", "crossings", "flags"):
        assert reports[0][field] == reports[1][field], field
    assert reports[0]["review_required"]
    assert 6 in reports[0]["flagged_frames"]


@pytest.mark.parametrize("bad_lowres", [0, -1, True, 1.5, "4", 100])
def test_invalid_downsample_factors_fail_explicitly(tmp_path, bad_lowres):
    path = tmp_path / "mask.png"
    _write(path, np.full((8, 8), 255, dtype=np.uint8), 8)
    with pytest.raises(ValueError, match="lowres"):
        maskgate._load(path, bad_lowres)
