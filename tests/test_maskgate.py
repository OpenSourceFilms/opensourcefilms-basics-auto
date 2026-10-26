"""Pre-VideoMaMa mask gate: flags, never modifies."""
import numpy as np
from PIL import Image

from basics_auto import maskgate


def _disc(h, w, cx, cy, r):
    y, x = np.ogrid[:h, :w]
    return (x - cx) ** 2 + (y - cy) ** 2 <= r * r


def _write(root, actor, f, m):
    d = root / actor
    d.mkdir(parents=True, exist_ok=True)
    Image.fromarray(m.astype(np.uint8) * 255, "L").save(d / f"{f:06d}.png")


def _shot(tmp_path, swallow_at=None):
    h, w = 120, 200
    for f in range(16):
        a = _disc(h, w, 60, 60, 25)
        b = _disc(h, w, 20 + 3 * f, 60, 15)             # b walks right, into a
        if f == swallow_at:                            # SAM gives a's pixels to b
            b = b | (a & (np.arange(w)[None, :] < 70))
            a = a & ~b
        else:
            a = a & ~b                                 # b in front: plain occlusion
        _write(tmp_path, "actor_001", f, a)
        _write(tmp_path, "actor_002", f, b)
    return tmp_path


def test_clean_crossing_only_contacts(tmp_path):
    rep = maskgate.run(_shot(tmp_path), p=maskgate.GateParams(lowres=1, contact_px=2, core_erode_px=2))
    assert not rep["review_required"]
    assert rep["crossings"]["actor_001+actor_002"]


def test_swallow_flags_jump_and_transfer(tmp_path):
    before = {p: p.read_bytes() for p in _shot(tmp_path, swallow_at=10).rglob("*.png")}
    rep = maskgate.run(tmp_path, tmp_path / "gate.json", maskgate.GateParams(lowres=1, contact_px=2, core_erode_px=2))
    assert rep["review_required"] and 10 in rep["flagged_frames"]
    kinds = {(x["kind"], x.get("actor")) for x in rep["flags"] if x["frame"] == 10}
    assert ("area_jump", "actor_002") in kinds and ("ownership_transfer", "actor_002") in kinds
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.png")}   # gate never repairs
    assert (tmp_path / "gate.json").is_file()
