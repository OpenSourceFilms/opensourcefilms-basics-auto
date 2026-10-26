"""Matte QC: stray islands + partial alpha on the visible core; review-only."""
import numpy as np
from PIL import Image

from basics_auto import matteqc


def _save(p, a, bits16=True):
    p.parent.mkdir(parents=True, exist_ok=True)
    if bits16:
        Image.fromarray((a * 65535).astype(np.uint16)).save(p)
    else:
        Image.fromarray((a * 255).astype(np.uint8), "L").save(p)


def test_stray_and_partial(tmp_path):
    body = np.zeros((200, 300), np.float32)
    body[40:180, 60:140] = 1.0
    rough = body > 0
    clean = body.copy()
    speck = body.copy()
    speck[10:16, 250:256] = 0.8                      # small island far from the body
    near = body.copy()
    near[100:104, 142:146] = 0.8                     # small island hugging the body: not stray
    faded = body * 0.7                               # visible body at 0.7 alpha
    for f, a in enumerate([clean, speck, near, faded]):
        _save(tmp_path / "alpha" / f"{f:06d}.png", a)
        _save(tmp_path / "rough" / f"{f:06d}.png", rough.astype(np.float32), bits16=False)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*.png")}
    rep = matteqc.run(tmp_path / "alpha", tmp_path / "qc.json", tmp_path / "rough")
    assert rep["stray_frames"] == [1]
    assert rep["frames"][1]["stray"][0]["bbox"] == [250, 10, 256, 16]
    assert rep["partial_core_frames"] == [3]
    assert rep["frames"][0]["core_mean"] > 0.99 and rep["review_required"]
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*.png")}
    assert (tmp_path / "qc.json").is_file()
