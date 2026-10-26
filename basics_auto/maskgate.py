"""Pre-VideoMaMa mask quality gate: flag suspicious SAM pixel ownership BEFORE expensive matting.

Reads an idfix output (actor_*/NNNNNN.png binary masks). Never modifies masks and never repairs: it
only emits review flags and a verdict. Signals, per actor and frame (at 1/`lowres` scale):

- area_jump: area deviates from the temporal median of its neighbours (+-`win` frames) by more than
  `area_tol` (ratio). Legitimate occlusion shrinks are usually gradual; SAM ownership errors jump.
- contact: two actors' masks touch/overlap (dilated by `contact_px`). This marks crossings, where
  ownership can go wrong; on its own it is informational, not a failure.
- ownership_transfer: pixels another actor owned in the previous frame (eroded core) now belong to this
  actor, AND this actor's area jumped. Real occluders take pixels gradually along a leading edge; a
  jump plus a large transfer may indicate an ownership-transfer failure.

Synthetic jumps in tracked mask area can be used as reference positives.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .util import atomic_json


@dataclass
class GateParams:
    lowres: int = 4
    win: int = 3
    area_tol: float = 0.25          # |area / neighbour median - 1|
    contact_px: int = 6             # at full res
    transfer_frac: float = 0.08     # of the gaining actor's area
    core_erode_px: int = 8          # at full res


def _load(p: Path, lowres: int) -> np.ndarray:
    """Read binary mask PNGs with a threshold relative to their bit depth.

    The original constant 127 is a midpoint only for 8-bit pixels. For
    16-bit PNGs it would treat almost-transparent antialiasing as owned.
    """
    if type(lowres) is not int or lowres < 1:
        raise ValueError("lowres must be a positive integer")
    with Image.open(p) as image:
        decoder_mode, image_format = image.mode, image.format
        m = np.asarray(image)
    m = m[..., 0] if m.ndim == 3 else m
    if m.ndim != 2:
        raise ValueError("mask must have two spatial dimensions")
    if (m.dtype == np.dtype("int32") and decoder_mode == "I"
            and image_format == "PNG"):
        # Older supported Pillow releases decode unsigned 16-bit grayscale
        # PNG as mode I / int32. Preserve the actual 16-bit numeric range
        # before OpenCV resizing and half-occupancy thresholding. Do not
        # silently reinterpret arbitrary signed 32-bit images.
        if m.size == 0 or m.min() < 0 or m.max() > 65535:
            raise ValueError("decoded 16-bit mask PNG values outside 0..65535")
        m = m.astype(np.uint16)
    if m.dtype == np.dtype("bool"):
        # PIL's 1-bit PNG mode is boolean; OpenCV cannot resize bool arrays.
        m = m.astype(np.uint8) * 255
    if m.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("mask PNG must use 1-, 8- or 16-bit pixels")
    if lowres > 1:
        shape = (m.shape[1] // lowres, m.shape[0] // lowres)
        if min(shape) < 1:
            raise ValueError("lowres exceeds mask dimensions")
        m = cv2.resize(m, shape, interpolation=cv2.INTER_AREA)
    return m > (np.iinfo(m.dtype).max / 2)


def load_masks(idfix_dir: Path, lowres: int, frames: range | None = None) -> dict[str, dict[int, np.ndarray]]:
    out = {}
    for d in sorted(Path(idfix_dir).glob("actor_*")):
        if not d.is_dir():
            continue
        out[d.name] = {}
        for p in sorted(d.glob("*.png")):
            if re.fullmatch(r"\d+", p.stem) and (frames is None or int(p.stem) in frames):
                out[d.name][int(p.stem)] = _load(p, lowres)
    return out


def _k(r: int) -> np.ndarray:
    return np.ones((2 * r + 1, 2 * r + 1), np.uint8)


def evaluate(masks: dict[str, dict[int, np.ndarray]], p: GateParams | None = None) -> dict:
    p = p or GateParams()
    frames = sorted({f for m in masks.values() for f in m})
    area = {a: {f: int(m[f].sum()) for f in m} for a, m in masks.items()}
    flags = []
    rel = {}
    for a, ar in area.items():
        for f in sorted(ar):
            nb = [ar[g] for g in range(f - p.win, f + p.win + 1) if g != f and g in ar]
            med = float(np.median(nb)) if nb else 0.0
            r = ar[f] / med if med > 0 else 1.0
            rel[(a, f)] = r
            if med > 0 and abs(r - 1) > p.area_tol:
                flags.append({"frame": f, "kind": "area_jump", "actor": a, "ratio": round(r, 3)})
    cr = max(1, p.contact_px // p.lowres)
    er = max(1, p.core_erode_px // p.lowres)
    names = sorted(masks)
    for f in frames:
        for i, a in enumerate(names):
            ma = masks[a].get(f)
            if ma is None or not ma.any():
                continue
            da = cv2.dilate(ma.astype(np.uint8), _k(cr)) > 0
            for b in names[i + 1:]:
                mb = masks[b].get(f)
                if mb is not None and (da & mb).any():
                    flags.append({"frame": f, "kind": "contact", "actors": [a, b],
                                  "px": int((da & mb).sum()) * p.lowres ** 2})
            for b in names:
                prev_b, cur_b = masks[b].get(f - 1), masks[b].get(f)
                if b == a or prev_b is None:
                    continue
                core = cv2.erode(prev_b.astype(np.uint8), _k(er)) > 0
                gain = core & ma & ~(cur_b if cur_b is not None else np.zeros_like(ma))
                frac = gain.sum() / max(1, ma.sum())
                if frac > p.transfer_frac and rel.get((a, f), 1) > 1 + p.area_tol:
                    flags.append({"frame": f, "kind": "ownership_transfer", "actor": a, "from": b,
                                  "frac": round(float(frac), 3), "ratio": round(rel[(a, f)], 3)})
    serious = [x for x in flags if x["kind"] != "contact"]
    crossings = {}
    for x in flags:
        if x["kind"] == "contact":
            crossings.setdefault("+".join(x["actors"]), []).append(x["frame"])
    return {"schema": "basics_auto.maskgate/1", "params": asdict(p), "frames": [frames[0], frames[-1]] if frames else [],
            "review_required": bool(serious), "flagged_frames": sorted({x["frame"] for x in serious}),
            "crossings": {k: _runs(v) for k, v in crossings.items()}, "flags": flags}


def _runs(fs: list[int]) -> list[list[int]]:
    out = []
    for f in sorted(set(fs)):
        if out and f == out[-1][1] + 1:
            out[-1][1] = f
        else:
            out.append([f, f])
    return out


def run(idfix_dir: Path, out_json: Path | None = None, p: GateParams | None = None) -> dict:
    p = p or GateParams()
    rep = evaluate(load_masks(Path(idfix_dir), p.lowres), p)
    rep["idfix_dir"] = str(idfix_dir)
    if out_json:
        atomic_json(Path(out_json), rep)
    return rep
