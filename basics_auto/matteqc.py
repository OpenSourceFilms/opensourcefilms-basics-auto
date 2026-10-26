"""Post-VideoMaMa matte QC for compositor utility mattes. Review-only: reads mattes, never modifies them.

Per frame (alpha canvas, optionally with the rough SAM mask the matte was made from):
- stray: alpha islands (alpha > `on`) smaller than `stray_max_px` lying more than `far_px` from the largest island
  (e.g. a speck on a wall lamp). Reported with bbox; the compositor decides.
- partial_core: share of the rough mask's eroded core with partial alpha (`lo` <= a < `hi`). Visible opaque body
  should be near 1; a high share is the occlusion "fade" (a synthetic actor partially fading).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .util import atomic_json
from .maskgate import _load as _load_binary_mask


@dataclass
class MatteQCParams:
    on: float = 0.1
    stray_max_px: int = 4000
    far_px: int = 60
    core_erode_px: int = 10
    lo: float = 0.35
    hi: float = 0.9
    partial_flag: float = 0.10      # flag a frame when > 10 % of the core is partial


def _alpha(p: Path) -> np.ndarray:
    im = Image.open(p)
    return np.asarray(im, np.float32) / (65535 if im.mode.startswith("I") else 255)


def check_frame(a: np.ndarray, rough: np.ndarray | None, p: MatteQCParams) -> dict:
    out: dict = {"stray": []}
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > p.on).astype(np.uint8), connectivity=8)
    if n > 2:
        big = st[1:, cv2.CC_STAT_AREA].argmax() + 1
        dist = cv2.distanceTransform((lab != big).astype(np.uint8), cv2.DIST_L2, 5)
        for i in range(1, n):
            area = int(st[i, cv2.CC_STAT_AREA])
            if i == big or area > p.stray_max_px:
                continue
            d = float(dist[lab == i].min())
            if d > p.far_px:
                x, y, w, h = (int(v) for v in st[i, :4])
                out["stray"].append({"px": area, "bbox": [x, y, x + w, y + h], "dist_px": round(d)})
    if rough is not None:
        k = np.ones((2 * p.core_erode_px + 1,) * 2, np.uint8)
        core = cv2.erode(rough.astype(np.uint8), k) > 0
        if core.any():
            part = (a[core] >= p.lo) & (a[core] < p.hi)
            out["partial_core"] = round(float(part.mean()), 4)
            out["core_mean"] = round(float(a[core].mean()), 4)
    return out


def run(alpha_dir: Path, out_json: Path | None = None, rough_dir: Path | None = None,
        p: MatteQCParams | None = None) -> dict:
    p = p or MatteQCParams()
    frames = {}
    for f in sorted(Path(alpha_dir).glob("*.png")):
        if not re.fullmatch(r"\d+", f.stem):
            continue
        rough = None
        a = _alpha(f)
        if rough_dir is not None and (rp := Path(rough_dir) / f.name).exists():
            # Preserve native 1-, 8- or 16-bit mask depth before thresholding.
            # Pillow's I;16 -> L conversion saturates even low alpha to 255,
            # incorrectly expanding the eroded 'opaque subject' core.
            rough = _load_binary_mask(rp, lowres=1)
            if rough.shape != a.shape:
                # Nearest-neighbour interpolation preserves already-binary
                # ownership. OpenCV resize does not accept boolean arrays.
                rough = cv2.resize(
                    rough.astype(np.uint8),
                    (a.shape[1], a.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0
        frames[int(f.stem)] = check_frame(a, rough, p)
    stray = sorted(f for f, r in frames.items() if r["stray"])
    partial = sorted(f for f, r in frames.items() if r.get("partial_core", 0) > p.partial_flag)
    rep = {"schema": "basics_auto.matteqc/1", "alpha_dir": str(alpha_dir), "rough_dir": str(rough_dir) if rough_dir else None,
           "params": asdict(p), "stray_frames": stray, "partial_core_frames": partial,
           "review_required": bool(stray or partial), "frames": frames}
    if out_json:
        atomic_json(Path(out_json), rep)
    return rep
