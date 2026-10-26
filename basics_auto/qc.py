"""QC renders: identity review contact sheets, matte overlays, review videos."""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180),
           (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (170, 110, 40)]


def actor_colour(actor: str) -> tuple[int, int, int]:
    try:
        i = int(actor.rsplit("_", 1)[1]) - 1
    except (ValueError, IndexError):
        # The builtin hash() is salted separately in each Python process.
        # Persisted QC sheets must use the same colour for the same actor ID
        # across different sessions and PYTHONHASHSEED values.
        i = int.from_bytes(hashlib.sha256(actor.encode("utf-8")).digest()[:8], "big")
    r, g, b = PALETTE[i % len(PALETTE)]
    return (b, g, r)


def _plate_lr(plate_dir: Path | None, fr: int, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    p = Path(plate_dir) / f"{fr:06d}.png" if plate_dir else None
    if p is not None and p.is_file():
        return cv2.resize(cv2.imread(str(p), cv2.IMREAD_COLOR), (w, h), interpolation=cv2.INTER_AREA)
    return np.full((h, w, 3), 40, np.uint8)


def overlay_frame(plate: np.ndarray, masks: list[tuple[str, np.ndarray, str]], title: str) -> np.ndarray:
    """masks: [(actor, bool mask, raw_id)] at plate resolution."""
    out = (plate.astype(np.float32) * 0.55).astype(np.uint8)
    for actor, m, raw in masks:
        col = np.array(actor_colour(actor), np.float32)
        out[m] = (out[m] * 0.35 + col * 0.65).astype(np.uint8)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, actor_colour(actor), 1)
        ys, xs = np.nonzero(m)
        if len(xs):
            x, y = int(xs.mean()), int(ys.mean())
            label = f"{actor.replace('actor_', 'A')}<{raw}"
            cv2.putText(out, label, (max(0, x - 30), y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 3)
            cv2.putText(out, label, (max(0, x - 30), y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(out, title, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def identity_sheets(out_dir: Path, per_frame, labels: dict, review: list[dict], plate_dir, frames: list[int],
                    lowres: int, *, context: int = 2, max_sheets: int = 60) -> list[str]:
    """One contact sheet per review cluster: frames f-context..f+context, actors coloured+labelled."""
    rdir = out_dir / "review"
    rdir.mkdir(exist_ok=True)
    clusters: list[list[dict]] = []
    for item in sorted(review, key=lambda r: r["frame"]):
        if clusters and item["frame"] - clusters[-1][-1]["frame"] <= context:
            clusters[-1].append(item)
        else:
            clusters.append([item])
    written = []
    fset = set(frames)
    for ci, cl in enumerate(clusters[:max_sheets]):
        f0 = cl[0]["frame"] - context
        f1 = cl[-1]["frame"] + context
        tiles = []
        for fr in range(f0, f1 + 1):
            if fr not in fset:
                continue
            dets = per_frame.get(fr, [])
            shape = dets[0].mask.shape if dets else next(iter(per_frame.values()))[0].mask.shape
            plate = _plate_lr(plate_dir, fr, shape)
            masks = [(labels.get((fr, j), "?"), d.mask, d.raw.replace("obj_", "o")) for j, d in enumerate(dets)
                     if labels.get((fr, j)) not in (None, "__dropped__")]
            flagged = any(it["frame"] == fr for it in cl)
            tile = overlay_frame(plate, masks, f"f{fr}" + (" *" if flagged else ""))
            if flagged:
                cv2.rectangle(tile, (0, 0), (tile.shape[1] - 1, tile.shape[0] - 1), (0, 0, 255), 3)
            tiles.append(tile)
        if not tiles:
            continue
        rows = [np.hstack(tiles[i:i + 5] + [np.zeros_like(tiles[0])] * (5 - len(tiles[i:i + 5])))
                for i in range(0, len(tiles), 5)]
        sheet = np.vstack(rows)
        reasons = "; ".join(sorted({f"{it['kind']}@{it['frame']}" for it in cl}))[:180]
        band = np.zeros((26, sheet.shape[1], 3), np.uint8)
        cv2.putText(band, reasons, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        path = rdir / f"cluster_{ci:03d}_f{cl[0]['frame']:06d}.png"
        cv2.imwrite(str(path), np.vstack([band, sheet]))
        written.append(str(path))
    return written


def actor_overlay_video(plate_dir: Path, actor_dirs: dict[str, Path], out_mp4: Path, fps: str,
                        frames: list[int], width: int = 1280) -> Path:
    """Create one 8-bit review MP4 without replacing existing files.

    Generated PNGs and the ffmpeg output live in an isolated temporary
    directory. A hard link publishes the completed MP4 with create-only
    semantics, avoiding the old overwrite-and-delete behaviour.
    """
    out_mp4 = Path(out_mp4)
    if out_mp4.exists() or out_mp4.is_symlink():
        raise FileExistsError("review video already exists: " + str(out_mp4))
    if not out_mp4.parent.is_dir() or out_mp4.parent.is_symlink():
        raise ValueError("review video parent must be an existing directory")
    if not frames or width < 1:
        raise ValueError("review video needs frames and a positive width")
    with tempfile.TemporaryDirectory(prefix=".basics-overlay-", dir=out_mp4.parent) as scratch:
        tmp = Path(scratch)
        for i, fr in enumerate(frames):
            plate_path = Path(plate_dir) / f"{fr:06d}.png"
            plate = cv2.imread(str(plate_path))
            if plate is None:
                raise ValueError("missing or unreadable plate: " + str(plate_path))
            h = round(plate.shape[0] * width / plate.shape[1])
            plate = cv2.resize(plate, (width, max(1, h)), interpolation=cv2.INTER_AREA)
            h = plate.shape[0]
            masks = []
            for a, d in actor_dirs.items():
                m = cv2.imread(str(Path(d) / f"{fr:06d}.png"), cv2.IMREAD_UNCHANGED)
                if m is None:
                    continue
                m = cv2.resize(m if m.ndim == 2 else m[..., 0], (width, h), interpolation=cv2.INTER_AREA)
                thr = 32768 if m.dtype == np.uint16 else 127
                if (m > thr).any():
                    masks.append((a, m > thr, ""))
            dest = tmp / f"{i:06d}.png"
            if not cv2.imwrite(str(dest), overlay_frame(plate, masks, f"f{fr}")):
                raise IOError("cannot write temporary review frame: " + str(dest))
        encoded = tmp / "overlay.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-n", "-framerate", fps, "-i", str(tmp / "%06d.png"),
                        "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(encoded)], check=True)
        if encoded.is_symlink() or not encoded.is_file():
            raise IOError("ffmpeg did not create a regular review MP4")
        # Both paths are on the same filesystem, because scratch was created
        # inside out_mp4.parent. link() never overwrites an existing target.
        os.link(encoded, out_mp4)
    return out_mp4
