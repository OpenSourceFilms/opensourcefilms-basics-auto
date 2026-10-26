"""Synthetic review-overlay smoke test. No production media."""
import numpy as np

from basics_auto import qc


def test_actor_overlay_with_synthetic_pixels():
    plate = np.full((96, 160, 3), 60, dtype=np.uint8)
    mask = np.zeros((96, 160), dtype=bool)
    mask[25:70, 30:90] = True
    original = plate.copy()
    overlay = qc.overlay_frame(plate, [("actor_001", mask, "synthetic")], "synthetic frame")
    assert overlay.shape == plate.shape
    assert overlay.dtype == np.uint8
    assert (overlay[45, 50] != original[45, 50]).any()
    assert np.array_equal(plate, original)  # original input is not mutated


def test_actor_colour_numeric_ids_preserve_palette():
    assert qc.actor_colour("actor_001") == (75, 25, 230)
    assert qc.actor_colour("actor_002") == (75, 180, 60)
    assert qc.actor_colour("actor_012") == (40, 110, 170)


def test_fallback_actor_colour_is_stable_across_hash_seeds(tmp_path):
    import json
    import os
    import subprocess
    import sys

    actor = "actor_subject_unknown"
    script = (
        "import json; from basics_auto.qc import actor_colour; "
        "print(json.dumps(actor_colour('actor_subject_unknown')))"
    )
    observed = []
    for seed in ("1", "2", "3", "random"):
        env = os.environ.copy()
        env["PYTHONHASHSEED"] = seed
        outcome = subprocess.run(
            [sys.executable, "-B", "-c", script],
            check=True, capture_output=True, text=True, timeout=15, env=env
        )
        observed.append(tuple(json.loads(outcome.stdout)))
    assert len(set(observed)) == 1
    assert observed[0] == qc.actor_colour(actor)


def test_unknown_actor_colour_is_repeatable_for_non_ascii_names():
    assert qc.actor_colour("actor_Žofie") == qc.actor_colour("actor_Žofie")
    assert len(qc.actor_colour("actor_Žofie")) == 3

def _make_plate(tmp_path):
    import cv2
    plate_dir = tmp_path / "plates"
    plate_dir.mkdir()
    plate = np.full((32, 48, 3), 60, np.uint8)
    assert cv2.imwrite(str(plate_dir / "000000.png"), plate)
    return plate_dir


def test_review_video_is_create_only_and_leaves_legacy_sibling_intact(tmp_path, monkeypatch):
    from pathlib import Path
    plate_dir = _make_plate(tmp_path)
    output = tmp_path / "review.mp4"
    legacy = output.with_suffix("")
    legacy.mkdir()
    (legacy / "keep.png").write_bytes(b"precious-existing-review-frame")
    original = (legacy / "keep.png").read_bytes()

    def fake_ffmpeg(argv, *, check):
        assert check is True
        assert "-n" in argv and "-y" not in argv
        assert Path(argv[-1]).parent != legacy
        Path(argv[-1]).write_bytes(b"fabricated-test-encoder-output")

    monkeypatch.setattr(qc.subprocess, "run", fake_ffmpeg)
    actual = qc.actor_overlay_video(plate_dir, {}, output, "24", [0], width=48)
    assert actual == output
    assert output.read_bytes() == b"fabricated-test-encoder-output"
    assert (legacy / "keep.png").read_bytes() == original
    assert not list(tmp_path.glob(".basics-overlay-*"))


def test_existing_review_video_cannot_be_overwritten(tmp_path, monkeypatch):
    plate_dir = _make_plate(tmp_path)
    output = tmp_path / "review.mp4"
    output.write_bytes(b"protected-existing-video")
    monkeypatch.setattr(qc.subprocess, "run", lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("ffmpeg must not run")))
    import pytest
    with pytest.raises(FileExistsError, match="already exists"):
        qc.actor_overlay_video(plate_dir, {}, output, "24", [0], width=48)
    assert output.read_bytes() == b"protected-existing-video"
    assert not list(tmp_path.glob(".basics-overlay-*"))


def test_failed_encode_cleans_only_its_own_temporary_files(tmp_path, monkeypatch):
    import subprocess
    import pytest
    plate_dir = _make_plate(tmp_path)
    output = tmp_path / "review.mp4"
    legacy = output.with_suffix("")
    legacy.mkdir()
    (legacy / "keep.png").write_bytes(b"keep-me")

    def failed_encode(argv, *, check):
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(qc.subprocess, "run", failed_encode)
    with pytest.raises(subprocess.CalledProcessError):
        qc.actor_overlay_video(plate_dir, {}, output, "24", [0], width=48)
    assert not output.exists()
    assert (legacy / "keep.png").read_bytes() == b"keep-me"
    assert not list(tmp_path.glob(".basics-overlay-*"))


def test_output_creation_race_is_non_destructive(tmp_path, monkeypatch):
    from pathlib import Path
    import pytest
    plate_dir = _make_plate(tmp_path)
    output = tmp_path / "review.mp4"

    def fake_race(argv, *, check):
        Path(argv[-1]).write_bytes(b"new-unpublished-review")
        output.write_bytes(b"another-process-created-this")

    monkeypatch.setattr(qc.subprocess, "run", fake_race)
    with pytest.raises(FileExistsError):
        qc.actor_overlay_video(plate_dir, {}, output, "24", [0], width=48)
    assert output.read_bytes() == b"another-process-created-this"
    assert not list(tmp_path.glob(".basics-overlay-*"))
