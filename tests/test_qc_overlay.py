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
