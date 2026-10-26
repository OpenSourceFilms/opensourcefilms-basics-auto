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
