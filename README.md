# BASICS Auto: mask QC and actor-overlay utilities

**Initial public core release, not the complete BASICS automation pipeline.**

This package provides independently useful CPU-side quality-control utilities developed by Open Source Films while applying third-party segmentation and matting research to filmmaking. It does not perform AI inference or implement a segmentation/matting model.

## Components

- **maskgate**: read-only temporal area-change, actor-contact and ownership-transfer warnings on binary actor mask PNG sequences.
  Handles 1-, 8- and 16-bit PNG masks with a consistent 50% occupancy
  threshold, including before low-resolution downsampling. A constant
  8-bit threshold previously misclassified faint 16-bit alpha as ownership.
- **matteqc**: read-only stray-alpha-island and partially transparent subject-core checks.
- **qc**: synthetic-compatible colour actor overlays and optional local ffmpeg review MP4 output.
  Numbered `actor_001` etc. labels retain the original palette, while
  other actor IDs now map through SHA-256 so their review colours remain
  consistent across Python processes, reboots and hash seeds.
  `actor_overlay_video()` now creates its video **only if the destination
  does not already exist**, using a private temporary directory next to the
  requested output. It does not reuse or delete a similarly named pre-existing
  frame directory and never passes FFmpeg an overwrite flag. If encoding
  fails, its own temporary files are removed and existing user media is
  untouched.

The tools generate human-review evidence. A clean flag report does not establish shot quality or reliable actor tracking.

## Install and test

Use Python 3.11 in a clean virtual environment. Run the complete synthetic suite:

```bash
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q tests/test_maskgate.py tests/test_matteqc.py tests/test_qc_overlay.py tests/test_maskgate_bitdepth.py
```

Expected result: **23 passed**, including create-only MP4 publishing, failed-encode cleanup and output-race safety. These use a mocked FFmpeg encoder; they are **not** a real video encoding quality or codec compatibility test. These tests generate 1-, 8- and 16-bit masks and frames in temporary directories; no model, GPU or production footage is required. The additional tests compare equivalent binary masks and synthetic actor-ownership flags across bit depths, as well as invalid downsample factors.

### Check your own PNG masks

```python
from pathlib import Path
from basics_auto import maskgate, matteqc

# actor_001/000000.png, actor_001/000001.png, actor_002/..., etc.
ownership = maskgate.run(
    Path("example_masks"),
    out_json=Path("reports/mask_review.json"),
)
print("Mask review required:", ownership["review_required"])

# Numbered 8- or 16-bit alpha PNGs; rough masks are optional.
alpha = matteqc.run(
    Path("example_alpha"),
    out_json=Path("reports/matte_review.json"),
    rough_dir=Path("example_rough"),
)
print("Matte review required:", alpha["review_required"])
```

The example paths are placeholders, not sample footage supplied with this repository. Mask ownership, alpha quality and coloured overlays are **review indicators** only, not automatic repairs or creative approval. The optional MP4 review function requires `ffmpeg` installed separately.

## Exclusions and caveats

The original private BASICS system includes model integrations, actor identity correction, media processing, orchestration and production-specific modules. Those are intentionally **not included in this first release**. No original footage, pretrained models, ComfyUI systems, service configuration or credentials are shipped. The public extraction has not been qualified as a full filmmaking pipeline.

See ATTRIBUTION.md. MIT covers only original OSF-authored code and documentation, not the models or libraries used elsewhere in the workflow.
