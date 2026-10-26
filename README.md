# BASICS Auto: mask QC and actor-overlay utilities

**Initial public core release, not the complete BASICS automation pipeline.**

This package provides independently useful CPU-side quality-control utilities developed by Open Source Films while applying third-party segmentation and matting research to filmmaking. It does not perform AI inference or implement a segmentation/matting model.

## Components

- **maskgate**: read-only temporal area-change, actor-contact and ownership-transfer warnings on binary actor mask PNG sequences.
- **matteqc**: read-only stray-alpha-island and partially transparent subject-core checks.
- **qc**: synthetic-compatible colour actor overlays and optional local ffmpeg review MP4 output.

The tools generate human-review evidence. A clean flag report does not establish shot quality or reliable actor tracking.

## Install and test

Use Python 3.11 in a clean virtual environment, then install requirements-dev.txt. Run pytest on tests/test_maskgate.py and tests/test_matteqc.py. All four tests use synthetic masks and temporary directories. The optional MP4 function requires ffmpeg installed separately.

## Exclusions and caveats

The original private BASICS system includes model integrations, actor identity correction, media processing, orchestration and production-specific modules. Those are intentionally **not included in this first release**. No original footage, pretrained models, ComfyUI systems, service configuration or credentials are shipped. The public extraction has not been qualified as a full filmmaking pipeline.

See ATTRIBUTION.md. MIT covers only original OSF-authored code and documentation, not the models or libraries used elsewhere in the workflow.
