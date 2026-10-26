# Upstream research and technology

This code is an **application and integration of existing video/mask-processing technology**, not an original AI segmentation model.

- OpenCV (https://opencv.org/): connected-component detection, mathematical morphology, image processing and overlay drawing.
- NumPy (https://numpy.org/): multidimensional numerical arrays.
- Pillow (https://python-pillow.org/): image loading and writing.
- FFmpeg (https://ffmpeg.org/): external encoder, optional and not included.
- Meta Segment Anything research and SAM 3 (https://github.com/facebookresearch/sam3): possible external providers of initial subject masks.
- VideoMaMa (https://github.com/cvlab-kaist/VideoMaMa): upstream matting research used in OSF experiments, with independent model and commercial-use terms.

OSF contributed these selected temporal mask/alpha diagnostic heuristics, review presentation code and synthetic regression tests to its filmmaking pipeline. No claim is made to the models, weights, segmentation/tracking inventions or independent library implementations. These upstream components are not vendored or relicensed by OSF.
