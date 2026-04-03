# Roadmap

## Current Priorities

### Audio False Positive Reduction
Audio detection has ~93% precision (10 FP per 100 kills in tapping). Current best approach: confidence thresholding (`conf >= 1.08`). Next steps:
- Validate confidence threshold across multiple sessions (currently tuned on warmup_20260322_1 only)
- Explore better visual confirmation signals (larger ROIs, template matching, or reintroducing OCR for bot count reading)
- Consider multi-signal fusion scoring (weighted audio confidence + visual support)

### Overlay Video Rendering Pipeline
`render_audio_kills.py` renders detected kills as a scrolling timeline strip with edge glow. Current state is functional. Next steps:
- Add confidence-based coloring to differentiate high/low confidence kills
- Add optional demo ground truth overlay (side-by-side comparison when CSV available)
- Parameterize rendering style (strip height, colors, glow timing)

### True TTFF via Video Crosshair Tracking
`acquisition_tracker.py` is a working prototype that detects crosshair acquisition from screen-center pixel motion. Next steps:
- Validate against more sessions (currently calibrated on training1 only)
- Integrate acquisition timing into the main engagement CSV pipeline
- Compare true TTFF distributions between tapping and spraying sessions

### Ground Truth CSV Commitment
Generate engagement CSVs from available `.dem` files and commit them to `tests/ground_truth/` so validation can run without the original demo files.

## Tech Debt

### Duplicated Scoring Logic
Four detectors implement their own `score()` / `validate()` / `optimal_match()` functions:
- `killfeed_detector.py` -- greedy cost-matrix matching
- `pixel_diff_detector.py` -- greedy nearest-match
- `kill_flash_detector.py` -- greedy nearest-match
- `audio_kill_detector.py` -- Hungarian algorithm (`linear_sum_assignment`)
- `render_audio_accuracy.py` -- duplicated Hungarian from audio_kill_detector (explicitly noted in source)

Should be consolidated into a shared `src/scoring.py` module. The `tests/test_scoring.py` skeleton is designed to make this refactor safe.

### Hardcoded Windows Paths
Several detectors have hardcoded `F:\P_rojects\...` paths as defaults:
- `killfeed_detector.py`
- `pixel_diff_detector.py`
- `render_kill_overlay.py`

These should be converted to CLI arguments (some already accept them) or config-file defaults.

### Missing Dependencies in requirements.txt
`pytesseract` is used by `calibrate_rois.py` and `kill_flash_detector.py` but listed as "optional" -- should be documented more clearly or added to an optional extras group. The primary audio pipeline does not require it.

### ROI Calibration Per-Video
ROIs in `pixel_diff_detector.py` were calibrated on training1 at 1280x960. The `kill_counter` ROI (0.2773, 0.9427) actually points at the HP display, not a kill counter. ROIs need per-session validation (use `calibrate_rois.py` or manual spot-checks).

## Future Ideas

- Dashboard / web UI for comparing sessions
- Automated warmup curve detection (TTFF trend over first N engagements)
- Multi-weapon support (currently AK-47 only; cycle times differ for other weapons)
- Spray pattern analysis using per-shot hit/miss sequences
