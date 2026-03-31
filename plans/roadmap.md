# Roadmap

## Current Priorities

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
`pytesseract` is used by `calibrate_rois.py` and `kill_flash_detector.py` but listed as "optional" -- should be documented more clearly or added to an optional extras group.

## Future Ideas

- Dashboard / web UI for comparing sessions
- Automated warmup curve detection (TTFF trend over first N engagements)
- Multi-weapon support (currently AK-47 only; cycle times differ for other weapons)
- Spray pattern analysis using per-shot hit/miss sequences
