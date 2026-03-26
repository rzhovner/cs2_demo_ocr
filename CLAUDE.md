# CS2 Demo + Video Analyzer

## What This Is

Tools for analyzing CS2 AimBotz 90° 100-bot challenge performance. Two data sources:
- **Demo files** (`.dem`) → `cs2_engagement_parser.py` → per-engagement CSV (shots, hits, timing)
- **Gameplay video** (`.mp4`) → `kill_flash_detector.py` → kill timestamps via OCR + flash detection

The demo parser is the primary data source. Video analysis supplements it for metrics the demo format can't provide (crosshair acquisition timing).

## Data Pipeline

```
.dem → cs2_engagement_parser.py → *_engagements.csv → plot_hitrate.py → *_hitrate.png
.mp4 → calibrate_rois.py → ROI coordinates → kill_flash_detector.py → *_kill_times.csv
```

## Key Domain Knowledge

### AK-47 Timing (CS2)
- **Cycle time: 100ms (0.1s) — 600 RPM — 10 rounds/second**
- No two shots can be closer than 100ms — this is a hard physical constraint
- `SHOT_MIN_GAP_S` must be `0.100`, never lower (0.05 was a past bug)
- Kill registers same tick as the lethal shot

### TTFF: Two Definitions
| Metric | Measures | Source |
|---|---|---|
| Parser TTFF (`ttff_ms`) | prev kill → first shot (rotation + acquisition + reaction) | Demo |
| True TTFF (planned) | crosshair on bot → first shot (pure reaction time) | Video |

The parser TTFF conflates rotation speed with reaction time. True TTFF requires video-based crosshair tracking.

### Workshop Demo Limitation
Workshop/offline server CS2 demos do **NOT** record per-tick player state (positions, angles). `parse_ticks()` returns empty DataFrames. Only game events (`weapon_fire`, `player_death`, `player_hurt`) are captured. This is why video analysis is needed for acquisition timing.

## File Layout

```
src/
  cs2_engagement_parser.py   # .dem → per-engagement CSV (main parser)
  plot_hitrate.py             # Rolling avg hit rate chart from CSV
  acquisition_tracker.py      # Video-based crosshair acquisition (WIP)
  calibrate_rois.py           # HUD region detection for video analysis
  kill_flash_detector.py      # Kill-feed OCR + flash detection from video
  render_kill_overlay.py      # Timeline overlay rendering on video frames
data/                         # .dem files, .mp4 recordings (gitignored, large)
outputs/                      # Generated images, reports, CSVs
```

## Working Conventions

- Before running any command that could take more than a few seconds (video processing, OCR, ML inference), estimate duration upfront
- After generating output images, read them directly with the Read tool — don't ask the user to check
- ROI coordinates in calibrate_rois.py and kill_flash_detector.py use **relative fractions** (0.0–1.0), not absolute pixels — this makes them resolution-independent
- Bot count OCR uses: fix_dropped_1 (leading "1" lost in OTSU threshold) + median_filter(3) for stabilization
- The "activation shot" (first shot that starts the AimBotz challenge) is not a real engagement — exclude or handle it

## Dependencies

```
pip install demoparser2 pandas matplotlib opencv-python numpy
```

Optional: `pytesseract` + Tesseract OCR installed for video analysis scripts. `scipy` for signal processing in kill_flash_detector. `ffmpeg` on PATH for video frame extraction.
