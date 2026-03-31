# CS2 Demo + Video Analyzer

## What This Is

Tools for analyzing CS2 AimBotz 90° 100-bot challenge performance. Two data sources:
- **Demo files** (`.dem`) → `cs2_engagement_parser.py` → per-engagement CSV (shots, hits, timing)
- **Gameplay video** (`.mp4`) → `kill_flash_detector.py` → kill timestamps via OCR + flash detection

The demo parser is the primary data source. Video analysis supplements it for metrics the demo format can't provide (crosshair acquisition timing).

See `docs/domain_knowledge.md` for AK-47 timing (100ms cycle), TTFF definitions, data pipeline, workshop demo limitations, and ROI coordinate conventions.

## File Layout

```
src/                             # Source code (all Python scripts)
  cs2_engagement_parser.py       #   .dem → per-engagement CSV (main parser)
  plot_hitrate.py                #   Rolling avg hit rate chart from CSV
  acquisition_tracker.py         #   Video-based crosshair acquisition (WIP)
  calibrate_rois.py              #   HUD region detection for video analysis
  kill_flash_detector.py         #   Three-signal kill detector (brightness + delta + OCR)
  killfeed_detector.py           #   Kill-feed red pixel detector
  pixel_diff_detector.py         #   Brightness + bot-count delta detector
  audio_kill_detector.py         #   Audio RMS energy kill detector
  render_kill_overlay.py         #   Bot-count timeline overlay on video
  render_audio_accuracy.py       #   Audio accuracy overlay on video
docs/                            # Human-readable knowledge and guides
  domain_knowledge.md            #   CS2 mechanics, pipeline, demo limitations
  detector_guide.md              #   Detector comparison and usage
  setup_and_usage.md             #   Installation and CLI usage
  glossary.md                    #   Project terminology
plans/                           # Project management
  roadmap.md                     #   Priorities and tech debt
  decisions.md                   #   Architectural decision log
  experiments/                   #   Hypothesis-driven experiment reports
tests/                           # Tests and ground truth
  test_scoring.py                #   Unit tests for score/match functions
  test_detectors.py              #   Integration test skeletons
  ground_truth/                  #   Canonical engagement CSVs from demo parser
data/                            # .dem files, .mp4 recordings (gitignored, large)
outputs/                         # Generated images, charts, overlay videos
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
