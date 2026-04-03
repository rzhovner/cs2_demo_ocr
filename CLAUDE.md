# CS2 Demo + Video Analyzer

## What This Is

Tools for analyzing CS2 AimBotz 90° 100-bot challenge performance. Two data sources:
- **Demo files** (`.dem`) → `cs2_engagement_parser.py` → per-engagement CSV (shots, hits, timing)
- **Gameplay video** (`.mp4`) → `audio_kill_detector.py` → kill timestamps via audio RMS energy

The demo parser is the primary data source. Audio kill detection is the primary video-analysis pipeline — fast (<1s), no OCR, no Tesseract. Visual detectors (killfeed, pixel-diff, kill-flash) exist for cross-validation but are secondary.

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
  audio_kill_detector.py         #   Audio RMS energy kill detector (PRIMARY pipeline)
  render_audio_kills.py          #   Video overlay: timeline strip + kill markers + edge glow
  render_kill_overlay.py         #   Bot-count timeline overlay on video (legacy, uses OCR)
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

## Primary Pipeline: Audio → Render

The established workflow for analyzing a gameplay video:

```bash
# 1. Detect kills from audio (fast, <1s per minute)
python3 src/audio_kill_detector.py video.mp4 --out kills.csv

# 2. Render overlay video with kill markers + original audio
python3 src/render_audio_kills.py --video video.mp4 --kills kills.csv --duration 0 --out overlay.mp4
ffmpeg -y -i overlay.mp4 -i video.mp4 -c:v copy -map 0:v -map 1:a -shortest final.mp4
```

Audio detection accuracy: ~98% recall / ~93% precision for tapping sessions. The ~7% false-positive rate (typically ~10 extra detections in 100 kills) correlates with audio confidence — kills with `confidence < 1.08` are most likely false positives. Visual confirmation signals (bot-count pixel delta, killfeed red pixels, kill counter brightness) were tested but lack per-event discriminating power for filtering individual false positives (see `plans/decisions.md`).

## Working Conventions

- Before running any command that could take more than a few seconds (video processing, OCR, ML inference), estimate duration upfront
- After generating output images, read them directly with the Read tool — don't ask the user to check
- ROI coordinates use **relative fractions** (0.0–1.0), not absolute pixels — resolution-independent
- The "activation shot" (first shot that starts the AimBotz challenge) is not a real engagement — exclude or handle it
- Audio is the primary kill detection pipeline; do not default to tesseract/OCR-based detection

## Dependencies

```
pip install demoparser2 pandas matplotlib opencv-python numpy
```

Required for audio pipeline: `scipy` for signal processing, `ffmpeg` on PATH for audio extraction and video muxing.
Optional (legacy visual detectors only): `pytesseract` + Tesseract OCR for kill_flash_detector.py.
