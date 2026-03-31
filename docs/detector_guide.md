# Kill Detector Guide

Four independent approaches for detecting kills from CS2 AimBotz gameplay recordings. Each uses different signals and has different accuracy/speed tradeoffs. All validate against demo-parser ground truth (`*_engagements.csv`).

## Detector Comparison

| Detector | Signal(s) | Requires | Speed | Best For |
|---|---|---|---|---|
| `audio_kill_detector.py` | Audio RMS energy peaks | ffmpeg | <1s per minute of video | Fast, reliable first pass |
| `kill_flash_detector.py` | Brightness + frame delta + OCR | OpenCV, Tesseract | Slow (frame-by-frame + OCR) | Highest accuracy with OCR confirmation |
| `killfeed_detector.py` | Kill feed red pixels + shift delta | OpenCV | Medium (frame-by-frame) | No OCR dependency |
| `pixel_diff_detector.py` | Kill counter brightness + bot-count delta | OpenCV | Medium (frame-by-frame) | No OCR, no Tesseract |

## Audio Kill Detector

**File:** `src/audio_kill_detector.py`

Kills produce distinctively loud audio events (shot impact + death sound). Short-time RMS energy with adaptive percentile thresholding detects these peaks.

- **Tapping sessions:** ~98% recall, ~93% precision (F1 ~96%)
- **Spraying sessions:** ~91% recall, ~98% precision (F1 ~94%)
- Auto-detects session type (tap vs spray) from energy distribution
- Quiet kills embedded in spray bursts may be missed (energy indistinguishable from non-lethal shots)

```bash
python src/audio_kill_detector.py video.mp4
python src/audio_kill_detector.py video.mp4 --validate engagements.csv
python src/audio_kill_detector.py video.mp4 --session-type spray
```

## Kill Flash Detector (Three-Signal)

**File:** `src/kill_flash_detector.py`

Three fused signals:
1. **Brightness spikes** in HUD regions (kill counter, center icon) when a kill registers
2. **Frame delta** in ammo counter (30->29->30 cycle per shot, no OCR needed)
3. **Bot count OCR** reading T-side alive count (decrement = kill)

Signal 3 (OCR) is the most reliable but requires Tesseract. Signals 1+2 work without OCR but are noisier.

```bash
python src/kill_flash_detector.py --calibrate          # check ROI alignment
python src/kill_flash_detector.py --detect --roi center_icon
python src/kill_flash_detector.py --validate engagements.csv
```

## Killfeed Detector

**File:** `src/killfeed_detector.py`

Tracks the CS2 kill feed UI (top-right corner):
1. **Red pixel percentage rise** in the top kill feed slot (new entry appears with red text)
2. **Frame delta** when kill feed shifts down (new entry pushes old ones)

Merged with deduplication window. Pure OpenCV, no OCR.

```bash
python src/killfeed_detector.py  # uses hardcoded paths, edit VIDEO_PATH
```

## Pixel Diff Detector

**File:** `src/pixel_diff_detector.py`

Two fused signals:
1. **Brightness delta** in kill counter HUD region (primary signal)
2. **Bot-count pixel delta** for confirmation (digits change on kill/respawn)

Fusion rule: brightness peak + bot-count delta within confirmation window = kill. No OCR, no Tesseract.

```bash
python src/pixel_diff_detector.py  # uses hardcoded paths, edit VIDEO_PATH
```

## Common Validation Pattern

All detectors share a validation workflow:
1. Detect kill timestamps from video/audio
2. Estimate time offset between video and demo via cross-correlation
3. Score detections against ground truth (TP/FP/FN, recall, precision, F1)
4. Generate report with timing error histograms

Ground truth comes from `cs2_engagement_parser.py` output. The `engagement_start_s` column provides the canonical kill timestamp from the demo file.
