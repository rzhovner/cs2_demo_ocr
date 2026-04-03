# Kill Detector Guide

Four independent approaches for detecting kills from CS2 AimBotz gameplay recordings. **Audio is the primary pipeline.** Visual detectors exist for cross-validation but cannot reliably filter individual audio false positives (see `plans/decisions.md`).

## Detector Comparison

| Detector | Signal(s) | Requires | Speed | Status |
|---|---|---|---|---|
| `audio_kill_detector.py` | Audio RMS energy peaks | ffmpeg, scipy | <1s per minute | **PRIMARY** — use this first |
| `kill_flash_detector.py` | Brightness + frame delta + OCR | OpenCV, Tesseract | Slow (frame-by-frame + OCR) | Legacy — OCR dependency |
| `killfeed_detector.py` | Kill feed red pixels + shift delta | OpenCV | Medium (frame-by-frame) | Secondary — weak in fast-kill sessions |
| `pixel_diff_detector.py` | Kill counter brightness + bot-count delta | OpenCV | Medium (frame-by-frame) | Secondary — noisy at 22x16px ROI |

## Audio Kill Detector

**File:** `src/audio_kill_detector.py`

Kills produce distinctively loud audio events (shot impact + death sound). Short-time RMS energy with adaptive percentile thresholding detects these peaks.

- **Tapping sessions:** ~98% recall, ~93% precision (F1 ~96%)
- **Spraying sessions:** ~91% recall, ~98% precision (F1 ~94%)
- Auto-detects session type (tap vs spray) from energy distribution
- Quiet kills embedded in spray bursts may be missed (energy indistinguishable from non-lethal shots)

```bash
# Detect kills
python3 src/audio_kill_detector.py video.mp4 --out kills.csv

# Detect + validate against demo ground truth
python3 src/audio_kill_detector.py video.mp4 --validate engagements.csv

# Force session type (default: auto-detect)
python3 src/audio_kill_detector.py video.mp4 --session-type spray
```

### Rendering Overlay Video

```bash
# Render overlay (timeline strip, kill markers, edge glow)
python3 src/render_audio_kills.py --video video.mp4 --kills kills.csv --duration 0 --out overlay.mp4

# Mux original game audio back in
ffmpeg -y -i overlay.mp4 -i video.mp4 -c:v copy -map 0:v -map 1:a -shortest final.mp4
```

### Known Limitations

- **~93% precision on tapping sessions**: ~10 false positives per 100 kills. These are loud non-lethal audio events indistinguishable from kill sounds by energy alone.
- **Confidence thresholding**: `confidence >= 1.08` (energy/threshold ratio) effectively trims false positives for standard 100-bot challenges. The 10 lowest-confidence detections are most likely false.
- **Visual fusion does not help**: Bot-count pixel delta, killfeed red%, kill counter brightness, and center icon brightness were all tested for per-event false positive filtering — none had sufficient discriminating power (see `plans/decisions.md`).

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

## Rendering Tools

| Script | Input | Output | Description |
|---|---|---|---|
| `render_audio_kills.py` | video + audio kills CSV | overlay .mp4 | Timeline strip, kill markers, edge glow, kill count |
| `render_kill_overlay.py` | video + OCR kill data | overlay .mp4 | Bot-count timeline (legacy, requires OCR data) |
| `render_audio_accuracy.py` | video + audio kills + ground truth | overlay .mp4 | Accuracy comparison overlay |
