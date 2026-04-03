# Architectural Decisions

## Why Video Analysis Exists Alongside Demo Parsing

**Decision:** Build video/audio-based kill detectors even though the demo parser extracts kill timestamps directly.

**Context:** CS2 workshop/offline demos do not record per-tick player state (positions, view angles). `parse_ticks()` returns empty DataFrames. The demo only contains discrete game events (`weapon_fire`, `player_death`, `player_hurt`). This means the demo cannot tell us where the crosshair was pointing between shots -- only when shots and kills happened.

**Consequence:** Crosshair acquisition timing (true TTFF) requires video analysis. The video detectors also serve as an independent validation source for demo-parser output.

## Why Multiple Kill Detectors

**Decision:** Implement 4 separate detectors (audio, kill flash, killfeed, pixel diff) rather than one unified detector.

**Context:** Each signal type has different failure modes:
- Audio: fast and reliable for tapping, misses quiet kills in spray bursts
- Kill flash brightness: no external dependencies, but noisy
- Killfeed red pixels: independent of HUD layout changes, but sensitive to UI scaling
- OCR bot count: most accurate but requires Tesseract and is slow

**Consequence:** Multiple detectors allow cross-validation and signal fusion. The audio detector is the recommended first pass; visual detectors provide confirmation.

## Why Relative ROI Coordinates

**Decision:** Define all HUD regions as relative fractions (0.0-1.0) instead of absolute pixel coordinates.

**Context:** CS2 gameplay is recorded at varying resolutions (1280x960, 1920x1080, etc.). Absolute pixel coordinates would break across resolutions.

**Consequence:** ROIs are resolution-independent. Conversion is a single multiply: `pixel_x = rel_x * frame_width`. All calibrated on 1280x960 reference resolution.

## Audio as Primary Kill Detection Pipeline

**Decision:** Use `audio_kill_detector.py` as the primary (and default) kill detection approach for video analysis. Visual detectors remain available for cross-validation but are not the default.

**Context:** Tesseract OCR was originally the most accurate signal for reading the bot-count HUD, but it's a heavy external dependency and slow. Audio RMS energy detection runs in <1s per minute, requires only ffmpeg + scipy, and achieves ~98% recall / ~93% precision on tapping sessions.

**Consequence:** The standard workflow is audio detection → render overlay. No OCR or Tesseract in the default path.

## Visual Confirmation Signals Cannot Filter Individual Audio False Positives

**Decision:** Do not rely on visual HUD signals (bot-count pixel delta, kill counter brightness, killfeed red pixels, center kill icon) for per-event false positive filtering on audio detections.

**Context:** Tested all four visual signals against 111 audio-detected kills in `warmup_20260322_1.mp4` (100 real + 1 post-challenge + 10 false positives):
- **Bot-count pixel delta** (22x16px ROI): fires on both kills and respawns (~199 peaks at thresh=5, ~110 at thresh=20). No discriminating power — mean delta near false positives ≈ mean delta near real kills (26.9 vs 25.2).
- **Kill counter brightness** (ROI at 0.2773, 0.9427): this ROI is actually the HP display, not the kill counter. Brightness flash prominence=3 gives 100 peaks but poor temporal alignment with audio kills (best offset match: 63/111).
- **Killfeed red rise**: baseline red% is high in fast-kill warmup sessions (killfeed always populated). Red rise threshold=4 yields only 14 peaks; delta-gated approach yields 78 — not enough.
- **Center icon brightness**: 140 peaks at prominence=3, only 86/111 matched via Hungarian algorithm.

**Consequence:** For false positive reduction, audio confidence thresholding (`conf >= 1.08` → 101 kills) is more reliable than any visual fusion signal tested. Future approaches might include OCR-based bot count reading (if Tesseract is reintroduced) or improved ROI calibration for non-training1 videos.

## Why Demo Parser Uses Kill Events as Engagement Boundaries

**Decision:** Define engagement N as all shots between kill N-1 and kill N, rather than using spatial proximity or time gaps.

**Context:** In the AimBotz 90-degree bot challenge, bots spawn one at a time. The player kills one bot, rotates to the next, and fires. Kill events are the natural boundary because there is exactly one target alive at any time.

**Consequence:** TTFF = time from previous kill to first shot at next target. This conflates rotation speed with reaction time, but is unambiguous and doesn't require spatial data (which the demo doesn't provide anyway).
