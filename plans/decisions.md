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

## Why Demo Parser Uses Kill Events as Engagement Boundaries

**Decision:** Define engagement N as all shots between kill N-1 and kill N, rather than using spatial proximity or time gaps.

**Context:** In the AimBotz 90-degree bot challenge, bots spawn one at a time. The player kills one bot, rotates to the next, and fires. Kill events are the natural boundary because there is exactly one target alive at any time.

**Consequence:** TTFF = time from previous kill to first shot at next target. This conflates rotation speed with reaction time, but is unambiguous and doesn't require spatial data (which the demo doesn't provide anyway).
