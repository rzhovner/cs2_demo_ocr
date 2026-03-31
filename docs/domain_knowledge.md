# Domain Knowledge

## Data Pipeline

```
.dem --> cs2_engagement_parser.py --> *_engagements.csv --> plot_hitrate.py --> *_hitrate.png
.mp4 --> calibrate_rois.py --> ROI coordinates --> kill_flash_detector.py --> *_kill_times.csv
.mp4 --> audio_kill_detector.py --> *_audio_kills.csv
```

The demo parser is the primary data source. Video/audio analysis supplements it for metrics the demo format cannot provide (crosshair acquisition timing, per-frame visual state).

## AK-47 Timing (CS2)

- **Cycle time: 100ms (0.1s) -- 600 RPM -- 10 rounds/second**
- No two shots can be closer than 100ms -- this is a hard physical constraint enforced by the engine
- `SHOT_MIN_GAP_S` must be `0.100`, never lower (0.05 was a past bug that caused false shot detections)
- Kill registers on the same tick as the lethal shot

## TTFF: Two Definitions

| Metric | Measures | Source |
|---|---|---|
| Parser TTFF (`ttff_ms`) | prev kill --> first shot (rotation + acquisition + reaction) | Demo |
| True TTFF (planned) | crosshair on bot --> first shot (pure reaction time) | Video |

The parser TTFF conflates rotation speed with reaction time. A slow 180-degree turn and a fast snap both get lumped into the same metric. True TTFF requires video-based crosshair tracking to isolate the moment the crosshair settles on the target from the moment the shot fires.

`acquisition_tracker.py` implements an experimental approach: tracking screen-center pixel motion to detect when the crosshair stops moving (acquisition moment), then measuring the gap to first shot (true reaction time).

## Workshop Demo Limitation

Workshop/offline server CS2 demos do **NOT** record per-tick player state (positions, view angles, velocities). Calling `parse_ticks()` on these demos returns empty DataFrames. Only game events are captured:

- `weapon_fire` (shot timestamps)
- `player_death` (kill timestamps, headshot flag)
- `player_hurt` (damage events, hit confirmation)

This is why video analysis is needed for acquisition timing -- the demo simply doesn't contain the player's crosshair position over time.

## ROI Coordinate System

All HUD region-of-interest (ROI) coordinates use **relative fractions** (0.0-1.0), not absolute pixels. This makes them resolution-independent. The reference resolution for calibration is 1280x960.

Example: `ROI_BOT_COUNT_REL = (0.5063, 0.0604, 0.5234, 0.0771)` means (x1=50.6%, y1=6.0%, x2=52.3%, y2=7.7%) of the frame.

To convert: `pixel_x = rel_x * frame_width`

## Bot Count OCR Challenges

The T-side alive count uses small pixel-art digits that OCR engines frequently misread:

- Leading "1" lost in OTSU thresholding (e.g., "12" reads as "2") -- fixed by `fix_dropped_1()`
- Common character confusions: `1 -> T/n/N/w`, `0 -> m/y/D/Q/O`
- Stabilized with `median_filter(window=3)` to suppress frame-to-frame noise

## Activation Shot

The first shot in a demo session is not a real engagement -- it's the shot that starts the AimBotz challenge (activates the bot spawning). It should be excluded from analysis or flagged as `engagement_id=0`.
