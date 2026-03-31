# Glossary

**Acquisition** -- The moment the crosshair settles on a target. Measured by `acquisition_tracker.py` via screen-center pixel motion analysis.

**Activation shot** -- The first shot in an AimBotz session that starts the bot challenge. Not a real engagement; should be excluded from analysis.

**Engagement** -- One kill sequence: from the previous bot's death (or session start) through all shots fired until the current bot dies. One row in the engagements CSV.

**Hit sequence** -- Per-shot pattern showing hits and misses, e.g., `M,Y,Y` means miss, hit, hit. `Y` = hit, `M` = miss.

**Inter-engagement gap** -- Time between one kill and the first shot at the next target. Same as TTFF in this project's context.

**One-tap** -- Killing a bot with a single shot (1 shot fired, 1 hit). AK-47 headshot does 111 damage, enough to one-tap bots.

**Pacing gap** -- Time between consecutive engagements. Reflects how quickly the player transitions from one target to the next.

**ROI (Region of Interest)** -- A rectangular area of the video frame used for signal extraction (brightness, OCR, pixel delta). Defined as relative fractions (0.0-1.0) for resolution independence.

**TTFF (Time to First Fire)** -- Milliseconds from engagement start (previous kill) to the first shot of the current engagement. Conflates rotation time + acquisition time + reaction time. See also: True TTFF.

**TTK (Time to Kill)** -- Milliseconds from the first shot to the killing shot. Zero for one-taps.

**True TTFF** -- Milliseconds from crosshair acquisition (settling on target) to first shot. Isolates pure reaction time by removing rotation time. Requires video analysis.

**Video offset** -- The time difference between video timestamps and demo timestamps. Estimated via cross-correlation of detected events against ground truth. `demo_time = video_time - offset`.
