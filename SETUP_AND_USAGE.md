# CS2 Engagement Parser — Setup & Usage

## Quick Setup

```bash
# 1. Make sure you have Python 3.8+
python --version

# 2. Install dependencies
pip install demoparser2 pandas

# 3. Run against your demo
python cs2_engagement_parser.py "path/to/your_demo.dem"
```

That's it. The script auto-detects tick rate, identifies the human player, and outputs a CSV.

---

## What It Does

Takes a CS2 `.dem` file from a **90° 100-kill bot challenge** (Aim Botz, etc.) and produces a full engagement map CSV — one row per kill, with every shot timestamped and tagged as hit or miss.

### Output Columns

| Column | Description |
|---|---|
| `engagement_id` | Sequential engagement number (1, 2, 3...) |
| `kill_tick` | Raw tick when the bot died |
| `kill_time_s` | Kill time in seconds from demo start |
| `engagement_start_s` | Estimated engagement start (previous kill time) |
| `ttff_ms` | **Time to First Fire** — ms from engagement start to first shot |
| `ttk_ms` | **Time to Kill** — ms from first shot to kill |
| `total_duration_ms` | Full engagement duration (start to kill) |
| `shots_fired` | Total weapon_fire events for this engagement |
| `shots_hit` | Shots that matched a player_hurt event |
| `shots_missed` | shots_fired - shots_hit |
| `hit_rate` | shots_hit / shots_fired |
| `headshot` | Whether the killing blow was a headshot |
| `damage_dealt` | Total damage across all hits |
| `weapon` | Weapon used for the kill |
| `hit_sequence` | Per-shot pattern, e.g. `M,Y,Y` |
| `inter_engagement_gap_ms` | Recovery time between previous kill and this engagement's first shot |
| `shot_1_time_s` ... `shot_N_time_s` | Individual shot timestamps |
| `shot_1_hit` ... `shot_N_hit` | Y/M for each individual shot |

### Terminal Summary

The script also prints aggregate KPIs to the console:

- Median/mean TTFF (reaction proxy)
- Median/mean TTK
- Hit rate, headshot rate
- One-tap rate and max one-tap streak
- Shots-to-kill distribution
- Pacing gaps between engagements
- TTFF consistency (coefficient of variation)

---

## Options

```bash
# Custom output path
python cs2_engagement_parser.py demo.dem --output my_results.csv

# Override tick rate (if auto-detect is wrong)
python cs2_engagement_parser.py demo.dem --tick-rate 128

# Print summary only, no CSV
python cs2_engagement_parser.py demo.dem --summary-only

# More shot columns (default: 15)
python cs2_engagement_parser.py demo.dem --max-shots 20
```

---

## How Engagements Are Detected

Since this is a bot challenge (not a live match), the logic is straightforward:

1. **Kill events** (`player_death`) define engagement boundaries
2. All `weapon_fire` events between kill N-1 and kill N belong to engagement N
3. Each shot is matched against `player_hurt` events within a small tick window to determine hit/miss
4. The **engagement start** for engagement N is the tick of kill N-1 (the moment the previous bot died and you're acquiring a new target)

This means **TTFF ≈ target acquisition time + reaction time** — the gap between your last kill and your first shot at the next bot.

---

## Mapping to Your Manual Spreadsheet

Your manually tagged data had these columns — here's how they correspond:

| Your Column | Parser Column |
|---|---|
| `engagement #` (A) | `engagement_id` |
| `e frametime` (B) | `engagement_start_s` (converted from X:YY to seconds) |
| `hit` (C) — Y/M | Not directly — the parser tracks per-shot hit/miss. A `hit_rate` of 1.0 = all shots landed |
| `shot#1` (D) | `shot_1_time_s` |
| `shot#2` (E) | `shot_2_time_s` |
| ... | ... |

The parser gives you *more* than your manual sheet: per-shot hit/miss, TTFF, TTK, damage, headshot, pacing, and aggregate stats.

---

## Troubleshooting

**"No weapon_fire events found"**
- Make sure the demo is from an actual gameplay session (not a GOTV broadcast of an empty server)
- Some workshop maps use non-standard bot spawning — the parser expects normal kill events

**TTFF values look wrong**
- Check tick rate: if the parser auto-detects 64 but you were on a 128-tick server, pass `--tick-rate 128`
- The first engagement won't have a meaningful TTFF (no previous kill to anchor from)

**Hit matching seems off**
- The default matching window is 8 ticks. If you're on a high-tick server or the weapon has slow projectile speed, you may need to edit `HIT_MATCH_WINDOW_TICKS` in the script (line ~32)

---

## Next Steps

Once you have the CSV, you can:
- Drop it into Google Sheets and recreate your colored engagement map at full scale
- Feed it into a visualization tool (Python, R, or a future dashboard)
- Compare multiple sessions by running the parser on different demos
- Look at TTFF distribution over time to spot warmup curves / fatigue patterns
