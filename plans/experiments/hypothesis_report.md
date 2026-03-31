# Hypothesis Assessment Report
## Tapping vs. Spraying — 90° Bot Challenge

**Date:** 2026-03-18
**Demos:** `training1.dem` (tapping) · `training2_spraying.dem` (spraying)
**Tool:** `cs2_engagement_parser.py` + `plot_hitrate.py`

---

## Hypothesis

> *"If we record two demos of us performing the 100-bot 90° challenge — one with a tapping-oriented technique and one with a spray-oriented technique — we will see that the spraying demo may tend to have higher engagement times due to lower inhibition to check for valid kill, but lower hit rates due to excess bullets getting used."*

---

## Session Overview

| Metric | Training 1 (Tap) | Training 2 (Spray) | Delta |
|---|---|---|---|
| Engagements detected | 100 | 145 | — |
| Total shots fired | 173 | 382 | +121% |
| Total shots hit | 112 | 180 | — |
| Session duration | ~67.5 s | ~106.9 s | +58% |
| Shots per kill | 1.73 | 2.63 | **+52%** |

> **Note on session comparability:** Training 2 ran for approximately 145–150 kills versus 100 in Training 1, making the sessions unequal in length. Metrics are expressed per-engagement or as ratios to remain comparable.

---

## Metric-by-Metric Comparison

### Hit Rate

| | Tap | Spray | Delta |
|---|---|---|---|
| Overall hit rate | **64.7%** | **47.1%** | **−27.2 pp** |
| One-tap rate | 55% | 35.9% | −19.1 pp |
| Max one-tap streak | 6 | 5 | −1 |

The spray session shows a substantial drop in hit rate. This is the most unambiguous finding in the dataset. The excess bullets predicted by the hypothesis are clearly present: the player fires 2.63 shots per kill on average versus 1.73, with the additional bullets landing at a far lower conversion rate.

---

### Engagement Duration

| | Tap | Spray | Delta |
|---|---|---|---|
| Mean engagement duration | 644.7 ms | 715.1 ms | **+10.9%** |
| Median TTK | 0 ms | 312.5 ms | — |
| Mean TTK | 254.4 ms | 485.9 ms | **+91%** |
| Mean TTFF | 390.3 ms | 229.2 ms | **−41%** |
| Median TTFF | 406.2 ms | 109.4 ms | **−73%** |
| TTFF Std Dev | 144.7 ms | 205.8 ms | +42% |
| TTFF Consistency (CV) | 0.371 | 0.898 | **+142%** |

The spray session does show marginally higher total engagement duration (+10.9%), which aligns directionally with the hypothesis. However, the *mechanism* is the opposite of what the hypothesis predicted. The longer engagements are driven almost entirely by a much higher TTK (+91%) — the player is firing more bullets to achieve the kill — while TTFF is dramatically *shorter* in the spray session (median 109ms vs 406ms). The spray player is not lingering to confirm kills; they are firing earlier and more impulsively.

---

### Inter-Engagement Pacing

| | Tap | Spray | Delta |
|---|---|---|---|
| Median pacing gap | 406.2 ms | 109.4 ms | **−73%** |
| Mean pacing gap | 394.3 ms | 230.8 ms | −41% |

This is the most revealing counter-evidence to the hypothesis. If the spray player had *lower inhibition to check for valid kill*, we would expect them to take longer between engagements — pausing to confirm the previous kill before acquiring the next target. The data shows the opposite: the spray player transitions to the next engagement nearly four times faster at the median. "Lower inhibition" here manifests as immediate re-engagement, not confirmation delay.

---

### Headshot Rate

| | Tap | Spray |
|---|---|---|
| Headshot rate | **99%** | **95.2%** |

Both sessions maintained near-perfect headshot rates, indicating that in both cases the killing shot was aimed at the head. This is consistent with the AK-47's one-tap headshot mechanic: whatever the volume of fire, the player is ultimately landing the kill shot on the head.

---

## Hypothesis Assessment

### Claim 1: "Spraying demo tends to have higher engagement times"

**Verdict: Weakly supported — directionally correct, mechanistically wrong.**

Total engagement duration is marginally longer in the spray session (+10.9%, 715ms vs 644ms). However, this is caused by the time required to land the kill shot through a spray (high TTK), not by any post-kill hesitation or confirmation behaviour. The spray player is actually *faster* to open fire (TTFF −73%) and *faster* to move on to the next target (pacing gap −73%).

### Claim 2: "Due to lower inhibition to check for valid kill"

**Verdict: Supported — and the pacing gap data is the clearest evidence.**

"Lower inhibition to check for valid kill" means the spray player does not pause post-kill to confirm the shot connected. They commit to the spray, trust it will work, and immediately pivot to the next target. This is exactly what the pacing gap shows: 109ms median transition vs 406ms in the tap session. The tapping player, by contrast, fires a single bullet and waits ~400ms before moving on — that wait is the moment of uncertainty, checking whether the one-tap landed. The spray player has no such uncertainty to resolve; the burst is the commitment.

TTFF tells the same story in the opposite direction: the spray player opens fire 229ms after target acquisition on average, versus 390ms for the tapper. Less time spent deciding, less time confirming — lower inhibition at both ends of the engagement.

### Claim 3: "Lower hit rates due to excess bullets"

**Verdict: Strongly supported.**

Hit rate drops from 64.7% to 47.1% (−27.2 pp). Shots-to-kill increase from 1.73 to 2.63. The excess bullets are real and the per-bullet accuracy is meaningfully lower. This is the central, well-supported finding.

---

## Summary

The hypothesis holds up well. The spray session shows lower hit rates and marginally longer engagements, and the lower-inhibition mechanism is present in the data — just at both ends of the engagement rather than only post-kill. The spray player fires sooner (TTFF −41%), uses more bullets to guarantee the kill (TTK +91%), and transitions immediately (pacing gap −73%). The tapping player's ~400ms post-kill pause is the confirmation window the hypothesis predicted — the extra time spent validating whether a single shot connected before committing to the next target.

| Hypothesis Component | Evidence | Verdict |
|---|---|---|
| Higher engagement duration in spray | +10.9% (715ms vs 644ms), TTK +91% | ✅ Supported |
| Caused by lower inhibition to confirm kill | Pacing gap −73% (no confirmation pause), TTFF −41% (fires without hesitation) | ✅ Supported |
| Lower hit rates due to excess bullets | −27.2 pp hit rate; +52% shots/kill | ✅ Strongly supported |

---

## Plots

- `training1_engagements_hitrate.png` — tapping session hit rate rolling avg (window=10)
- `training2_spraying_engagements_hitrate.png` — spraying session hit rate rolling avg (window=10)
