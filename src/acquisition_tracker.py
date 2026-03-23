"""
acquisition_tracker.py
======================
Computes per-engagement crosshair acquisition timestamps from the gameplay
video, producing two new metrics:

  acquisition_ms  — ms from previous kill to crosshair-on-bot
  true_ttff_ms    — ms from crosshair-on-bot to first shot (pure reaction)

Why video instead of demo:
  Workshop/offline CS2 demos do not record per-tick entity state.
  parse_ticks() returns empty DataFrames for all props. Only competitive/
  GOTV demos have full entity snapshots. The gameplay MP4 is used instead.

Detection strategy (motion-settling):
  The CS2 crosshair is fixed at screen center. While the player is rotating
  to find the next bot, there is high motion in the crosshair zone. Once the
  player settles their aim on the bot, motion drops. The last low-motion
  frame before the first shot = acquisition moment.

  This avoids needing to classify what is in the zone — works regardless of
  bot skin, lighting, or map background.

Video details (training1):
  File   : Counter-Strike 2 - 2026-03-17 4-28-50 PM.mp4
  FPS    : 60
  Dur    : 73.7s
  Offset : video_time = demo_time + VIDEO_OFFSET (1.64s)

Usage:
  python acquisition_tracker.py                        # training1, default paths
  python acquisition_tracker.py video.mp4 engagements.csv 1.64
  python acquisition_tracker.py --calibrate 8          # plot first N engagements
"""

import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

VIDEO_PATH    = r"Counter-Strike 2 - 2026-03-17 4-28-50 PM.mp4"
CSV_PATH      = r"training1_engagements.csv"
VIDEO_OFFSET  = 1.64    # video_time = demo_time + VIDEO_OFFSET
ZONE_R        = 80      # crosshair zone half-width in pixels
SETTLE_THRESH = 4.0     # mean pixel diff below this = "settled on target"
FPS           = 60


# ── coordinate helpers ────────────────────────────────────────────────────────

def demo_to_video(t: float) -> float:
    return t + VIDEO_OFFSET

def video_to_demo(t: float) -> float:
    return t - VIDEO_OFFSET


# ── core frame analysis ───────────────────────────────────────────────────────

def zone_motion(frame, prev_frame, cx, cy, r) -> float:
    """Mean absolute pixel change in a square zone around (cx, cy)."""
    def gray_crop(f):
        return cv2.cvtColor(
            f[max(0, cy-r):cy+r, max(0, cx-r):cx+r],
            cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
    return float(np.mean(np.abs(gray_crop(frame) - gray_crop(prev_frame))))


def scan_window(cap, cx, cy, t_start_v, t_end_v):
    """
    Sequentially read frames in [t_start_v, t_end_v] (video seconds).
    Returns list of (video_time, motion_score) pairs.
    """
    cap.set(cv2.CAP_PROP_POS_MSEC, t_start_v * 1000)
    prev = None
    scores = []
    t = t_start_v

    while t < t_end_v:
        ret, frame = cap.read()
        if not ret:
            break
        if prev is not None:
            scores.append((t, zone_motion(frame, prev, cx, cy, ZONE_R)))
        prev = frame
        t += 1.0 / FPS

    return scores


def acquisition_from_scores(scores, settle_thresh=SETTLE_THRESH):
    """
    Given (time, motion) pairs for an engagement window, return the video
    timestamp of crosshair acquisition.

    Strategy: working backward from the last frame (closest to the shot),
    find the last contiguous run of low-motion frames. The start of that
    run is when the player settled their aim = acquisition.

    Fallback: minimum-motion frame.
    """
    if not scores:
        return None

    # Walk backward; find first (= most recent) low-motion frame, then
    # extend backward to find the start of that settled run.
    low = [(t, m) for t, m in scores if m <= settle_thresh]
    if not low:
        # No clearly settled frame — use minimum motion as best estimate
        return min(scores, key=lambda x: x[1])[0]

    # Last low-motion frame (closest to shot)
    last_low_t = low[-1][0]

    # Walk backward from there to find where the settling started
    settled_start_t = last_low_t
    for t, m in reversed(scores):
        if t > last_low_t:
            continue
        if m <= settle_thresh:
            settled_start_t = t
        else:
            break  # hit a high-motion frame — settling started after this

    return settled_start_t


# ── per-engagement processing ─────────────────────────────────────────────────

def process_engagement(cap, cx, cy, row):
    """
    Returns (acquisition_demo_s, motion_scores) for one engagement row.
    motion_scores is [(video_time, score), ...] — kept for calibration plots.
    """
    prev_kill_s  = row["engagement_start_s"]
    first_shot_s = row["shot_1_time_s"]

    if pd.isna(prev_kill_s) or pd.isna(first_shot_s):
        return None, []

    t_start_v = max(0.0, demo_to_video(prev_kill_s))
    t_end_v   = demo_to_video(first_shot_s)

    if t_end_v - t_start_v < 1.0 / FPS:
        # Window too narrow (e.g. one-tap immediately after kill)
        return video_to_demo(t_start_v), []

    scores = scan_window(cap, cx, cy, t_start_v, t_end_v)
    acq_v  = acquisition_from_scores(scores)

    return (video_to_demo(acq_v) if acq_v is not None else None), scores


# ── calibration plot ──────────────────────────────────────────────────────────

# Palette
C_MOTION    = "#2c7bb6"
C_THRESH    = "#e67e22"
C_ACQ       = "#27ae60"
C_SHOT      = "#8e44ad"
C_ROT_BG    = "#d4e8f7"   # rotation zone fill
C_REACT_BG  = "#d5f0de"   # reaction zone fill


def calibrate(cap, cx, cy, df, n=8, manual_acq_demo_s=None):
    """
    Plot motion profiles for the first N engagements on a shared relative
    x-axis (ms from engagement start).  Purpose: validate that the
    motion-settling detector correctly identifies crosshair acquisition.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(n, 1, figsize=(15, 2.6 * n),
                             gridspec_kw={"hspace": 0.55})
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i >= len(df):
            ax.set_visible(False)
            continue

        row      = df.iloc[i]
        eng_id   = int(row["engagement_id"])
        eng_start = row["engagement_start_s"]
        shot1_s  = row["shot_1_time_s"]

        # Window length in ms (relative timeline)
        window_ms = (shot1_s - eng_start) * 1000 if not pd.isna(shot1_s) else 0

        # ── handle degenerate windows (activation shot, instant kill) ──
        if window_ms < 30:
            ax.set_facecolor("#f5f5f5")
            ax.text(0.5, 0.5,
                    f"Eng {eng_id}  —  window {window_ms:.0f} ms  "
                    f"(activation shot or instant transition — excluded)",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#999999", style="italic")
            ax.set_xlim(0, 500)
            ax.set_yticks([])
            ax.set_xlabel("ms from engagement start", fontsize=8)
            continue

        acq_demo_s, scores = process_engagement(cap, cx, cy, row)

        if not scores:
            ax.text(0.5, 0.5, f"Eng {eng_id} — no frames",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=9, color="#999999")
            continue

        # ── relative timeline (ms from engagement start) ──
        times_ms  = [(video_to_demo(t) - eng_start) * 1000 for t, _ in scores]
        motion    = [m for _, m in scores]
        acq_ms    = (acq_demo_s - eng_start) * 1000 if acq_demo_s is not None else window_ms
        true_ttff = window_ms - acq_ms
        rotation  = acq_ms

        # ── background zones ──
        ax.axvspan(0,       acq_ms,    color=C_ROT_BG,   alpha=0.55, zorder=0, label="Rotation")
        ax.axvspan(acq_ms,  window_ms, color=C_REACT_BG, alpha=0.65, zorder=0, label="Reaction")

        # ── motion signal ──
        ax.plot(times_ms, motion, color=C_MOTION, linewidth=1.4,
                zorder=2, label="Zone motion")

        # ── settle threshold ──
        ax.axhline(SETTLE_THRESH, color=C_THRESH, linestyle="--",
                   linewidth=0.9, alpha=0.9, zorder=1,
                   label=f"Settle threshold ({SETTLE_THRESH})")

        # ── acquisition line ──
        ax.axvline(acq_ms, color=C_ACQ, linewidth=2.0, zorder=3)
        ax.text(acq_ms + 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"acq\n{acq_ms:.0f}ms", color=C_ACQ, fontsize=7.5,
                va="top", zorder=4)

        # ── first shot line ──
        ax.axvline(window_ms, color=C_SHOT, linewidth=2.0,
                   linestyle="-.", zorder=3, label="First shot")
        ax.text(window_ms - 2, ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else 1,
                f"shot\n{window_ms:.0f}ms", color=C_SHOT, fontsize=7.5,
                ha="right", va="top", zorder=4)

        # ── title ──
        hs    = row["hit_sequence"]
        ax.set_title(
            f"Eng {eng_id}   ·   rotation {rotation:.0f} ms   →   "
            f"reaction {true_ttff:.0f} ms   ·   seq: {hs}",
            fontsize=9, loc="left", pad=4, color="#222222"
        )

        # ── axes ──
        y_max = max(max(motion) * 1.15, SETTLE_THRESH * 3)
        ax.set_ylim(0, y_max)
        ax.set_xlim(-window_ms * 0.02, window_ms * 1.08)
        ax.set_ylabel("motion", fontsize=8, color="#555555")
        ax.tick_params(axis="both", labelsize=7.5)
        if i == n - 1:
            ax.set_xlabel("ms from engagement start", fontsize=9)

    # ── shared legend (first real axes) ──
    legend_patches = [
        mpatches.Patch(color=C_ROT_BG,  alpha=0.8, label="Rotation  (prev kill → crosshair on bot)"),
        mpatches.Patch(color=C_REACT_BG, alpha=0.9, label="Reaction  (crosshair on bot → first shot)"),
        plt.Line2D([0], [0], color=C_MOTION,  linewidth=1.4, label="Zone motion score"),
        plt.Line2D([0], [0], color=C_THRESH,  linewidth=1, linestyle="--", label=f"Settle threshold ({SETTLE_THRESH})"),
        plt.Line2D([0], [0], color=C_ACQ,     linewidth=2,   label="Detected acquisition"),
        plt.Line2D([0], [0], color=C_SHOT,    linewidth=2,   linestyle="-.", label="First shot"),
    ]
    fig.legend(handles=legend_patches, loc="upper right",
               fontsize=8, framealpha=0.9,
               bbox_to_anchor=(0.98, 0.98), ncol=2)

    fig.suptitle(
        f"Crosshair Acquisition — Motion Profile Analysis  ·  first {n} engagements\n"
        f"ZONE_R={ZONE_R}px   SETTLE_THRESH={SETTLE_THRESH}   offset={VIDEO_OFFSET}s",
        fontsize=11, y=1.005, color="#111111"
    )

    out = "acquisition_calibration.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(video_path, csv_path, video_offset, calibrate_n=None):
    global VIDEO_OFFSET
    VIDEO_OFFSET = video_offset

    df  = pd.read_csv(csv_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"ERROR: cannot open video: {video_path}")
        sys.exit(1)

    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cx, cy = w // 2, h // 2
    print(f"Video: {w}x{h} @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps  "
          f"crosshair zone: ({cx}±{ZONE_R}, {cy}±{ZONE_R})")

    if calibrate_n is not None:
        # Manual ground truth for first 8 engagements (demo seconds)
        # Derived from user's Google Sheet (X:YY second:frame @ 60fps, offset 1.64s)
        manual_acq = [
            5.383 - VIDEO_OFFSET,  # eng 1  e=5:23
            5.517 - VIDEO_OFFSET,  # eng 2  e=5:31
            6.350 - VIDEO_OFFSET,  # eng 3  e=6:21
            6.717 - VIDEO_OFFSET,  # eng 4  e=6:43
            7.350 - VIDEO_OFFSET,  # eng 5  e=7:21
            7.783 - VIDEO_OFFSET,  # eng 6  e=7:47
            8.467 - VIDEO_OFFSET,  # eng 7  e=8:28
            8.683 - VIDEO_OFFSET,  # eng 8  e=8:41
        ]
        calibrate(cap, cx, cy, df, n=calibrate_n, manual_acq_demo_s=manual_acq)
        cap.release()
        return

    # Full run
    acq_times = []
    for _, row in df.iterrows():
        eng_id = int(row["engagement_id"])
        acq_demo_s, _ = process_engagement(cap, cx, cy, row)
        acq_times.append(acq_demo_s)
        status = f"{acq_demo_s:.3f}s" if acq_demo_s is not None else "NONE"
        print(f"Eng {eng_id:3d}: acquisition={status}")

    cap.release()

    df["acquisition_s"]  = acq_times
    df["acquisition_ms"] = ((df["acquisition_s"] - df["engagement_start_s"]) * 1000).round(1)
    df["true_ttff_ms"]   = ((df["shot_1_time_s"]  - df["acquisition_s"])     * 1000).round(1)

    out = csv_path.replace(".csv", "_with_acquisition.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved: {out}")
    cols = ["engagement_id", "ttff_ms", "acquisition_ms", "true_ttff_ms"]
    print(df[cols].head(15).to_string(index=False))
    print(f"\nMean true_ttff_ms : {df['true_ttff_ms'].mean():.1f}")
    print(f"Mean ttff_ms      : {df['ttff_ms'].mean():.1f}")


if __name__ == "__main__":
    args = sys.argv[1:]

    if args and args[0] == "--calibrate":
        n = int(args[1]) if len(args) > 1 else 8
        main(VIDEO_PATH, CSV_PATH, VIDEO_OFFSET, calibrate_n=n)
    else:
        vp     = args[0] if len(args) > 0 else VIDEO_PATH
        cp     = args[1] if len(args) > 1 else CSV_PATH
        offset = float(args[2]) if len(args) > 2 else VIDEO_OFFSET
        main(vp, cp, offset)
