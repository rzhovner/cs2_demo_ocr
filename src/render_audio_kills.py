"""
render_audio_kills.py
=====================
Render gameplay video with audio-detected kill markers overlaid.

Overlay elements:
  - Bottom timeline strip (scrolling, with kill markers + playhead)
  - Edge glow on kill events
  - Top-right kill counter
  - Energy waveform in the strip

Usage:
  python src/render_audio_kills.py --video VIDEO --kills CSV [--duration 15] [--out OUT]
"""

import argparse
import cv2
import numpy as np
import pandas as pd

# ── rendering constants ──────────────────────────────────────────────────────
STRIP_H = 80
STRIP_PAST_S = 3.0
STRIP_FUTURE_S = 3.0

GLOW_RAMP_S = 0.4
GLOW_FADE_S = 0.2
GLOW_MAX_ALPHA = 0.45
GLOW_THICKNESS = 5

# Colors (BGR)
C_KILL = (60, 60, 235)       # red
C_KILL_LOW = (80, 130, 235)  # dimmer red for low-confidence
C_WHITE = (255, 255, 255)
C_AMBER = (0, 190, 245)
C_BG = (26, 26, 26)
C_PLAYHEAD = (220, 220, 220)
C_ENERGY = (185, 128, 41)    # blue-ish for energy waveform
C_THRESH = (80, 80, 120)     # dim line for threshold

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Render audio kill overlay")
    p.add_argument("--video", required=True)
    p.add_argument("--kills", required=True, help="Audio kills CSV (from audio_kill_detector)")
    p.add_argument("--duration", type=float, default=15.0, help="Seconds to render (0=full)")
    p.add_argument("--out", default=None)
    return p.parse_args()


# ── edge glow ────────────────────────────────────────────────────────────────

def compute_glow(t, kill_times):
    best = 0.0
    for kt in kill_times:
        dt = t - kt
        if -GLOW_RAMP_S <= dt < 0:
            best = max(best, 1.0 - abs(dt) / GLOW_RAMP_S)
        elif 0 <= dt <= GLOW_FADE_S:
            best = max(best, 1.0 - dt / GLOW_FADE_S)
    return best


def draw_edge_glow(frame, intensity, h, w):
    if intensity < 0.02:
        return
    alpha = intensity * GLOW_MAX_ALPHA
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (GLOW_THICKNESS, h), C_AMBER, -1)
    cv2.rectangle(overlay, (w - GLOW_THICKNESS, 0), (w, h), C_AMBER, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ── timeline strip ───────────────────────────────────────────────────────────

def draw_strip(frame, t, kill_times, kill_confs, n_kills, w, h):
    strip_y = h - STRIP_H
    window = STRIP_PAST_S + STRIP_FUTURE_S
    t_left = t - STRIP_PAST_S
    t_right = t + STRIP_FUTURE_S

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, strip_y), (w, h), C_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    margin = 12
    graph_x0 = margin
    graph_x1 = w - margin
    graph_w = graph_x1 - graph_x0
    mid_y = strip_y + STRIP_H // 2

    def time_to_x(ts):
        frac = (ts - t_left) / window
        return int(graph_x0 + frac * graph_w)

    # Time axis
    cv2.line(frame, (graph_x0, mid_y), (graph_x1, mid_y), (50, 50, 50), 1)

    # Second markers
    for s in range(max(0, int(t_left)), int(t_right) + 1):
        x = time_to_x(s)
        if graph_x0 <= x <= graph_x1:
            cv2.line(frame, (x, mid_y - 4), (x, mid_y + 4), (80, 80, 80), 1)
            cv2.putText(frame, f"{s}s", (x - 8, mid_y + 20),
                        FONT, 0.3, (100, 100, 100), 1, cv2.LINE_AA)

    # Kill markers — vertical lines + circles
    for kt, conf in zip(kill_times, kill_confs):
        if t_left <= kt <= t_right:
            x = time_to_x(kt)
            color = C_KILL if conf >= 1.1 else C_KILL_LOW
            # Vertical tick
            cv2.line(frame, (x, strip_y + 8), (x, h - 24), color, 2, cv2.LINE_AA)
            # Circle at top
            cv2.circle(frame, (x, strip_y + 8), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, strip_y + 8), 4, C_WHITE, 1, cv2.LINE_AA)

    # Playhead
    ph_x = time_to_x(t)
    cv2.line(frame, (ph_x, strip_y + 2), (ph_x, h - 2), C_PLAYHEAD, 2)

    # Top border
    cv2.line(frame, (0, strip_y), (w, strip_y), (60, 60, 60), 1)

    # Kill count — left side of strip
    count_text = f"KILLS {n_kills}"
    cv2.putText(frame, count_text, (margin + 4, strip_y + 16),
                FONT, 0.4, C_KILL, 1, cv2.LINE_AA)



# ── flash overlay on kill frame ──────────────────────────────────────────────

def draw_kill_flash(frame, t, kill_times, h, w):
    """Brief white flash on the exact kill frame."""
    for kt in kill_times:
        dt = t - kt
        if 0 <= dt <= 0.05:  # 3 frames at 60fps
            alpha = 0.15 * (1.0 - dt / 0.05)
            overlay = np.full_like(frame, 255)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            return


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load kills
    kills_df = pd.read_csv(args.kills)
    kill_times = kills_df["time_s"].values
    kill_confs = kills_df["confidence"].values if "confidence" in kills_df.columns \
        else np.ones(len(kill_times))

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_duration = total / fps

    duration = args.duration if args.duration > 0 else vid_duration
    end_frame = min(total, int(duration * fps))

    print(f"Video: {w}x{h} {fps:.1f}fps ({vid_duration:.1f}s)")
    print(f"Kills: {len(kill_times)} total, "
          f"{np.sum(kill_times <= duration)} in first {duration:.0f}s")
    print(f"Rendering {end_frame} frames ({duration:.0f}s)...")

    # Output
    out_path = args.out or args.video.replace(".mp4", f"_kills_{int(duration)}s.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame_idx in range(end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        n_kills = int(np.searchsorted(kill_times, t, side="right"))

        draw_edge_glow(frame, compute_glow(t, kill_times), h, w)
        draw_kill_flash(frame, t, kill_times, h, w)
        draw_strip(frame, t, kill_times, kill_confs, n_kills, w, h)

        out.write(frame)

        if frame_idx % 300 == 0 and frame_idx > 0:
            print(f"  {frame_idx}/{end_frame} ({frame_idx/end_frame*100:.0f}%)")

    cap.release()
    out.release()
    print(f"Done. {end_frame} frames -> {out_path}")


if __name__ == "__main__":
    main()
