"""
render_kill_overlay.py  (v2 — anticipation-first)
==================================================
Renders gameplay video with a scrolling signal strip that lets you
ANTICIPATE kill events before they happen, then verify alignment
between the bot-count OCR detection and demo-parser ground truth.

Design principles:
  - Raw gameplay is sacred — never occlude center screen
  - Anticipation > reaction: kills visible 4s ahead in the strip
  - Associative UX: the source ROI is shown live in the strip inset

Overlay elements:
  0. ROI highlight — thin box on the actual bot-count HUD digits
  1. Signal strip  — scrolling bot-count step function + live ROI inset
  2. Edge glow     — subtle ramp as kills approach, fade after
  3. Corner tally  — small kill counter, top-right

Usage:
  python src/render_kill_overlay.py
  python src/render_kill_overlay.py --offset 1.64
"""

import argparse
import cv2
import numpy as np
import pandas as pd

# ── defaults ─────────────────────────────────────────────────────────────────
VIDEO_PATH = r"F:\P_rojects\rz_DemoViewer\Counter-Strike 2 - 2026-03-17 4-28-50 PM.mp4"
DET_CSV    = r"F:\P_rojects\rz_DemoViewer\Counter-Strike 2 - 2026-03-17 4-28-50 PM_kill_times_count.csv"
OCR_CSV    = r"F:\P_rojects\rz_DemoViewer\Counter-Strike 2 - 2026-03-17 4-28-50 PM_ocr_dump.csv"
GT_CSV     = r"F:\P_rojects\rz_DemoViewer\data\training1_engagements.csv"
OFFSET     = 1.64
OUT_PATH   = r"F:\P_rojects\rz_DemoViewer\outputs\kill_overlay_v2.mp4"

# Bot count ROI (relative coords — from kill_flash_detector.py)
ROI_BOT_COUNT_REL = (0.5063, 0.0604, 0.5234, 0.0771)

# Signal strip
STRIP_H       = 90        # px height of bottom strip
STRIP_PAST_S  = 2.0       # seconds of history visible
STRIP_FUTURE_S = 4.0      # seconds of lookahead
STRIP_BG      = (26, 26, 26)
STRIP_BG_ALPHA = 0.70
Y_MIN, Y_MAX  = 9.5, 12.5 # bot count axis range (tight = visible steps)

# Edge glow
GLOW_RAMP_S   = 1.5       # seconds before kill: ramp up
GLOW_FADE_S   = 0.3       # seconds after kill: fade out
GLOW_MAX_ALPHA = 0.40
GLOW_THICKNESS = 4

# Colors (BGR)
C_STEP    = (185, 128, 41)   # blue step line
C_DET     = (60, 60, 235)    # red — detected kills
C_GT      = (80, 200, 80)    # green — ground truth
C_AMBER   = (0, 190, 245)    # amber — approach glow
C_WHITE   = (255, 255, 255)
C_SHADOW  = (0, 0, 0)
C_PLAYHEAD = (220, 220, 220)
C_ROI_BOX = (0, 190, 245)    # amber highlight on HUD

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Render kill-overlay video (v2)")
    p.add_argument("--video",  default=VIDEO_PATH)
    p.add_argument("--det",    default=DET_CSV)
    p.add_argument("--ocr",    default=OCR_CSV)
    p.add_argument("--gt",     default=GT_CSV)
    p.add_argument("--offset", type=float, default=OFFSET)
    p.add_argument("--out",    default=OUT_PATH)
    return p.parse_args()


def resolve_roi(rel_roi, frame_w, frame_h):
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * frame_w), round(y1f * frame_h),
            round(x2f * frame_w), round(y2f * frame_h))


# ── Element 0: ROI highlight ────────────────────────────────────────────────

def draw_roi_highlight(frame, roi, pulse):
    """Thin box on the actual bot-count digits in the game HUD.
    pulse: 0.0 (dim) → 1.0 (bright) when value just changed."""
    x1, y1, x2, y2 = roi
    pad = 3
    brightness = 0.35 + 0.65 * pulse
    color = tuple(int(c * brightness) for c in C_ROI_BOX)
    thickness = 2 if pulse > 0.3 else 1
    cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), color, thickness)


# ── Element 1: Signal strip with source inset ───────────────────────────────

def draw_signal_strip(frame, t, ocr_times, ocr_stable, det_times, gt_video,
                      cur_stable, w, h):
    """
    Bottom-edge strip: current value readout (left) + scrolling step function (right).
    Kill transitions marked with red dots on the step line, GT with green triangles below.
    """
    strip_y = h - STRIP_H
    window = STRIP_PAST_S + STRIP_FUTURE_S

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, strip_y), (w, h), STRIP_BG, -1)
    cv2.addWeighted(overlay, STRIP_BG_ALPHA, frame, 1.0 - STRIP_BG_ALPHA, 0, frame)

    # ── Left panel: current stabilized value as large text ──
    panel_w = 70
    panel_x, panel_y = 6, strip_y + 4
    panel_h = STRIP_H - 8

    # Value text — large and unambiguous
    val_str = str(int(cur_stable)) if cur_stable is not None and not np.isnan(cur_stable) else "?"
    val_color = C_WHITE
    cv2.putText(frame, val_str, (panel_x + 8, panel_y + panel_h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, C_SHADOW, 4, cv2.LINE_AA)
    cv2.putText(frame, val_str, (panel_x + 8, panel_y + panel_h - 14),
                cv2.FONT_HERSHEY_DUPLEX, 1.8, val_color, 2, cv2.LINE_AA)
    # Small label
    cv2.putText(frame, "bots", (panel_x + 10, panel_y + 14),
                FONT, 0.35, (140, 140, 140), 1, cv2.LINE_AA)

    # Separator line
    sep_x = panel_x + panel_w
    cv2.line(frame, (sep_x, strip_y + 4), (sep_x, h - 4), (80, 80, 80), 1)

    # ── Step function area ──
    graph_x0 = sep_x + 8
    graph_x1 = w - 10
    graph_w = graph_x1 - graph_x0
    graph_y0 = strip_y + 8
    graph_y1 = h - 8
    graph_h = graph_y1 - graph_y0

    t_left  = t - STRIP_PAST_S
    t_right = t + STRIP_FUTURE_S

    def time_to_x(ts):
        frac = (ts - t_left) / window
        return int(graph_x0 + frac * graph_w)

    def val_to_y(v):
        frac = (v - Y_MIN) / (Y_MAX - Y_MIN)
        return int(graph_y1 - frac * graph_h)

    # Faint Y-axis grid lines for 10, 11, 12
    for v in [10, 11, 12]:
        y = val_to_y(v)
        cv2.line(frame, (graph_x0, y), (graph_x1, y), (50, 50, 50), 1)
        cv2.putText(frame, str(v), (graph_x1 + 2, y + 4),
                    FONT, 0.3, (90, 90, 90), 1, cv2.LINE_AA)

    # Build step segments: collapse consecutive same-value samples
    mask = (ocr_times >= t_left - 0.1) & (ocr_times <= t_right + 0.1)
    vis_times = ocr_times[mask]
    vis_vals  = ocr_stable[mask]

    # Collapse into (start_t, end_t, value) segments
    segments = []
    if len(vis_times) > 0:
        seg_start = vis_times[0]
        seg_val = vis_vals[0]
        for i in range(1, len(vis_times)):
            if vis_vals[i] != seg_val:
                segments.append((seg_start, vis_times[i], seg_val))
                seg_start = vis_times[i]
                seg_val = vis_vals[i]
        segments.append((seg_start, vis_times[-1], seg_val))

    # Draw collapsed step function — clear horizontal bars with vertical transitions
    for i, (t0, t1, v) in enumerate(segments):
        x0 = max(graph_x0, time_to_x(t0))
        x1 = min(graph_x1, time_to_x(t1))
        y = val_to_y(v)
        # Horizontal bar
        cv2.line(frame, (x0, y), (x1, y), C_STEP, 3, cv2.LINE_AA)
        # Vertical connector to next segment
        if i + 1 < len(segments):
            next_y = val_to_y(segments[i + 1][2])
            cv2.line(frame, (x1, y), (x1, next_y), C_STEP, 2, cv2.LINE_AA)

    # Playhead (draw before markers so markers are on top)
    ph_x = time_to_x(t)
    cv2.line(frame, (ph_x, strip_y + 2), (ph_x, h - 2), C_PLAYHEAD, 1)

    # GT kill markers (green triangles, drawn ABOVE step line)
    for gt in gt_video:
        if t_left <= gt <= t_right:
            x = time_to_x(gt)
            idx = np.searchsorted(ocr_times, gt, side="right") - 1
            idx = max(0, min(idx, len(ocr_stable) - 1))
            y = val_to_y(ocr_stable[idx])
            # Small downward triangle above the step line
            tri = np.array([[x, y - 5], [x - 4, y - 13], [x + 4, y - 13]], np.int32)
            cv2.fillPoly(frame, [tri], C_GT)

    # Detected kill markers (red circles AT the step transition)
    for dt in det_times:
        if t_left <= dt <= t_right:
            x = time_to_x(dt)
            idx = np.searchsorted(ocr_times, dt, side="right") - 1
            idx = max(0, min(idx, len(ocr_stable) - 1))
            y = val_to_y(ocr_stable[idx])
            cv2.circle(frame, (x, y), 5, C_DET, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, y), 5, (255, 255, 255), 1, cv2.LINE_AA)

    # Top border
    cv2.line(frame, (0, strip_y), (w, strip_y), (60, 60, 60), 1)


# ── Element 2: Edge glow ────────────────────────────────────────────────────

def compute_approach_intensity(t, kill_times):
    """0.0–1.0 intensity based on proximity to nearest kill.
    Ramps up GLOW_RAMP_S before, fades GLOW_FADE_S after."""
    best = 0.0
    for kt in kill_times:
        dt = t - kt
        if -GLOW_RAMP_S <= dt < 0:
            # Approaching: ramp up
            intensity = 1.0 - abs(dt) / GLOW_RAMP_S
        elif 0 <= dt <= GLOW_FADE_S:
            # Just happened: fade out
            intensity = 1.0 - dt / GLOW_FADE_S
        else:
            continue
        best = max(best, intensity)
    return best


def draw_edge_glow(frame, intensity, color, h, w):
    """Thin left+right edge highlight."""
    if intensity <= 0.02:
        return
    alpha = intensity * GLOW_MAX_ALPHA
    t = GLOW_THICKNESS
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (t, h), color, -1)
    cv2.rectangle(overlay, (w - t, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


# ── Element 3: Corner tally ─────────────────────────────────────────────────

def draw_tally(frame, det_n, gt_n, w):
    """Kill counter, top-right corner."""
    scale = 0.55
    thick = 1
    det_text = f"DET {det_n}"
    gt_text = f"GT {gt_n}"
    (dtw, dth), _ = cv2.getTextSize(det_text, FONT, scale, thick)
    (gtw, gth), _ = cv2.getTextSize(gt_text, FONT, scale, thick)
    total_w = dtw + gtw + 16
    ox = w - total_w - 14
    oy = 24
    # Pill background
    overlay = frame.copy()
    cv2.rectangle(overlay, (ox - 8, oy - dth - 6), (ox + total_w + 4, oy + 8),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    # Text
    cv2.putText(frame, det_text, (ox, oy), FONT, scale, C_DET, thick, cv2.LINE_AA)
    cv2.putText(frame, gt_text, (ox + dtw + 16, oy), FONT, scale, C_GT, thick, cv2.LINE_AA)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Load data
    det_df = pd.read_csv(args.det)
    det_times = det_df["video_s"].values

    gt_df = pd.read_csv(args.gt)
    gt_video = gt_df["kill_time_s"].values + args.offset

    ocr_df = pd.read_csv(args.ocr)
    ocr_times  = ocr_df["time_s"].values
    ocr_stable = ocr_df["bot_count_stable"].values.astype(float)

    print(f"Detected kills: {len(det_times)}")
    print(f"GT kills:       {len(gt_video)}")
    print(f"OCR samples:    {len(ocr_times)}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        return

    fps     = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_f / fps

    # Resolve bot count ROI
    roi = resolve_roi(ROI_BOT_COUNT_REL, w, h)

    print(f"Video: {w}x{h} {fps:.1f}fps {total_f} frames ({duration:.1f}s)")
    print(f"Bot count ROI: {roi}")

    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))
    if not out.isOpened():
        print(f"ERROR: cannot open output {args.out}")
        return

    # Track previous stable value for ROI pulse
    prev_stable = None

    print("Rendering...")
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        # Look up current stabilized value
        ocr_idx = np.searchsorted(ocr_times, t, side="right") - 1
        ocr_idx = max(0, min(ocr_idx, len(ocr_times) - 1))
        cur_stable = ocr_stable[ocr_idx]

        # ROI pulse: bright when value just changed
        pulse = 0.0
        if prev_stable is not None and cur_stable != prev_stable:
            pulse = 1.0
        elif hasattr(main, '_pulse_decay'):
            pulse = max(0, main._pulse_decay - 0.08)
        main._pulse_decay = pulse
        prev_stable = cur_stable

        # Kill counters
        det_n = int(np.searchsorted(det_times, t, side="right"))
        gt_n  = int(np.searchsorted(gt_video, t, side="right"))

        # Approach intensity (use detected kills for anticipation)
        approach = compute_approach_intensity(t, det_times)

        # ── Draw overlays ──
        draw_roi_highlight(frame, roi, pulse)
        draw_edge_glow(frame, approach, C_AMBER, h, w)
        draw_signal_strip(frame, t, ocr_times, ocr_stable,
                          det_times, gt_video, cur_stable, w, h)
        draw_tally(frame, det_n, gt_n, w)

        out.write(frame)
        frame_idx += 1

        if frame_idx % 300 == 0:
            pct = frame_idx / total_f * 100
            print(f"  {frame_idx}/{total_f} ({pct:.0f}%)")

    cap.release()
    out.release()
    print(f"Done. {frame_idx} frames written to {args.out}")


if __name__ == "__main__":
    main()
