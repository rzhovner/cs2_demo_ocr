"""
render_audio_accuracy.py
========================
Renders gameplay video with an audio-kill-detection accuracy overlay.

Shows each detection classified as TP / FP, each missed GT kill as FN,
a scrolling timeline strip, per-detection confidence, and running F1.

Usage:
  python src/render_audio_accuracy.py data/training1_03172026.mp4 \
      --kills data/training1_03172026_audio_kills.csv \
      --gt data/training1_engagements.csv
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

# ---- matching (duplicated from audio_kill_detector to keep standalone) ------

def optimal_match(det_times, gt_times, tolerance):
    from scipy.optimize import linear_sum_assignment
    if len(det_times) == 0 or len(gt_times) == 0:
        return 0, len(det_times), len(gt_times), []
    cost = np.abs(det_times[:, None] - gt_times[None, :])
    n, m = cost.shape
    big = tolerance + 1
    size = max(n, m)
    padded = np.full((size, size), big)
    padded[:n, :m] = cost
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(padded)
    matched = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m and cost[r, c] <= tolerance:
            matched.append((r, c, cost[r, c]))
    tp = len(matched)
    fp = len(det_times) - tp
    fn = len(gt_times) - tp
    return tp, fp, fn, matched


def estimate_offset(det_times, gt_times, tol=0.5):
    best_off, best_tp = 0, 0
    for off in np.arange(-7, 7, 0.1):
        tp, _, _, _ = optimal_match(det_times - off, gt_times, tol)
        if tp > best_tp:
            best_tp = tp
            best_off = off
    for off in np.arange(best_off - 0.5, best_off + 0.5, 0.01):
        tp, _, _, _ = optimal_match(det_times - off, gt_times, tol)
        if tp > best_tp:
            best_tp = tp
            best_off = off
    return best_off


# ---- overlay constants ------------------------------------------------------

STRIP_H = 80
STRIP_PAST_S = 3.0
STRIP_FUTURE_S = 5.0
STRIP_BG = (20, 20, 20)

GLOW_RAMP_S = 0.8
GLOW_HOLD_S = 0.4
GLOW_MAX_ALPHA = 0.35
GLOW_THICKNESS = 5

# Colors (BGR)
C_TP = (80, 200, 80)       # green
C_FP = (60, 60, 235)       # red
C_FN = (0, 165, 255)       # orange
C_WHITE = (255, 255, 255)
C_SHADOW = (0, 0, 0)
C_PLAYHEAD = (200, 200, 200)
C_CONF_BAR = (185, 128, 41)  # blue
C_THRESH_LINE = (80, 80, 140)

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ---- classify detections ----------------------------------------------------

def classify_events(det_times, det_conf, gt_video_times, tolerance=0.45):
    """
    Returns:
      det_labels: list of 'tp' or 'fp' per detection
      gt_labels:  list of 'matched' or 'fn' per GT kill
      matched_pairs: list of (det_idx, gt_idx, error_s)
    """
    tp, fp, fn, matched = optimal_match(det_times, gt_video_times, tolerance)
    matched_det = set()
    matched_gt = set()
    pairs = []
    for r, c, err in matched:
        matched_det.add(r)
        matched_gt.add(c)
        pairs.append((r, c, err))

    det_labels = ["tp" if i in matched_det else "fp" for i in range(len(det_times))]
    gt_labels = ["matched" if i in matched_gt else "fn" for i in range(len(gt_video_times))]
    return det_labels, gt_labels, pairs


# ---- drawing helpers --------------------------------------------------------

def shadow_text(frame, text, pos, scale, color, thick=1):
    cv2.putText(frame, text, (pos[0]+1, pos[1]+1), FONT, scale, C_SHADOW, thick+1, cv2.LINE_AA)
    cv2.putText(frame, text, pos, FONT, scale, color, thick, cv2.LINE_AA)


def draw_stats_panel(frame, t, det_times, det_labels, gt_times, gt_labels, w):
    """Top-right panel with running TP/FP/FN/F1 stats."""
    # Count events up to time t
    tp = sum(1 for i, dt in enumerate(det_times) if dt <= t and det_labels[i] == "tp")
    fp = sum(1 for i, dt in enumerate(det_times) if dt <= t and det_labels[i] == "fp")
    fn = sum(1 for i, gt in enumerate(gt_times) if gt <= t and gt_labels[i] == "fn")
    total_gt = sum(1 for gt in gt_times if gt <= t)

    prec = tp / (tp + fp) * 100 if tp + fp else 0
    rec = tp / total_gt * 100 if total_gt else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0

    # Panel background
    panel_w, panel_h = 220, 120
    px = w - panel_w - 12
    py = 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Title
    shadow_text(frame, "AUDIO KILL DETECTOR", (px + 8, py + 18), 0.4, (160, 160, 160), 1)

    # Stats
    y = py + 40
    shadow_text(frame, f"TP: {tp}", (px + 10, y), 0.5, C_TP, 1)
    shadow_text(frame, f"FP: {fp}", (px + 80, y), 0.5, C_FP, 1)
    shadow_text(frame, f"FN: {fn}", (px + 150, y), 0.5, C_FN, 1)

    y += 28
    shadow_text(frame, f"Recall:    {rec:.1f}%", (px + 10, y), 0.45, C_WHITE, 1)
    y += 22
    shadow_text(frame, f"Precision: {prec:.1f}%", (px + 10, y), 0.45, C_WHITE, 1)
    y += 22
    # F1 in larger text, color-coded
    f1_color = C_TP if f1 >= 90 else C_FN if f1 >= 80 else C_FP
    shadow_text(frame, f"F1: {f1:.1f}%", (px + 10, y), 0.6, f1_color, 2)


def draw_timeline_strip(frame, t, det_times, det_labels, det_conf,
                        gt_times, gt_labels, w, h):
    """Bottom scrolling timeline with classified markers."""
    strip_y = h - STRIP_H
    window = STRIP_PAST_S + STRIP_FUTURE_S
    t_left = t - STRIP_PAST_S
    t_right = t + STRIP_FUTURE_S

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, strip_y), (w, h), STRIP_BG, -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Top border
    cv2.line(frame, (0, strip_y), (w, strip_y), (60, 60, 60), 1)

    graph_x0 = 10
    graph_x1 = w - 10
    graph_w = graph_x1 - graph_x0
    marker_y_det = strip_y + 30  # detection row
    marker_y_gt = strip_y + 58   # GT row
    conf_y_top = strip_y + 6
    conf_y_bot = strip_y + 22

    def time_to_x(ts):
        frac = (ts - t_left) / window
        return int(graph_x0 + frac * graph_w)

    # Row labels
    shadow_text(frame, "DET", (graph_x0, marker_y_det + 4), 0.3, (120, 120, 120), 1)
    shadow_text(frame, "GT",  (graph_x0, marker_y_gt + 4), 0.3, (120, 120, 120), 1)

    # Playhead
    ph_x = time_to_x(t)
    cv2.line(frame, (ph_x, strip_y + 2), (ph_x, h - 2), C_PLAYHEAD, 1)

    # Time ticks
    tick_start = int(t_left)
    for ts in range(tick_start, int(t_right) + 1):
        if t_left <= ts <= t_right:
            x = time_to_x(ts)
            cv2.line(frame, (x, h - 4), (x, h - 12), (70, 70, 70), 1)
            shadow_text(frame, f"{ts}s", (x - 8, h - 2), 0.25, (90, 90, 90), 1)

    # GT markers
    for i, gt in enumerate(gt_times):
        if t_left <= gt <= t_right:
            x = time_to_x(gt)
            color = C_TP if gt_labels[i] == "matched" else C_FN
            if gt_labels[i] == "fn":
                # X mark for missed
                sz = 5
                cv2.line(frame, (x-sz, marker_y_gt-sz), (x+sz, marker_y_gt+sz), C_FN, 2, cv2.LINE_AA)
                cv2.line(frame, (x+sz, marker_y_gt-sz), (x-sz, marker_y_gt+sz), C_FN, 2, cv2.LINE_AA)
            else:
                # Filled triangle for matched
                tri = np.array([[x, marker_y_gt - 6], [x - 5, marker_y_gt + 4], [x + 5, marker_y_gt + 4]], np.int32)
                cv2.fillPoly(frame, [tri], C_TP)

    # Detection markers with confidence bars
    for i, dt in enumerate(det_times):
        if t_left <= dt <= t_right:
            x = time_to_x(dt)
            color = C_TP if det_labels[i] == "tp" else C_FP

            # Confidence bar (vertical bar above marker, height proportional to confidence)
            conf = det_conf[i] if i < len(det_conf) else 1.0
            bar_h = int(min(conf - 1.0, 0.4) / 0.4 * (conf_y_bot - conf_y_top))
            if bar_h > 0:
                cv2.line(frame, (x, conf_y_bot), (x, conf_y_bot - bar_h), color, 2)

            # Circle marker
            cv2.circle(frame, (x, marker_y_det), 6, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (x, marker_y_det), 6, C_WHITE, 1, cv2.LINE_AA)

            # FP label
            if det_labels[i] == "fp":
                shadow_text(frame, "FP", (x - 6, marker_y_det - 10), 0.3, C_FP, 1)


def draw_event_flash(frame, t, det_times, det_labels, gt_times, gt_labels, h, w):
    """Edge glow when a kill event is near."""
    best_intensity = 0.0
    best_color = C_TP

    # Check detections
    for i, dt in enumerate(det_times):
        diff = t - dt
        if -GLOW_RAMP_S <= diff <= GLOW_HOLD_S:
            if diff < 0:
                intensity = 1.0 - abs(diff) / GLOW_RAMP_S
            else:
                intensity = 1.0 - diff / GLOW_HOLD_S
            if intensity > best_intensity:
                best_intensity = intensity
                best_color = C_TP if det_labels[i] == "tp" else C_FP

    # Check FN GT kills
    for i, gt in enumerate(gt_times):
        if gt_labels[i] == "fn":
            diff = t - gt
            if 0 <= diff <= GLOW_HOLD_S * 2:
                intensity = 1.0 - diff / (GLOW_HOLD_S * 2)
                if intensity > best_intensity:
                    best_intensity = intensity
                    best_color = C_FN

    if best_intensity > 0.02:
        alpha = best_intensity * GLOW_MAX_ALPHA
        tk = GLOW_THICKNESS
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (tk, h), best_color, -1)
        cv2.rectangle(overlay, (w - tk, 0), (w, h), best_color, -1)
        cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)


def draw_kill_popup(frame, t, det_times, det_labels, det_conf, w, h):
    """Brief popup text when a kill is detected."""
    for i, dt in enumerate(det_times):
        diff = t - dt
        if 0 <= diff < 1.5:
            alpha = max(0, 1.0 - diff / 1.5)
            label = det_labels[i].upper()
            conf = det_conf[i] if i < len(det_conf) else 1.0
            text = f"{label}  conf={conf:.2f}x"
            color = C_TP if det_labels[i] == "tp" else C_FP

            y_pos = int(h * 0.15 - diff * 20)  # float upward
            (tw, th), _ = cv2.getTextSize(text, FONT, 0.5, 1)
            x_pos = (w - tw) // 2

            overlay = frame.copy()
            cv2.putText(overlay, text, (x_pos, y_pos), FONT, 0.5, color, 1, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
            break  # only show most recent


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Render audio kill detection accuracy overlay")
    ap.add_argument("video", help="Source gameplay .mp4")
    ap.add_argument("--kills", help="Audio kills CSV (from audio_kill_detector.py)")
    ap.add_argument("--gt", help="Ground-truth engagements CSV")
    ap.add_argument("--tolerance", type=float, default=0.45)
    ap.add_argument("--out", help="Output video path")
    ap.add_argument("--max-seconds", type=float, default=None,
                    help="Only render first N seconds (for quick preview)")
    args = ap.parse_args()

    # Auto-derive paths
    base = args.video.replace(".mp4", "")
    kills_csv = args.kills or (base + "_audio_kills.csv")
    out_path = args.out or os.path.join("outputs", os.path.basename(base) + "_audio_accuracy.mp4")

    if not os.path.exists(kills_csv):
        print(f"ERROR: kills CSV not found: {kills_csv}")
        print(f"Run: python src/audio_kill_detector.py {args.video}")
        sys.exit(1)
    if not args.gt:
        print("ERROR: --gt (ground truth CSV) required")
        sys.exit(1)

    # Load data
    kills_df = pd.read_csv(kills_csv)
    det_times = kills_df["time_s"].values
    det_conf = kills_df["confidence"].values if "confidence" in kills_df.columns else np.ones(len(det_times))

    gt_df = pd.read_csv(args.gt)
    gt_col = "kill_time_s" if "kill_time_s" in gt_df.columns else "engagement_start_s"
    gt_demo_times = gt_df[gt_col].dropna().values

    # Estimate offset and shift GT to video time
    offset = estimate_offset(det_times, gt_demo_times)
    gt_video_times = gt_demo_times + offset
    print(f"Offset: {offset:+.2f}s (audio -> demo)")

    # Classify
    det_labels, gt_labels, pairs = classify_events(
        det_times, det_conf, gt_video_times, args.tolerance)

    tp = sum(1 for l in det_labels if l == "tp")
    fp = sum(1 for l in det_labels if l == "fp")
    fn = sum(1 for l in gt_labels if l == "fn")
    prec = tp / (tp + fp) * 100 if tp + fp else 0
    rec = tp / len(gt_demo_times) * 100
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    print(f"Classification: TP={tp} FP={fp} FN={fn}")
    print(f"  Recall={rec:.1f}%  Precision={prec:.1f}%  F1={f1:.1f}%")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open {args.video}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_f / fps

    max_frames = total_f
    if args.max_seconds:
        max_frames = min(total_f, int(args.max_seconds * fps))

    print(f"Video: {w}x{h} {fps:.1f}fps {duration:.1f}s")
    print(f"Rendering {max_frames} frames...")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps

        draw_event_flash(frame, t, det_times, det_labels, gt_video_times, gt_labels, h, w)
        draw_timeline_strip(frame, t, det_times, det_labels, det_conf,
                           gt_video_times, gt_labels, w, h)
        draw_stats_panel(frame, t, det_times, det_labels, gt_video_times, gt_labels, w)
        draw_kill_popup(frame, t, det_times, det_labels, det_conf, w, h)

        out.write(frame)

        if frame_idx % 300 == 0:
            pct = frame_idx / max_frames * 100
            print(f"  {frame_idx}/{max_frames} ({pct:.0f}%)")

    cap.release()
    out.release()
    print(f"\nDone. {frame_idx + 1} frames -> {out_path}")
    print(f"Final: TP={tp} FP={fp} FN={fn} | Recall={rec:.1f}% Precision={prec:.1f}% F1={f1:.1f}%")


if __name__ == "__main__":
    main()
