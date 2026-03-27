"""
Pixel-diff kill detector — NO OCR, NO Tesseract.

Detects kills from CS2 AimBotz video using two fused signals:
  Signal A: Brightness spike in kill-flash HUD region (center icon)
  Signal B: Pixel-delta spike in bot-count HUD region (digits change on kill)

Fusion rule: A spike in BOTH signals within a confirmation window = kill.

Validates against demo-parser ground truth and outputs a report image.

Usage:
    python pixel_diff_detector.py
"""

import sys
import time
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ── paths (training1 pair: video + demo ground truth) ────────────────────────
VIDEO_PATH = r"F:\P_rojects\rz_DemoViewer\data\training1_03172026.mp4"
GROUND_TRUTH_CSV = r"F:\P_rojects\rz_DemoViewer\data\training1_engagements.csv"
OUTPUT_REPORT = r"F:\P_rojects\rz_DemoViewer\outputs\pixeldiff_validation.png"

# ── ROI definitions (relative fractions, resolution-independent) ─────────────
# Brightness ROIs — track mean HSV-V for kill flash detection
BRIGHTNESS_ROIS = {
    "center_icon":      (0.4727, 0.9115, 0.5273, 0.9771),
    "center_icon_wide": (0.4453, 0.9010, 0.5547, 0.9896),
    "kill_counter":     (0.2773, 0.9427, 0.3516, 0.9771),
}

# Delta ROIs — track pixel change for state-change detection
DELTA_ROIS = {
    "bot_count": (0.5063, 0.0604, 0.5234, 0.0771),
    "ammo_mag":  (0.6094, 0.9375, 0.6836, 0.9771),
}

# ── detection parameters ─────────────────────────────────────────────────────
SUBSAMPLE = 2              # read every Nth frame for brightness; delta every frame

# Signal A: kill_counter brightness is the most reliable single ROI
PRIMARY_ROI = "kill_counter"
PEAK_PROMINENCE = 5.0       # min brightness delta rise
PEAK_MIN_GAP_S = 0.3        # min seconds between kill peaks

# Signal B: bot-count pixel delta (confirmation filter)
BOT_DELTA_THRESH = 5.0      # min pixel delta to count as digit change
BOT_DELTA_MIN_GAP_S = 0.2   # min seconds between bot-count changes

# Fusion
FUSION_WINDOW_S = 0.4       # max time between brightness and bot-delta to confirm
REQUIRE_BOT_DELTA = False    # test kill_counter alone first

# Validation
MATCH_TOLERANCE_S = 0.5      # max time offset to match detected vs ground truth kill


# ── helpers ──────────────────────────────────────────────────────────────────

def resolve_roi(rel_roi, w, h):
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * w), round(y1f * h), round(x2f * w), round(y2f * h))


def mean_brightness(frame, x1, y1, x2, y2):
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]))


def gray_patch(frame, x1, y1, x2, y2):
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)


# ── scan ─────────────────────────────────────────────────────────────────────

def scan_video(video_path):
    """Single-pass video scan. Returns times, brightness dict, delta dict."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    bright_abs = {name: resolve_roi(roi, w, h) for name, roi in BRIGHTNESS_ROIS.items()}
    delta_abs = {name: resolve_roi(roi, w, h) for name, roi in DELTA_ROIS.items()}

    print(f"Video: {w}x{h} {fps:.0f}fps {total} frames ({total/fps:.1f}s)")
    print(f"Subsample: {SUBSAMPLE} -> {fps/SUBSAMPLE:.0f} samples/s")

    times = []
    brightness = {name: [] for name in BRIGHTNESS_ROIS}
    delta_buf = {name: [] for name in DELTA_ROIS}
    shot_delta = {name: [] for name in DELTA_ROIS}
    prev_gray = {name: None for name in DELTA_ROIS}

    t0 = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Delta: every frame
        for name, (x1, y1, x2, y2) in delta_abs.items():
            g = gray_patch(frame, x1, y1, x2, y2)
            if prev_gray[name] is not None:
                delta_buf[name].append(float(np.mean(np.abs(g - prev_gray[name]))))
            prev_gray[name] = g

        # Subsample: brightness + flush delta buckets
        if idx % SUBSAMPLE == 0:
            times.append(idx / fps)
            for name, (x1, y1, x2, y2) in bright_abs.items():
                brightness[name].append(mean_brightness(frame, x1, y1, x2, y2))
            for name in DELTA_ROIS:
                shot_delta[name].append(max(delta_buf[name]) if delta_buf[name] else 0.0)
                delta_buf[name] = []

        idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"Scanned {idx} frames in {elapsed:.1f}s ({idx/elapsed:.0f} fps)")

    return (np.array(times),
            {k: np.array(v) for k, v in brightness.items()},
            {k: np.array(v) for k, v in shot_delta.items()})


# ── detection ────────────────────────────────────────────────────────────────

def detect_brightness_peaks_single(times, brightness_arr, fps_eff):
    """Detect kill-flash peaks from brightness first-difference for one ROI."""
    delta = np.diff(brightness_arr, prepend=brightness_arr[0])
    delta_pos = np.clip(delta, 0, None)
    peaks, props = find_peaks(
        delta_pos,
        prominence=PEAK_PROMINENCE,
        distance=max(1, int(PEAK_MIN_GAP_S * fps_eff))
    )
    return times[peaks], delta_pos, peaks


def detect_brightness_voted(times, brightness_dict, fps_eff):
    """
    Multi-ROI voting: detect peaks in each ROI, then cluster peaks across ROIs.
    A kill event needs >= BRIGHT_VOTE_THRESHOLD ROIs to spike within VOTE_WINDOW_S.
    Returns voted kill times (mean of contributing peak times).
    """
    # Get peaks per ROI
    all_peaks = {}
    for roi_name in BRIGHT_VOTE_ROIS:
        if roi_name not in brightness_dict:
            continue
        peak_times, _, _ = detect_brightness_peaks_single(
            times, brightness_dict[roi_name], fps_eff)
        all_peaks[roi_name] = peak_times
        print(f"  ROI {roi_name}: {len(peak_times)} peaks")

    # Collect all peak times with their ROI label
    events = []
    for roi_name, ptimes in all_peaks.items():
        for t in ptimes:
            events.append((t, roi_name))
    events.sort(key=lambda x: x[0])

    # Cluster: greedily group peaks within VOTE_WINDOW_S
    voted_kills = []
    i = 0
    while i < len(events):
        cluster_start = events[i][0]
        cluster_rois = {events[i][1]}
        cluster_times = [events[i][0]]
        j = i + 1
        while j < len(events) and events[j][0] - cluster_start <= VOTE_WINDOW_S:
            cluster_rois.add(events[j][1])
            cluster_times.append(events[j][0])
            j += 1
        if len(cluster_rois) >= BRIGHT_VOTE_THRESHOLD:
            voted_kills.append(np.mean(cluster_times))
        i = j if j > i + 1 else i + 1

    # Enforce minimum gap between voted kills
    if voted_kills:
        filtered = [voted_kills[0]]
        for t in voted_kills[1:]:
            if t - filtered[-1] >= PEAK_MIN_GAP_S:
                filtered.append(t)
        voted_kills = filtered

    return np.array(voted_kills)


def detect_bot_delta_peaks(times, delta_arr, fps_eff):
    """Detect bot-count digit changes from pixel delta."""
    peaks, _ = find_peaks(
        delta_arr,
        height=BOT_DELTA_THRESH,
        distance=max(1, int(BOT_DELTA_MIN_GAP_S * fps_eff))
    )
    return times[peaks], peaks


def fuse_kills(bright_times, bot_delta_times, window_s=FUSION_WINDOW_S):
    """
    Confirm brightness-voted kills with bot-delta support.
    A kill needs a bot-delta peak within `window_s` of the brightness vote.
    Returns confirmed kill timestamps.
    """
    kills = []
    used_bot = set()

    for bt in bright_times:
        for j, dt in enumerate(bot_delta_times):
            if j in used_bot:
                continue
            if abs(bt - dt) <= window_s:
                kills.append(bt)
                used_bot.add(j)
                break

    return np.array(kills)


# ── validation ───────────────────────────────────────────────────────────────

def load_ground_truth(csv_path, time_col="kill_time_s"):
    """Load kill timestamps from demo engagement CSV."""
    df = pd.read_csv(csv_path)
    return df[time_col].values


def estimate_offset(detected, truth, search_range=5.0, step=0.01):
    """
    Estimate the systematic time offset between detected and truth timestamps
    by maximizing the number of matches across a range of offsets.
    Returns: best_offset (detected = truth + offset)
    """
    best_offset = 0.0
    best_matches = 0
    for offset in np.arange(-search_range, search_range, step):
        shifted = detected - offset
        matches = 0
        remaining = list(truth)
        for t in sorted(shifted):
            dists = [abs(t - gt) for gt in remaining]
            if dists and min(dists) <= 0.3:
                matches += 1
                remaining.pop(int(np.argmin(dists)))
        if matches > best_matches:
            best_matches = matches
            best_offset = offset
    return best_offset, best_matches


def score(detected, truth, tolerance_s=MATCH_TOLERANCE_S):
    """Score detected kills against ground truth. Returns dict of metrics."""
    matched = 0
    timing_errors = []
    unmatched_truth = list(truth)
    false_positives = []

    for det_t in detected:
        dists = [abs(det_t - gt) for gt in unmatched_truth]
        if dists and min(dists) <= tolerance_s:
            best = int(np.argmin(dists))
            timing_errors.append(det_t - unmatched_truth[best])
            matched += 1
            unmatched_truth.pop(best)
        else:
            false_positives.append(det_t)

    n_truth = len(truth)
    n_det = len(detected)
    recall = matched / n_truth * 100 if n_truth else 0
    precision = matched / n_det * 100 if n_det else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0

    return {
        "n_truth": n_truth,
        "n_detected": n_det,
        "matched": matched,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "false_positives": false_positives,
        "missed": unmatched_truth,
        "timing_errors": timing_errors,
        "mean_timing_error_ms": np.mean(np.abs(timing_errors)) * 1000 if timing_errors else 0,
        "median_timing_error_ms": np.median(np.abs(timing_errors)) * 1000 if timing_errors else 0,
    }


# ── report ───────────────────────────────────────────────────────────────────

def plot_report(times, brightness, delta, bright_peaks_t, bot_peaks_t,
                fused_kills, truth_kills, metrics, out_path):
    """Generate a validation report image."""
    fig, axes = plt.subplots(4, 1, figsize=(18, 14), gridspec_kw={"hspace": 0.45})

    bright_arr = brightness[PRIMARY_ROI]
    bot_delta_arr = delta["bot_count"]
    bright_delta = np.clip(np.diff(bright_arr, prepend=bright_arr[0]), 0, None)

    # Panel 1: Brightness signal + peaks
    ax = axes[0]
    ax.fill_between(times, bright_delta, color="#e67e22", alpha=0.6, linewidth=0)
    for t in bright_peaks_t:
        ax.axvline(t, color="#e74c3c", linewidth=0.8, alpha=0.8)
    for t in truth_kills:
        ax.axvline(t, color="#27ae60", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(f"Signal A: Brightness delta ({PRIMARY_ROI}) — {len(bright_peaks_t)} peaks",
                 fontsize=10, loc="left")
    ax.set_ylabel("delta V")

    # Panel 2: Bot-count pixel delta + peaks
    ax = axes[1]
    ax.fill_between(times[:len(bot_delta_arr)], bot_delta_arr, color="#8e44ad", alpha=0.5)
    ax.axhline(BOT_DELTA_THRESH, color="#6c3483", linewidth=0.8, linestyle="--")
    for t in bot_peaks_t:
        ax.axvline(t, color="#e74c3c", linewidth=0.8, alpha=0.8)
    for t in truth_kills:
        ax.axvline(t, color="#27ae60", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title(f"Signal B: Bot-count pixel delta — {len(bot_peaks_t)} peaks (thresh={BOT_DELTA_THRESH})",
                 fontsize=10, loc="left")
    ax.set_ylabel("px delta")

    # Panel 3: Fused kills vs ground truth timeline
    ax = axes[2]
    ax.set_yticks([])
    for i, t in enumerate(truth_kills):
        ax.axvline(t, color="#27ae60", linewidth=1.5, alpha=0.7)
    for i, t in enumerate(fused_kills):
        ax.axvline(t, color="#e74c3c", linewidth=1.5, alpha=0.7)
    # Mark false positives
    for t in metrics["false_positives"]:
        ax.axvline(t, color="#e74c3c", linewidth=2.5, alpha=1.0, linestyle=":")
    # Mark misses
    for t in metrics["missed"]:
        ax.axvline(t, color="#f39c12", linewidth=2.5, alpha=1.0, linestyle=":")
    ax.set_title(
        f"Fused: {metrics['matched']}/{metrics['n_truth']} matched | "
        f"{len(metrics['false_positives'])} FP | {len(metrics['missed'])} missed | "
        f"green=truth red=detected orange-dotted=missed red-dotted=FP",
        fontsize=10, loc="left")

    # Panel 4: Timing error histogram
    ax = axes[3]
    if metrics["timing_errors"]:
        errors_ms = np.array(metrics["timing_errors"]) * 1000
        ax.hist(errors_ms, bins=30, color="#3498db", alpha=0.7, edgecolor="#2c3e50")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Timing error (ms) — negative=early, positive=late")
        ax.set_ylabel("Count")
    ax.set_title(
        f"Timing: mean={metrics['mean_timing_error_ms']:.0f}ms | "
        f"median={metrics['median_timing_error_ms']:.0f}ms",
        fontsize=10, loc="left")

    # Scorecard
    fig.suptitle(
        f"PIXEL-DIFF KILL DETECTOR — VALIDATION REPORT\n"
        f"Recall: {metrics['recall']:.1f}%  |  Precision: {metrics['precision']:.1f}%  |  "
        f"F1: {metrics['f1']:.1f}%  |  "
        f"Detected: {metrics['n_detected']}  |  Truth: {metrics['n_truth']}  |  "
        f"Tolerance: {MATCH_TOLERANCE_S}s",
        fontsize=12, fontweight="bold", y=1.02)

    for ax in axes:
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(labelsize=7)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Report saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("PIXEL-DIFF KILL DETECTOR — Validation Run")
    print("=" * 60)

    # Load ground truth
    truth_kills = load_ground_truth(GROUND_TRUTH_CSV)
    print(f"Ground truth: {len(truth_kills)} kills from {GROUND_TRUTH_CSV}")

    # Scan video (NO OCR)
    times, brightness, delta = scan_video(VIDEO_PATH)
    fps_eff = 1.0 / (times[1] - times[0]) if len(times) > 1 else 30.0
    print(f"Effective sample rate: {fps_eff:.1f} Hz")

    # Detect Signal A: primary ROI brightness peaks
    print(f"Signal A ({PRIMARY_ROI} brightness):")
    for roi_name in BRIGHTNESS_ROIS:
        pts, _, _ = detect_brightness_peaks_single(times, brightness[roi_name], fps_eff)
        print(f"  {roi_name}: {len(pts)} peaks")

    bright_peaks_t, bright_delta_arr, bright_peaks_idx = detect_brightness_peaks_single(
        times, brightness[PRIMARY_ROI], fps_eff)
    print(f"  Using {PRIMARY_ROI}: {len(bright_peaks_t)} peaks")

    # Detect Signal B: bot-count pixel delta
    bot_peaks_t, bot_peaks_idx = detect_bot_delta_peaks(
        times, delta["bot_count"], fps_eff)
    print(f"Signal B (bot delta):  {len(bot_peaks_t)} peaks")

    # Fuse or use brightness directly
    if REQUIRE_BOT_DELTA:
        fused_kills = fuse_kills(bright_peaks_t, bot_peaks_t)
        print(f"Fused kills ({PRIMARY_ROI} + bot-delta): {len(fused_kills)}")
    else:
        fused_kills = bright_peaks_t
        print(f"Using {PRIMARY_ROI} directly: {len(fused_kills)}")

    # Estimate video-demo offset
    offset, offset_matches = estimate_offset(fused_kills, truth_kills)
    print(f"\nEstimated video-demo offset: {offset:+.2f}s ({offset_matches} matches at that offset)")

    # Filter detected kills to the demo's kill range (with offset)
    demo_start = truth_kills[0] + offset - 1.0   # 1s grace before first kill
    demo_end = truth_kills[-1] + offset + 1.0     # 1s grace after last kill
    in_range = fused_kills[(fused_kills >= demo_start) & (fused_kills <= demo_end)]
    out_range = fused_kills[(fused_kills < demo_start) | (fused_kills > demo_end)]
    print(f"Detected in demo range [{demo_start:.1f}s, {demo_end:.1f}s]: {len(in_range)}")
    if len(out_range):
        print(f"Detected OUTSIDE demo range (pre/post challenge): {len(out_range)} — "
              f"times: {[f'{t:.2f}' for t in out_range]}")

    # Apply offset and score only in-range detections
    fused_kills_aligned = in_range - offset
    metrics = score(fused_kills_aligned, truth_kills)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Ground truth kills : {metrics['n_truth']}")
    print(f"  Detected kills     : {metrics['n_detected']}")
    print(f"  Matched            : {metrics['matched']}")
    print(f"  Recall             : {metrics['recall']:.1f}%")
    print(f"  Precision          : {metrics['precision']:.1f}%")
    print(f"  F1                 : {metrics['f1']:.1f}%")
    print(f"  False positives    : {len(metrics['false_positives'])}")
    print(f"  Missed kills       : {len(metrics['missed'])}")
    print(f"  Mean timing error  : {metrics['mean_timing_error_ms']:.0f}ms")
    print(f"  Median timing error: {metrics['median_timing_error_ms']:.0f}ms")
    if metrics["false_positives"]:
        print(f"  FP times: {[f'{t:.2f}' for t in metrics['false_positives']]}")
    if metrics["missed"]:
        print(f"  Missed times: {[f'{t:.2f}' for t in metrics['missed'][:10]]}")
    # Diagnostic: for each FP, show nearest truth; for each miss, show nearest detection
    if metrics["false_positives"]:
        print(f"\n  FP diagnosis (nearest truth kill):")
        for fp in metrics["false_positives"][:15]:
            dists = [abs(fp - gt) for gt in truth_kills]
            nearest = truth_kills[np.argmin(dists)]
            print(f"    det={fp:.2f}s  nearest_truth={nearest:.2f}s  gap={min(dists):.3f}s")
    if metrics["missed"]:
        print(f"\n  Missed diagnosis (nearest detection):")
        for ms in metrics["missed"][:15]:
            if len(fused_kills_aligned):
                dists = [abs(ms - d) for d in fused_kills_aligned]
                nearest = fused_kills_aligned[np.argmin(dists)]
                print(f"    truth={ms:.2f}s  nearest_det={nearest:.2f}s  gap={min(dists):.3f}s")
    print(f"{'='*60}")

    # Generate report
    plot_report(times, brightness, delta,
                bright_peaks_t, bot_peaks_t,
                fused_kills, truth_kills, metrics, OUTPUT_REPORT)

    # Pass/fail gate
    if metrics["recall"] >= 95 and metrics["precision"] >= 95:
        print("\n>>> PASS: >=95% recall AND precision")
        return 0
    else:
        print("\n>>> FAIL: below 95% threshold — needs tuning")
        return 1


if __name__ == "__main__":
    sys.exit(main())
