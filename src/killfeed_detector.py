"""
Kill feed detector — detects kills by tracking the CS2 kill feed UI element.

Each kill produces a red entry in the top-right kill feed. We detect:
  1. Rising edges of red pixel % in the top kill feed slot (new entry appears)
  2. Large frame deltas when the kill feed shifts (new entry pushes old ones down)

No OCR. No Tesseract. Pure OpenCV pixel analysis.

Validates against demo-parser ground truth with automatic offset estimation.
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

# ── paths ────────────────────────────────────────────────────────────────────
VIDEO_PATH = r"F:\P_rojects\rz_DemoViewer\data\training1_03172026.mp4"
GROUND_TRUTH_CSV = r"F:\P_rojects\rz_DemoViewer\data\training1_engagements.csv"
OUTPUT_REPORT = r"F:\P_rojects\rz_DemoViewer\outputs\killfeed_validation.png"

# ── ROI definitions (relative fractions) ─────────────────────────────────────
# Full kill feed area (top-right), EXCLUDING net_graph overlay at very top
KILLFEED_FULL = (0.8203, 0.045, 0.9961, 0.1458)

# Top slot only — where the newest entry appears
# Each entry is ~30px high at 960p. Top slot starts below the net_graph area.
KILLFEED_TOP_SLOT = (0.8203, 0.045, 0.9961, 0.085)

# ── detection parameters ─────────────────────────────────────────────────────
# Red detection in HSV: CS2 kill feed entries are red
RED_HUE_LO1, RED_HUE_HI1 = 0, 10      # red wraps around hue 0
RED_HUE_LO2, RED_HUE_HI2 = 170, 180
RED_SAT_MIN = 80
RED_VAL_MIN = 80

# Kill detection from red signal
RED_RISE_THRESH = 8.0       # min red% increase (frame-to-frame) to count as new entry
RED_MIN_GAP_S = 0.15        # min seconds between detected kills (AK can kill every 0.3s)

# Kill detection from delta signal (catches rapid kills when red stays high)
DELTA_THRESH = 4.5          # real kill feed shifts are 5-8; was 12 (too high, missed everything)
DELTA_MIN_GAP_S = 0.35      # AK-47 cycle + aim time means kills >0.3s apart

# Fusion: combine red-rise and delta signals
DEDUP_WINDOW_S = 0.2        # merge detections from both signals within this window

# Validation
MATCH_TOLERANCE_S = 0.5


# ── helpers ──────────────────────────────────────────────────────────────────

def resolve_roi(rel_roi, w, h):
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * w), round(y1f * h), round(x2f * w), round(y2f * h))


def red_percentage(frame_crop):
    """Percentage of pixels that are red in a BGR crop."""
    hsv = cv2.cvtColor(frame_crop, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, (RED_HUE_LO1, RED_SAT_MIN, RED_VAL_MIN),
                           (RED_HUE_HI1, 255, 255))
    m2 = cv2.inRange(hsv, (RED_HUE_LO2, RED_SAT_MIN, RED_VAL_MIN),
                           (RED_HUE_HI2, 255, 255))
    mask = m1 | m2
    return np.sum(mask > 0) / mask.size * 100


# ── scan ─────────────────────────────────────────────────────────────────────

def scan_video(video_path):
    """Scan every frame. Returns times, red_pct, delta arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    top_roi = resolve_roi(KILLFEED_TOP_SLOT, w, h)
    full_roi = resolve_roi(KILLFEED_FULL, w, h)

    print(f"Video: {w}x{h} {fps:.0f}fps {total} frames ({total/fps:.1f}s)")
    print(f"Kill feed top slot: {top_roi}")
    print(f"Kill feed full ROI: {full_roi}")

    times = []
    red_pcts = []       # red % in top slot
    deltas = []         # pixel delta in full kill feed
    prev_gray = None

    t0 = time.time()
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t = idx / fps
        x1, y1, x2, y2 = top_roi
        crop_top = frame[y1:y2, x1:x2]

        fx1, fy1, fx2, fy2 = full_roi
        crop_full = frame[fy1:fy2, fx1:fx2]

        # Red percentage in top slot
        red_pcts.append(red_percentage(crop_top))

        # Frame delta in FULL kill feed ROI
        gray_full = cv2.cvtColor(crop_full, cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev_gray is not None:
            deltas.append(float(np.mean(np.abs(gray_full - prev_gray))))
        else:
            deltas.append(0.0)
        prev_gray = gray_full

        times.append(t)
        idx += 1

    cap.release()
    elapsed = time.time() - t0
    print(f"Scanned {idx} frames in {elapsed:.1f}s ({idx/elapsed:.0f} fps)")

    return np.array(times), np.array(red_pcts), np.array(deltas), fps


# ── detection ────────────────────────────────────────────────────────────────

def detect_kills_from_red(times, red_pcts, fps):
    """Detect kills from rising edges of red content in top kill feed slot."""
    # Compute frame-to-frame red increase
    red_rise = np.diff(red_pcts, prepend=red_pcts[0])
    red_rise = np.clip(red_rise, 0, None)  # only care about increases

    min_dist = max(1, int(RED_MIN_GAP_S * fps))
    peaks, _ = find_peaks(red_rise, height=RED_RISE_THRESH, distance=min_dist)
    return times[peaks], red_rise, peaks


def detect_kills_from_delta(times, deltas, fps, red_pcts=None, red_gate_min=3.0):
    """
    Detect kills from pixel delta spikes in kill feed.
    If red_pcts provided, reject peaks where red content < red_gate_min
    (filters console text, menu changes, and other non-kill-feed deltas).
    """
    min_dist = max(1, int(DELTA_MIN_GAP_S * fps))
    peaks, _ = find_peaks(deltas, height=DELTA_THRESH, distance=min_dist)

    if red_pcts is not None and len(peaks):
        # Gate: only keep peaks where kill feed has red content (= kill entries visible)
        valid = [p for p in peaks if p < len(red_pcts) and red_pcts[p] >= red_gate_min]
        peaks = np.array(valid, dtype=int)

    return times[peaks], peaks


def merge_detections(red_times, delta_times, window_s=DEDUP_WINDOW_S):
    """
    Merge detections from red-rise and delta signals.
    Deduplicate events within window_s (keep earliest).
    """
    all_times = sorted(list(red_times) + list(delta_times))
    if not all_times:
        return np.array([])

    merged = [all_times[0]]
    for t in all_times[1:]:
        if t - merged[-1] >= window_s:
            merged.append(t)
    return np.array(merged)


# ── validation ───────────────────────────────────────────────────────────────

def estimate_offset(detected, truth, search_range=10.0, resolution=0.001):
    """
    Estimate video-demo time offset via cross-correlation of impulse trains.
    Creates dense impulse signals, cross-correlates, finds peak lag.
    Returns: (offset, n_matches_at_that_offset)
    """
    if len(detected) == 0 or len(truth) == 0:
        return 0.0, 0

    # Create impulse trains at `resolution` second bins
    t_min = min(detected.min(), truth.min()) - search_range
    t_max = max(detected.max(), truth.max()) + search_range
    n_bins = int((t_max - t_min) / resolution) + 1

    det_signal = np.zeros(n_bins)
    tru_signal = np.zeros(n_bins)

    for t in detected:
        idx = int((t - t_min) / resolution)
        if 0 <= idx < n_bins:
            det_signal[idx] = 1.0
    for t in truth:
        idx = int((t - t_min) / resolution)
        if 0 <= idx < n_bins:
            tru_signal[idx] = 1.0

    # Gaussian blur to create tolerance window (~150ms sigma)
    from scipy.ndimage import gaussian_filter1d
    sigma = 0.15 / resolution  # 150ms in bins
    det_smooth = gaussian_filter1d(det_signal, sigma)
    tru_smooth = gaussian_filter1d(tru_signal, sigma)

    # Cross-correlate (only search within range)
    max_lag = int(search_range / resolution)
    best_lag = 0
    best_corr = -1

    # Use numpy correlate in 'full' mode, then search around center
    corr = np.correlate(det_smooth, tru_smooth, mode='full')
    center = len(tru_smooth) - 1
    search_lo = center - max_lag
    search_hi = center + max_lag
    search_lo = max(0, search_lo)
    search_hi = min(len(corr) - 1, search_hi)

    best_idx = search_lo + np.argmax(corr[search_lo:search_hi + 1])
    best_lag = best_idx - center
    offset = best_lag * resolution

    # Count matches at this offset
    shifted = detected - offset
    matches = 0
    remaining = list(truth)
    for t in sorted(shifted):
        dists = [abs(t - gt) for gt in remaining]
        if dists and min(dists) <= 0.3:
            matches += 1
            remaining.pop(int(np.argmin(dists)))

    return offset, matches


def score(detected, truth, tolerance_s=MATCH_TOLERANCE_S):
    """
    Score detected kills against ground truth using optimal (closest-pair) matching.
    Sorts both arrays and matches nearest pairs, avoiding the greedy ordering bug
    where an early detection steals a truth kill from a closer later detection.
    """
    det_sorted = np.sort(detected)
    tru_sorted = np.sort(truth)

    # Build cost matrix and match greedily by smallest distance (globally)
    pairs = []
    for i, d in enumerate(det_sorted):
        for j, t in enumerate(tru_sorted):
            dist = abs(d - t)
            if dist <= tolerance_s:
                pairs.append((dist, i, j))
    pairs.sort()  # smallest distance first

    used_det = set()
    used_tru = set()
    timing_errors = []
    for dist, i, j in pairs:
        if i not in used_det and j not in used_tru:
            used_det.add(i)
            used_tru.add(j)
            timing_errors.append(det_sorted[i] - tru_sorted[j])

    matched = len(timing_errors)
    false_positives = [det_sorted[i] for i in range(len(det_sorted)) if i not in used_det]
    missed = [tru_sorted[j] for j in range(len(tru_sorted)) if j not in used_tru]

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
        "missed": missed,
        "timing_errors": timing_errors,
        "mean_timing_error_ms": np.mean(np.abs(timing_errors)) * 1000 if timing_errors else 0,
        "median_timing_error_ms": np.median(np.abs(timing_errors)) * 1000 if timing_errors else 0,
    }


# ── report ───────────────────────────────────────────────────────────────────

def plot_report(times, red_pcts, red_rise, deltas,
                red_kill_times, delta_kill_times, merged_kills,
                truth_kills_aligned, metrics, offset, out_path):
    """Generate validation report."""
    fig, axes = plt.subplots(5, 1, figsize=(18, 16), gridspec_kw={"hspace": 0.5})

    # Panel 1: Raw red percentage
    ax = axes[0]
    ax.fill_between(times, red_pcts, color="#e74c3c", alpha=0.4)
    ax.set_ylabel("Red %")
    ax.set_title("Kill Feed Top Slot — Red Pixel Content", fontsize=10, loc="left")

    # Panel 2: Red rise (first derivative) + detected peaks
    ax = axes[1]
    ax.fill_between(times, red_rise, color="#e74c3c", alpha=0.6)
    ax.axhline(RED_RISE_THRESH, color="black", linewidth=0.8, linestyle="--")
    for t in red_kill_times:
        ax.axvline(t, color="#c0392b", linewidth=0.8, alpha=0.8)
    for t in truth_kills_aligned:
        ax.axvline(t, color="#27ae60", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Red rise")
    ax.set_title(f"Red Rising Edge — {len(red_kill_times)} peaks (thresh={RED_RISE_THRESH})",
                 fontsize=10, loc="left")

    # Panel 3: Frame delta in full kill feed
    ax = axes[2]
    ax.fill_between(times, deltas, color="#8e44ad", alpha=0.5)
    ax.axhline(DELTA_THRESH, color="black", linewidth=0.8, linestyle="--")
    for t in delta_kill_times:
        ax.axvline(t, color="#6c3483", linewidth=0.8, alpha=0.8)
    for t in truth_kills_aligned:
        ax.axvline(t, color="#27ae60", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("Pixel delta")
    ax.set_title(f"Kill Feed Frame Delta — {len(delta_kill_times)} peaks (thresh={DELTA_THRESH})",
                 fontsize=10, loc="left")

    # Panel 4: Merged kills vs truth
    ax = axes[3]
    ax.set_yticks([])
    for t in truth_kills_aligned:
        ax.axvline(t, color="#27ae60", linewidth=1.5, alpha=0.7)
    for t in merged_kills:
        ax.axvline(t, color="#e74c3c", linewidth=1.5, alpha=0.7)
    for t in metrics["false_positives"]:
        ax.axvline(t + offset, color="#e74c3c", linewidth=2.5, linestyle=":", alpha=1.0)
    for t in metrics["missed"]:
        ax.axvline(t + offset, color="#f39c12", linewidth=2.5, linestyle=":", alpha=1.0)
    ax.set_title(
        f"Merged: {metrics['matched']}/{metrics['n_truth']} matched | "
        f"{len(metrics['false_positives'])} FP | {len(metrics['missed'])} missed",
        fontsize=10, loc="left")

    # Panel 5: Timing error
    ax = axes[4]
    if metrics["timing_errors"]:
        errors_ms = np.array(metrics["timing_errors"]) * 1000
        ax.hist(errors_ms, bins=30, color="#3498db", alpha=0.7, edgecolor="#2c3e50")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Timing error (ms)")
    ax.set_title(
        f"Timing: mean |error|={metrics['mean_timing_error_ms']:.0f}ms  "
        f"median={metrics['median_timing_error_ms']:.0f}ms  offset={offset:+.2f}s",
        fontsize=10, loc="left")

    for ax in axes:
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"KILL FEED DETECTOR — VALIDATION REPORT\n"
        f"Recall: {metrics['recall']:.1f}%  |  Precision: {metrics['precision']:.1f}%  |  "
        f"F1: {metrics['f1']:.1f}%  |  "
        f"Detected: {metrics['n_detected']}  |  Truth: {metrics['n_truth']}",
        fontsize=12, fontweight="bold", y=1.02)

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Report saved: {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("KILL FEED DETECTOR — Validation Run")
    print("=" * 60)

    # Load ground truth
    truth_kills = pd.read_csv(GROUND_TRUTH_CSV)["kill_time_s"].values
    print(f"Ground truth: {len(truth_kills)} kills")

    # Scan video
    times, red_pcts, deltas, fps = scan_video(VIDEO_PATH)

    # Detect from red rising edges
    red_kill_times, red_rise, red_peaks = detect_kills_from_red(times, red_pcts, fps)
    print(f"Red-rise kills:  {len(red_kill_times)}")

    # Detect from frame delta, gated by red content (rejects console/menu noise)
    delta_kill_times, delta_peaks = detect_kills_from_delta(times, deltas, fps, red_pcts=red_pcts)
    print(f"Delta kills:     {len(delta_kill_times)}")

    # Merge both signals
    merged = merge_detections(red_kill_times, delta_kill_times)
    print(f"Merged kills:    {len(merged)}")

    # Also test delta-only (it found exactly 100 in first run)
    print(f"\n--- Testing delta-only vs merged ---")

    # Show first/last 5 detections for debugging alignment
    print(f"\nFirst 10 delta detections (video time): {[f'{t:.2f}' for t in delta_kill_times[:10]]}")
    print(f"Last 5 delta detections (video time):  {[f'{t:.2f}' for t in delta_kill_times[-5:]]}")
    print(f"First 5 truth kills (demo time):       {[f'{t:.3f}' for t in truth_kills[:5]]}")
    print(f"Last 5 truth kills (demo time):        {[f'{t:.3f}' for t in truth_kills[-5:]]}")

    # Test multiple signal combos
    for label, kills in [("delta-only", delta_kill_times), ("red-only", red_kill_times), ("merged", merged)]:
        off, om = estimate_offset(kills, truth_kills)
        ds = truth_kills[0] + off - 1.0
        de = truth_kills[-1] + off + 1.0
        ir = kills[(kills >= ds) & (kills <= de)]
        al = ir - off
        m = score(al, truth_kills)
        print(f"  {label:12s}: offset={off:+.2f}s  in_range={len(ir):3d}  "
              f"matched={m['matched']:3d}  recall={m['recall']:.1f}%  prec={m['precision']:.1f}%  "
              f"F1={m['f1']:.1f}%  |err|={m['median_timing_error_ms']:.0f}ms")

    # Use best performer
    offset, offset_matches = estimate_offset(delta_kill_times, truth_kills)
    print(f"\nUsing delta-only. Offset: {offset:+.2f}s ({offset_matches} best matches)")

    # Filter to demo range (using delta-only)
    demo_start = truth_kills[0] + offset - 1.0
    demo_end = truth_kills[-1] + offset + 1.0
    in_range = delta_kill_times[(delta_kill_times >= demo_start) & (delta_kill_times <= demo_end)]
    out_range = delta_kill_times[(delta_kill_times < demo_start) | (delta_kill_times > demo_end)]
    print(f"In demo range: {len(in_range)}  Outside: {len(out_range)}")

    # Score
    aligned = in_range - offset
    metrics = score(aligned, truth_kills)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Ground truth       : {metrics['n_truth']}")
    print(f"  Detected (in range): {metrics['n_detected']}")
    print(f"  Matched            : {metrics['matched']}")
    print(f"  Recall             : {metrics['recall']:.1f}%")
    print(f"  Precision          : {metrics['precision']:.1f}%")
    print(f"  F1                 : {metrics['f1']:.1f}%")
    print(f"  False positives    : {len(metrics['false_positives'])}")
    print(f"  Missed kills       : {len(metrics['missed'])}")
    print(f"  Mean |timing err|  : {metrics['mean_timing_error_ms']:.0f}ms")
    print(f"  Median |timing err|: {metrics['median_timing_error_ms']:.0f}ms")

    # Diagnosis
    if metrics["false_positives"]:
        print(f"\n  FP diagnosis:")
        for fp in metrics["false_positives"][:10]:
            dists = [abs(fp - gt) for gt in truth_kills]
            print(f"    det={fp:.2f}s  nearest_truth={truth_kills[np.argmin(dists)]:.2f}s  gap={min(dists):.3f}s")
    if metrics["missed"]:
        print(f"\n  Missed diagnosis:")
        for ms in metrics["missed"][:10]:
            if len(aligned):
                dists = [abs(ms - d) for d in aligned]
                print(f"    truth={ms:.2f}s  nearest_det={aligned[np.argmin(dists)]:.2f}s  gap={min(dists):.3f}s")
    print(f"{'='*60}")

    # Report
    truth_aligned_for_plot = truth_kills + offset
    plot_report(times, red_pcts, red_rise, deltas,
                red_kill_times, delta_kill_times, merged,
                truth_aligned_for_plot, metrics, offset, OUTPUT_REPORT)

    if metrics["recall"] >= 95 and metrics["precision"] >= 95:
        print("\n>>> PASS: >=95% recall AND precision")
        return 0
    else:
        print(f"\n>>> FAIL: needs tuning")
        return 1


if __name__ == "__main__":
    sys.exit(main())
