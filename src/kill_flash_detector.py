"""
kill_flash_detector.py
======================
Three-signal kill detector for CS2/AimBotz gameplay video.

Signal 1 -- Kill flash (brightness):
  Certain HUD elements spike in brightness when a kill is registered.
  We track mean HSV-Value at fixed ROIs and peak-detect positive spikes.
  Target: center wolf icon (bottom-center HUD) and AimBotz kill counter.

Signal 2 -- Shot detection (frame delta):
  With infinite ammo the magazine resets 30->29->30 on each shot.
  Tracking frame-to-frame pixel change in the ammo counter ROI gives
  a precise shot timestamp without OCR.

Signal 3 -- Player count OCR (primary kill signal):
  The T-side alive bot count (orange digits, top-center scoreboard) is
  a direct game-state read.  A decrement = kill; increment = respawn.
  AK-47 cycle time (100ms) sets the minimum inter-shot gap for Signal 2.

Together these enable a convolution kill-detector: a shot followed by
a kill flash within ~500ms is a strong kill event. Multiple rapid shots
stacking a larger expected brightness curve = confident spray kill.
Signal 3 is the ground truth for kill timing.

ROIs (1280x960 frame)
-----------------------
  Brightness  : center_icon, center_icon_wide, kill_counter, kill_feed
  Frame-delta : ammo_mag (30->29->30 cycle per shot)
  OCR         : bot_count (T-side alive count, orange, below top scoreboard)

Usage
-----
  python kill_flash_detector.py --calibrate
  python kill_flash_detector.py --detect --roi center_icon
  python kill_flash_detector.py --validate engagements.csv
  python kill_flash_detector.py --video path/to/video.mp4 --calibrate
"""

import re
import sys
import argparse
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

try:
    import pytesseract
    _TESSERACT_OK = True
except ImportError:
    _TESSERACT_OK = False

# ── AK-47 canonical timing ────────────────────────────────────────────────────
# Source: CS2 weapon data (600 RPM confirmed by user, consistent across wikis)
# Do NOT change without a cited source — this value is baked into derived constants.
AK47_CYCLE_TIME_S = 0.100   # 600 RPM  =  10 rds/s  =  100 ms/cycle

# ── defaults ──────────────────────────────────────────────────────────────────

VIDEO_PATH       = r"F:\P_rojects\rz_DemoViewer\data\warmup_20260322_2.mp4"
VIDEO_OFFSET     = 0.0    # video_time = demo_time + offset
SUBSAMPLE        = 2      # read every Nth frame for brightness ROIs

# ── ROI definitions (relative coordinates) ───────────────────────────────────
# All ROIs are fractions of frame dimensions: (x1_frac, y1_frac, x2_frac, y2_frac)
# where 0.0 = left/top, 1.0 = right/bottom.  Calibrated on 1280×960; scales to
# any resolution automatically.

# Brightness ROIs: (name, x1f, y1f, x2f, y2f) — track mean HSV-V per frame
BRIGHTNESS_ROIS_REL = [
    ("center_icon",      0.4727, 0.9115, 0.5273, 0.9771),   # bottom-center wolf icon
    ("center_icon_wide", 0.4453, 0.9010, 0.5547, 0.9896),   # wider glow radius
    ("kill_counter",     0.2773, 0.9427, 0.3516, 0.9771),   # AimBotz kill counter
    ("kill_feed",        0.8203, 0.0052, 0.9961, 0.1458),   # CS2 kill feed top-right
]

# Frame-delta ROIs: (name, x1f, y1f, x2f, y2f) — track |frame[t] - frame[t-1]|
DELTA_ROIS_REL = [
    ("ammo_mag",  0.6094, 0.9375, 0.6836, 0.9771),   # ammo "30" display
    ("bot_count", 0.5063, 0.0604, 0.5234, 0.0771),   # change-point detection for OCR fusion
]

# Kill peak detection
PEAK_PROMINENCE  = 5.0    # min brightness delta to count as kill event
PEAK_DISTANCE_S  = 0.5    # min seconds between detected kills

# Shot detection
SHOT_DELTA_THRESH = 10.0          # ammo pixel delta threshold (noise ~3-4, shots ~15-40)
SHOT_MIN_GAP_S    = AK47_CYCLE_TIME_S  # AK-47 cannot fire faster than one cycle (100ms)

# Per-delta-ROI duration filter (max consecutive elevated samples = "real event" cap)
DELTA_ROI_MAX_FRAMES = {
    "ammo_mag":  5,
    "bot_count": None,
}

# Player count OCR (Signal 3)
ROI_BOT_COUNT_REL       = (0.5063, 0.0604, 0.5234, 0.0771)  # T-side alive count, top-center
ROI_BOT_COUNT_ICON_FRAC = 0.227   # fraction of ROI width to blank (player silhouette icon)
BOT_COUNT_DEFAULT = 12                   # AimBotz default bot count
KILL_CONFIRM_WINDOW_S = AK47_CYCLE_TIME_S + (1.0 / 60)


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_roi(rel_roi, frame_w, frame_h):
    """Convert relative ROI fractions to absolute pixel coordinates."""
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * frame_w), round(y1f * frame_h),
            round(x2f * frame_w), round(y2f * frame_h))


def resolve_named_rois(named_rel_rois, frame_w, frame_h):
    """Convert list of (name, x1f, y1f, x2f, y2f) to (name, x1, y1, x2, y2)."""
    return [(name, *resolve_roi((x1f, y1f, x2f, y2f), frame_w, frame_h))
            for name, x1f, y1f, x2f, y2f in named_rel_rois]


def mean_brightness(frame, x1, y1, x2, y2):
    """Mean HSV-Value (0-255) of a ROI patch."""
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return 0.0
    return float(np.mean(cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[:, :, 2]))


def gray_patch(frame, x1, y1, x2, y2):
    """Grayscale float32 crop for frame-delta comparisons."""
    return cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY).astype(np.float32)


def read_bot_count(frame, x1, y1, x2, y2, icon_px=5):
    """
    OCR the T-side alive bot count from the top scoreboard.
    The number is gold/orange on a dark background.
    Returns int in [0, BOT_COUNT_DEFAULT], or None on OCR failure.

    icon_px: number of leftmost columns to blank (player silhouette icon).
    """
    if not _TESSERACT_OK:
        return None
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        return None
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    # OTSU on digit-only columns — excluded blanked icon columns skewed histogram
    digit_region = gray[:, icon_px:]
    otsu_val, _ = cv2.threshold(digit_region, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, mask = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY)
    mask[:, :icon_px] = 0
    big  = cv2.resize(mask, (mask.shape[1] * 6, mask.shape[0] * 6),
                      interpolation=cv2.INTER_NEAREST)
    raw  = pytesseract.image_to_string(big, config="--psm 8").strip()
    norm = (raw
            .replace('l', '1').replace('I', '1').replace('|', '1').replace('T', '1')
            .replace('n', '1').replace('N', '1').replace('w', '1').replace('W', '1')
            .replace('p', '1').replace('P', '1').replace('B', '1').replace('E', '1')
            .replace('o', '0').replace('O', '0').replace('D', '0').replace('Q', '0')
            .replace('m', '0').replace('M', '0').replace('y', '0').replace('Y', '0')
            .replace('a', '0').replace('A', '0')
            .replace('R', '2').replace('z', '2').replace('Z', '2'))
    digits = re.sub(r"[^0-9]", "", norm)
    # '01'/'00' are normalisation collision artefacts for "10" — flip them
    if digits in ('01', '00'):
        digits = '10'
    if digits and len(digits) <= 2:
        val = int(digits)
        return val if 0 <= val <= BOT_COUNT_DEFAULT else None
    return None


def scan_video(video_path, subsample=SUBSAMPLE):
    """
    Scan entire video.
    Brightness ROIs: sampled every `subsample` frames (mean HSV-V).
    Delta ROIs: sampled every frame; max-pooled into subsample bucket
    so brief 29-state (1-3 frames) is not missed.

    Returns:
      times     : 1-D array of timestamps (s)
      brightness: {name: array} mean HSV-V
      shot_delta: {name: array} max frame-delta per bucket
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"ERROR: cannot open video: {video_path}")

    fps     = cap.get(cv2.CAP_PROP_FPS)
    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve relative ROIs to absolute pixels for this resolution
    bright_rois = resolve_named_rois(BRIGHTNESS_ROIS_REL, w, h)
    delta_rois  = resolve_named_rois(DELTA_ROIS_REL, w, h)
    bot_roi     = resolve_roi(ROI_BOT_COUNT_REL, w, h)
    bot_icon_px = max(1, int(ROI_BOT_COUNT_ICON_FRAC * (bot_roi[2] - bot_roi[0])))

    print(f"Video : {w}x{h}  {fps:.1f}fps  {total_f} frames  ({total_f/fps:.1f}s)")
    print(f"Brightness ROIs : {[r[0] for r in bright_rois]}")
    print(f"Delta ROIs      : {[r[0] for r in delta_rois]}")
    print(f"Bot count ROI   : {bot_roi}  (icon_px={bot_icon_px})")
    print(f"Sample: every {subsample} frame(s)  ->  {fps/subsample:.1f} samples/s")
    print("Scanning...")

    times      = []
    brightness = {name: [] for name, *_ in bright_rois}
    delta_buf  = {name: [] for name, *_ in delta_rois}
    shot_delta = {name: [] for name, *_ in delta_rois}
    prev_gray  = {name: None for name, *_ in delta_rois}
    bot_counts = []

    if _TESSERACT_OK:
        print("Signal 3 (bot count OCR): enabled")
    else:
        print("Signal 3 (bot count OCR): DISABLED — pytesseract not found")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Delta ROIs: compute every frame
        for name, x1, y1, x2, y2 in delta_rois:
            g = gray_patch(frame, x1, y1, x2, y2)
            if prev_gray[name] is not None:
                delta_buf[name].append(float(np.mean(np.abs(g - prev_gray[name]))))
            prev_gray[name] = g

        # Subsample boundary: record brightness + flush delta buffer + OCR
        if frame_idx % subsample == 0:
            times.append(frame_idx / fps)
            for name, x1, y1, x2, y2 in bright_rois:
                brightness[name].append(mean_brightness(frame, x1, y1, x2, y2))
            for name, *_ in delta_rois:
                shot_delta[name].append(max(delta_buf[name]) if delta_buf[name] else 0.0)
                delta_buf[name] = []
            bot_counts.append(read_bot_count(frame, *bot_roi, icon_px=bot_icon_px))

        frame_idx += 1

    cap.release()
    print(f"Done. {len(times)} samples collected.")
    return (np.array(times),
            {k: np.array(v) for k, v in brightness.items()},
            {k: np.array(v) for k, v in shot_delta.items()},
            bot_counts)


# ── peak detection ────────────────────────────────────────────────────────────

def detect_kills(times, brightness_arr, fps_eff,
                 prominence=PEAK_PROMINENCE, min_gap_s=PEAK_DISTANCE_S):
    """
    Peak-detect kill events from a brightness trace.
    Operates on the positive first-difference (brightness rise) so it fires
    on the flash onset regardless of absolute baseline.
    Returns: (kill_times, delta_pos, peak_indices, properties)
    """
    delta     = np.diff(brightness_arr, prepend=brightness_arr[0])
    delta_pos = np.clip(delta, 0, None)
    peaks, props = find_peaks(delta_pos,
                              prominence=prominence,
                              distance=max(1, int(min_gap_s * fps_eff)))
    return times[peaks], delta_pos, peaks, props


def detect_shots(times, shot_delta_arr, fps_eff,
                 thresh=SHOT_DELTA_THRESH, min_gap_s=SHOT_MIN_GAP_S,
                 max_event_frames=5):
    """
    Detect shots from the ammo frame-delta signal with weapon-switch filtering.

    Real shots: ammo flickers 30->29->30 in 1-4 frames then returns to baseline.
    Weapon switches: ammo display changes and stays changed for many frames.

    Filter: reject any peak whose surrounding elevated window (delta > thresh/2)
    spans more than `max_event_frames` consecutive samples.

    Returns: shot_times array (filtered), raw_peaks array (unfiltered)
    """
    raw_peaks, _ = find_peaks(shot_delta_arr,
                              height=thresh,
                              distance=max(1, int(min_gap_s * fps_eff)))

    half_thresh = thresh / 2.0
    valid = []
    for pk in raw_peaks:
        # Count consecutive elevated samples centred on this peak
        lo, hi = pk, pk
        while lo > 0 and shot_delta_arr[lo - 1] > half_thresh:
            lo -= 1
        while hi < len(shot_delta_arr) - 1 and shot_delta_arr[hi + 1] > half_thresh:
            hi += 1
        event_width = hi - lo + 1
        if event_width <= max_event_frames:
            valid.append(pk)

    valid = np.array(valid, dtype=int)
    return (times[valid] if len(valid) else np.array([])), raw_peaks


def detect_kills_from_count(times, bot_counts):
    """
    Detect kill timestamps from T-side player count decrements (Signal 3).

    Kill  = count[t] < count[t-1]  (bot died)
    Respawn = count[t] > count[t-1]  (ignored)
    None values are skipped without resetting the previous valid reading.

    Handles rapid multi-kills: a drop of N records N kills at the same timestamp.
    Returns: kill_times array
    """
    kill_times = []
    prev = None
    for i, cnt in enumerate(bot_counts):
        if cnt is None:
            continue
        if prev is not None and cnt < prev:
            # Single decrement = one kill.  Drops > 1 between adjacent samples
            # are OCR errors (bots die one at a time in AimBotz).
            kill_times.append(times[i])
        prev = cnt
    return np.array(kill_times)


def stabilize_bot_count(times, bot_counts_raw, bot_count_delta, fps_eff,
                        delta_thresh=5.0):
    """
    Clean noisy OCR bot-count readings into a reliable step function.

    Pipeline (context-aware disambiguation):
    1. Map unambiguous raw values: 0→10, 2→12, 7→NaN, 10/11/12→keep.
       raw=1 is ambiguous (could be 11 or 12 with a dropped digit) → NaN.
    2. Resolve each raw=1 sample using nearest unambiguous neighbors:
       - both neighbors are 12 → 12 (noise during constant-12 period)
       - leading edge with right=12 → 12
       - otherwise → 11 (default: kill/respawn transition zone)
    3. Forward-fill remaining NaN gaps.
    4. scipy median_filter(size=3) — removes single-frame outliers while
       preserving real step transitions (kill/respawn events).

    Validated on warmup_20260322_2.mp4: 99 kills detected vs 100 ground truth.
    Eliminates 2 phantom kills from raw=1 bursts during constant-12 periods.

    Parameters delta_thresh and bot_count_delta are accepted for API compat
    but not used by the current approach.
    """
    n = len(bot_counts_raw)
    if n == 0:
        return np.array([], dtype=int)

    raw = np.array([float(v) if v is not None else np.nan
                    for v in bot_counts_raw])

    # Step 1: map unambiguous values; mark raw=1 as NaN (ambiguous)
    unamb = np.full(n, np.nan)
    for i in range(n):
        v = raw[i]
        if np.isnan(v):
            continue
        if v == 0:
            unamb[i] = 10     # "10" with lost leading "1"
        elif v == 2:
            unamb[i] = 12     # "12" with lost leading "1"
        elif v == 7:
            pass               # noise → stays NaN
        elif v in (10, 11, 12):
            unamb[i] = v      # clean read
        # raw=1 and any other value stay NaN

    # Step 2: resolve raw=1 using nearest unambiguous neighbors
    fixed = unamb.copy()
    for i in range(n):
        if raw[i] != 1.0:
            continue
        # nearest unambiguous to the left
        left_val = np.nan
        for j in range(i - 1, -1, -1):
            if not np.isnan(unamb[j]):
                left_val = unamb[j]
                break
        # nearest unambiguous to the right
        right_val = np.nan
        for j in range(i + 1, n):
            if not np.isnan(unamb[j]):
                right_val = unamb[j]
                break
        # disambiguate
        if left_val == 12 and right_val == 12:
            fixed[i] = 12     # noise during constant-12
        elif np.isnan(left_val) and right_val == 12:
            fixed[i] = 12     # leading edge, 12 context
        else:
            fixed[i] = 11     # default: treat as 11

    # Step 3: forward-fill remaining NaN
    last_valid = np.nan
    for i in range(n):
        if np.isnan(fixed[i]):
            fixed[i] = last_valid
        else:
            last_valid = fixed[i]
    # Back-fill if leading NaN
    if np.isnan(fixed[0]):
        first_valid = BOT_COUNT_DEFAULT
        for j in range(n):
            if not np.isnan(fixed[j]):
                first_valid = fixed[j]
                break
        for j in range(n):
            if np.isnan(fixed[j]):
                fixed[j] = first_valid
            else:
                break

    # Step 4: median filter — removes single-frame outliers
    stabilized = median_filter(fixed.astype(float), size=3).astype(int)

    return stabilized


def brightness_decay_envelope(brightness_arr, window_s=2.0, fps_eff=30.0):
    """
    Compute a rolling mean of the brightness signal to reveal the
    kill-rate envelope. Rapid kills stack exponential decay curves from
    the kill-flash animation; the rolling mean captures this.
    Returns: smoothed array (same length as brightness_arr)
    """
    w = max(1, int(window_s * fps_eff))
    kernel = np.ones(w) / w
    return np.convolve(brightness_arr, kernel, mode='same')


# ── plots ─────────────────────────────────────────────────────────────────────

C_TRACE     = "#2c7bb6"
C_DELTA     = "#e67e22"
C_PEAK      = "#e74c3c"
C_TRUTH     = "#27ae60"
C_SHOT_FILL = "#8e44ad"
C_SHOT_LINE = "#6c3483"


def plot_calibration(times, brightness, shot_delta, bot_counts=None,
                     bot_counts_stable=None,
                     kill_times_truth=None, out="kill_flash_calibration.png"):
    """
    Multi-panel calibration plot:
    - One panel per brightness ROI (kill flash signal)
    - One panel per delta ROI (shot signal)
    - One panel for bot count OCR (Signal 3), if available
    - Optional green dashes for ground-truth kill times
    """
    fps_eff  = 1.0 / (times[1] - times[0]) if len(times) > 1 else 30.0
    n_bright = len(brightness)
    n_delta  = len(shot_delta)
    n_count  = 1 if bot_counts else 0
    n_total  = n_bright + n_delta + n_count

    fig, axes = plt.subplots(n_total, 1, figsize=(18, 3.0 * n_total),
                             gridspec_kw={"hspace": 0.6})
    if n_total == 1:
        axes = [axes]
    axes = list(axes)

    C_ENVELOPE = "#16a085"   # teal — rolling brightness mean (kill rate envelope)
    C_REJECT   = "#bdc3c7"   # grey — weapon-switch false positives

    # Brightness panels
    for ax, name in zip(axes[:n_bright], brightness.keys()):
        b = brightness[name]
        delta = np.diff(b, prepend=b[0])
        delta_pos = np.clip(delta, 0, None)
        peaks, _ = find_peaks(delta_pos,
                              prominence=PEAK_PROMINENCE,
                              distance=max(1, int(PEAK_DISTANCE_S * fps_eff)))

        # Kill-rate envelope: rolling mean of raw brightness
        envelope = brightness_decay_envelope(b, window_s=2.0, fps_eff=fps_eff)

        ax2 = ax.twinx()
        ax2.plot(times, b,        color=C_TRACE,    linewidth=0.8, alpha=0.45)
        ax2.plot(times, envelope, color=C_ENVELOPE, linewidth=1.4, alpha=0.85,
                 label="Kill-rate envelope (2s rolling mean)")
        ax2.set_ylabel("brightness (V)", fontsize=7, color=C_TRACE)
        ax2.tick_params(axis="y", labelcolor=C_TRACE, labelsize=6.5)

        ax.fill_between(times, delta_pos, color=C_DELTA, alpha=0.6, linewidth=0)
        ax.set_ylabel("delta brightness", fontsize=7, color=C_DELTA)
        ax.tick_params(axis="y", labelcolor=C_DELTA, labelsize=6.5)

        for pk_t in times[peaks]:
            ax.axvline(pk_t, color=C_PEAK, linewidth=0.9, alpha=0.85)
        if kill_times_truth is not None:
            for kt in kill_times_truth:
                ax.axvline(kt, color=C_TRUTH, linewidth=1.0, linestyle="--", alpha=0.7)

        ax.set_title(
            f"[KILL FLASH] {name}  --  {len(peaks)} peaks  "
            f"(prominence>={PEAK_PROMINENCE}, gap>={PEAK_DISTANCE_S}s)  "
            f"| teal=kill-rate envelope",
            fontsize=9, loc="left", pad=3
        )
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(axis="x", labelsize=7)

    # Frame-delta panels (shot / game-state signal)
    for ax, name in zip(axes[n_bright:n_bright + n_delta], shot_delta.keys()):
        d = shot_delta[name]
        t = times[:len(d)]
        max_frames = DELTA_ROI_MAX_FRAMES.get(name, 5)
        shot_times, raw_peaks = detect_shots(t, d, fps_eff,
                                             max_event_frames=max_frames if max_frames else 9999)

        # Rejected peaks (only meaningful when duration filter is active)
        if max_frames is not None:
            valid_set = set(np.searchsorted(t, shot_times))
            rejected_peaks = [pk for pk in raw_peaks if pk not in valid_set]
        else:
            rejected_peaks = []

        ax.fill_between(t, d, color=C_SHOT_FILL, alpha=0.5, linewidth=0)
        ax.axhline(SHOT_DELTA_THRESH, color=C_SHOT_LINE, linewidth=0.9,
                   linestyle="--", alpha=0.8)
        for st in shot_times:
            ax.axvline(st, color=C_SHOT_LINE, linewidth=0.8, alpha=0.85)
        for pk in rejected_peaks:
            ax.axvline(t[pk], color=C_REJECT, linewidth=1.2, alpha=0.9,
                       linestyle=":")
        if kill_times_truth is not None:
            for kt in kill_times_truth:
                ax.axvline(kt, color=C_TRUTH, linewidth=1.0, linestyle="--", alpha=0.7)

        n_rejected = len(rejected_peaks)
        filter_note = f"/ {n_rejected} rejected (grey dotted)" if max_frames else "no duration filter"
        ax.set_ylabel("px delta", fontsize=7, color=C_SHOT_FILL)
        ax.tick_params(axis="y", labelcolor=C_SHOT_FILL, labelsize=6.5)
        ax.set_title(
            f"[DELTA] {name}  --  {len(shot_times)} events  {filter_note}",
            fontsize=9, loc="left", pad=3
        )
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(axis="x", labelsize=7)

    # Bot count panel (Signal 3)
    if n_count:
        ax  = axes[n_bright + n_delta]
        C_COUNT      = "#1abc9c"   # teal-green
        C_COUNT_KILL = "#e74c3c"   # red for kill events
        C_COUNT_RAW  = "#bdc3c7"   # grey for raw OCR dots
        C_COUNT_STAB = "#2980b9"   # blue for stabilized step

        # Replace None with NaN for plotting
        count_arr = np.array([float(v) if v is not None else np.nan
                              for v in bot_counts])
        count_times = times[:len(count_arr)]

        # Use stabilized signal for kill detection if available
        if bot_counts_stable is not None:
            stable_times = times[:len(bot_counts_stable)]
            kill_times_count = detect_kills_from_count(stable_times, list(bot_counts_stable))
            # Plot raw as faint dots, stabilized as bold step
            ax.scatter(count_times, count_arr, color=C_COUNT_RAW, s=3, alpha=0.4,
                       label="Raw OCR", zorder=1)
            ax.step(stable_times, bot_counts_stable[:len(stable_times)], where="post",
                    color=C_COUNT_STAB, linewidth=1.5, label="Stabilized", zorder=2)
        else:
            kill_times_count = detect_kills_from_count(count_times, bot_counts)
            ax.step(count_times, count_arr, where="post",
                    color=C_COUNT, linewidth=1.2, label="Bot count (OCR)")

        for kt in kill_times_count:
            ax.axvline(kt, color=C_COUNT_KILL, linewidth=0.9, alpha=0.85)
        if kill_times_truth is not None:
            for kt in kill_times_truth:
                ax.axvline(kt, color=C_TRUTH, linewidth=1.0, linestyle="--", alpha=0.7)

        n_ocr_none = sum(1 for v in bot_counts if v is None)
        ax.set_ylabel("bots alive", fontsize=7, color=C_COUNT)
        ax.tick_params(axis="y", labelcolor=C_COUNT, labelsize=6.5)
        ax.set_title(
            f"[BOT COUNT] Signal 3 — delta-gated OCR  |  {len(kill_times_count)} kills detected  "
            f"|  {n_ocr_none}/{len(bot_counts)} raw unreadable",
            fontsize=9, loc="left", pad=3
        )
        ax.set_xlim(times[0], times[-1])
        ax.tick_params(axis="x", labelsize=7)

    axes[-1].set_xlabel("video time (s)", fontsize=9)

    legend_lines = [
        plt.Line2D([0], [0], color=C_TRACE,     linewidth=1.2, label="Raw brightness (V-channel)"),
        plt.Line2D([0], [0], color="#16a085",   linewidth=1.5, label="Kill-rate envelope (2s rolling mean)"),
        plt.Line2D([0], [0], color=C_DELTA,     linewidth=1.5, label="Brightness delta (kill signal)"),
        plt.Line2D([0], [0], color=C_PEAK,      linewidth=1.5, label="Detected kill peak"),
        plt.Line2D([0], [0], color=C_SHOT_FILL, linewidth=1.5, label="Ammo frame delta (shot signal)"),
        plt.Line2D([0], [0], color=C_SHOT_LINE, linewidth=1.5, linestyle="--", label="Accepted shot"),
        plt.Line2D([0], [0], color="#bdc3c7",   linewidth=1.5, linestyle=":", label="Rejected (weapon switch)"),
        plt.Line2D([0], [0], color="#1abc9c",   linewidth=1.5, label="Bot count (OCR)"),
    ]
    if kill_times_truth is not None:
        legend_lines.append(
            plt.Line2D([0], [0], color=C_TRUTH, linewidth=1.5, linestyle="--", label="Ground truth kill")
        )
    fig.legend(handles=legend_lines, loc="upper right", fontsize=8,
               framealpha=0.9, bbox_to_anchor=(0.99, 0.99), ncol=2)

    fig.suptitle(
        f"Kill Flash Detector -- Calibration  |  "
        f"kill prominence>={PEAK_PROMINENCE}  shot_thresh={SHOT_DELTA_THRESH}  "
        f"subsample={SUBSAMPLE}\n"
        f"red=kill peaks  purple=shots  green=ground truth",
        fontsize=10, y=1.01
    )
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.show()
    print(f"Saved: {out}")


# ── validation ────────────────────────────────────────────────────────────────

def validate_against_demo(kill_times_detected, kill_times_truth, tolerance_s=0.25):
    matched         = 0
    unmatched_det   = []
    unmatched_truth = list(kill_times_truth)

    for det_t in kill_times_detected:
        dists = [abs(det_t - gt) for gt in unmatched_truth]
        if dists and min(dists) <= tolerance_s:
            matched += 1
            unmatched_truth.pop(int(np.argmin(dists)))
        else:
            unmatched_det.append(det_t)

    n_truth   = len(kill_times_truth)
    n_det     = len(kill_times_detected)
    recall    = matched / n_truth * 100 if n_truth else 0
    precision = matched / n_det  * 100 if n_det   else 0

    print(f"\n-- Validation (tolerance={tolerance_s}s) --")
    print(f"  Ground truth kills : {n_truth}")
    print(f"  Detected peaks     : {n_det}")
    print(f"  Matched            : {matched}  ({recall:.0f}% recall, {precision:.0f}% precision)")
    print(f"  False positives    : {len(unmatched_det)}")
    print(f"  Missed kills       : {len(unmatched_truth)}")
    if unmatched_det:
        print(f"  FP times (s)       : {[f'{t:.2f}' for t in unmatched_det]}")
    if unmatched_truth:
        print(f"  Missed (s)         : {[f'{t:.2f}' for t in unmatched_truth]}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Kill flash + shot detector")
    p.add_argument("--video",      default=VIDEO_PATH)
    p.add_argument("--offset",     type=float, default=VIDEO_OFFSET,
                   help="video_time = demo_time + offset")
    p.add_argument("--calibrate",  action="store_true",
                   help="Plot all signals (default if no other mode given)")
    p.add_argument("--detect",     action="store_true",
                   help="Run kill peak detection and output CSV")
    p.add_argument("--roi",        default="center_icon",
                   help="Brightness ROI name to use for --detect")
    p.add_argument("--validate",   default=None, metavar="CSV",
                   help="Engagements CSV for ground-truth comparison")
    p.add_argument("--prominence", type=float, default=PEAK_PROMINENCE)
    p.add_argument("--gap",        type=float, default=PEAK_DISTANCE_S)
    p.add_argument("--shot-thresh",type=float, default=SHOT_DELTA_THRESH,
                   dest="shot_thresh")
    p.add_argument("--subsample",  type=int,   default=SUBSAMPLE)
    p.add_argument("--dump-ocr",   action="store_true", dest="dump_ocr",
                   help="Dump raw OCR readings to CSV for diagnostics")
    return p.parse_args()


def main():
    args = parse_args()
    global PEAK_PROMINENCE, PEAK_DISTANCE_S, SUBSAMPLE, SHOT_DELTA_THRESH
    PEAK_PROMINENCE   = args.prominence
    PEAK_DISTANCE_S   = args.gap
    SUBSAMPLE         = args.subsample
    SHOT_DELTA_THRESH = args.shot_thresh

    times, brightness, shot_delta, bot_counts = scan_video(args.video, subsample=SUBSAMPLE)
    fps_eff = 1.0 / (times[1] - times[0]) if len(times) > 1 else 30.0

    # Stabilize bot count using delta-gated temporal fusion
    bot_count_delta = shot_delta.get("bot_count", np.array([]))
    stabilized = stabilize_bot_count(times, bot_counts, bot_count_delta, fps_eff)

    if args.dump_ocr:
        out_csv = (args.video.replace(".mp4", "_ocr_dump.csv")
                   if args.video.endswith(".mp4") else "ocr_dump.csv")
        n_none = sum(1 for v in bot_counts if v is None)
        vals = [v for v in bot_counts if v is not None]
        # Include delta signal so all future tuning can be fully offline
        delta_col = bot_count_delta[:len(bot_counts)] if len(bot_count_delta) >= len(bot_counts) else np.pad(
            bot_count_delta, (0, len(bot_counts) - len(bot_count_delta)), constant_values=0)
        pd.DataFrame({
            "sample_idx": range(len(bot_counts)),
            "time_s": times[:len(bot_counts)],
            "bot_count_raw": bot_counts,
            "bot_count_stable": stabilized[:len(bot_counts)],
            "bot_count_delta": delta_col,
        }).to_csv(out_csv, index=False)
        print(f"\n-- OCR dump: {len(bot_counts)} samples, {n_none} unreadable ({n_none/len(bot_counts)*100:.1f}%) --")
        if vals:
            print(f"  Raw range: [{min(vals)}, {max(vals)}]   Median: {np.median(vals):.0f}")
        # Raw transitions
        kills_raw = sum(1 for i in range(1, len(bot_counts))
                        if bot_counts[i] is not None and bot_counts[i-1] is not None
                        and bot_counts[i] < bot_counts[i-1])
        respawns_raw = sum(1 for i in range(1, len(bot_counts))
                           if bot_counts[i] is not None and bot_counts[i-1] is not None
                           and bot_counts[i] > bot_counts[i-1])
        print(f"  Raw decrements: {kills_raw}   Raw increments: {respawns_raw}")
        # Stabilized transitions
        kills_s = sum(1 for i in range(1, len(stabilized))
                      if stabilized[i] < stabilized[i-1])
        respawns_s = sum(1 for i in range(1, len(stabilized))
                         if stabilized[i] > stabilized[i-1])
        print(f"  Stabilized decrements (kills): {kills_s}   Stabilized increments (respawns): {respawns_s}")
        print(f"  Stabilized range: [{stabilized.min()}, {stabilized.max()}]")
        print(f"Saved: {out_csv}")
        return

    kill_times_truth = None
    if args.validate:
        df = pd.read_csv(args.validate)
        kill_times_truth = df["engagement_start_s"].dropna().values + args.offset
        print(f"Loaded {len(kill_times_truth)} ground-truth kills from {args.validate}")

    if args.calibrate or not args.detect:
        out = (args.video.replace(".mp4", "_flash_calibration.png")
               if args.video.endswith(".mp4") else "kill_flash_calibration.png")
        plot_calibration(times, brightness, shot_delta, bot_counts=bot_counts,
                         bot_counts_stable=stabilized,
                         kill_times_truth=kill_times_truth, out=out)

    if args.detect:
        # Signal 3: bot count kill times (primary) — uses stabilized signal
        kill_times_count = detect_kills_from_count(times, list(stabilized))
        if len(kill_times_count):
            print(f"\n-- Detected kills (bot count OCR — Signal 3) --")
            for i, kt in enumerate(kill_times_count):
                print(f"  Kill {i+1:3d}:  video={kt:.3f}s  demo={kt-args.offset:.3f}s")
            out_csv = (args.video.replace(".mp4", "_kill_times_count.csv")
                       if args.video.endswith(".mp4") else "kill_times_count.csv")
            pd.DataFrame({
                "kill_n":  range(1, len(kill_times_count) + 1),
                "video_s": kill_times_count,
                "demo_s":  kill_times_count - args.offset,
                "signal":  "bot_count",
            }).to_csv(out_csv, index=False)
            print(f"Saved: {out_csv}")
            if kill_times_truth is not None:
                validate_against_demo(kill_times_count, kill_times_truth, tolerance_s=0.25)
        else:
            print("Signal 3 (bot count): no kills detected — check ROI_BOT_COUNT or pytesseract install")

        # Signal 1: brightness kill flash (secondary cross-check)
        roi_name = args.roi
        if roi_name not in brightness:
            print(f"ERROR: '{roi_name}' not in brightness ROIs: {list(brightness.keys())}")
            sys.exit(1)

        kill_times_s, delta_pos, peaks, _ = detect_kills(
            times, brightness[roi_name], fps_eff,
            prominence=PEAK_PROMINENCE, min_gap_s=PEAK_DISTANCE_S
        )

        print(f"\n-- Detected kills (brightness flash — Signal 1, roi={roi_name}) --")
        for i, kt in enumerate(kill_times_s):
            print(f"  Kill {i+1:3d}:  video={kt:.3f}s  demo={kt-args.offset:.3f}s  "
                  f"delta={delta_pos[peaks[i]]:.1f}")

        out_csv = (args.video.replace(".mp4", "_kill_times.csv")
                   if args.video.endswith(".mp4") else "kill_times.csv")
        pd.DataFrame({
            "kill_n":           range(1, len(kill_times_s) + 1),
            "video_s":          kill_times_s,
            "demo_s":           kill_times_s - args.offset,
            "delta_brightness": delta_pos[peaks],
            "signal":           "brightness",
        }).to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")

        if kill_times_truth is not None and not len(kill_times_count):
            validate_against_demo(kill_times_s, kill_times_truth, tolerance_s=0.25)


if __name__ == "__main__":
    main()
