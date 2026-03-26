"""
calibrate_rois.py
=================
Phase 0 validation: verifies that pytesseract can reliably read the key
OCR targets from warmup_20260322_2.mp4 before the full parser is built.

Targets
-------
1. Ammo magazine count  — bottom-right HUD, "30 3↑" format (we want the "30")
2. AimBotz kill counter — yellow overlay banner, "X / 100" format (we want X)
3. Bot count            — T-side alive bots, orange digits below top scoreboard

Output
------
- Prints OCR reads + confidence for each ROI at each timestamp
- Saves cropped ROI images to outputs/calibration/ for manual inspection
- Saves top-strip annotated frame for bot-count ROI visual verification
"""

import os
import re
import sys

import cv2
import numpy as np
import pytesseract


VIDEO_PATH = r"F:\P_rojects\rz_DemoViewer\data\warmup_20260322_2.mp4"
OUT_DIR    = r"F:\P_rojects\rz_DemoViewer\outputs\calibration"

# ── ROI definitions (relative coordinates) ────────────────────────────────────
# All ROIs are fractions of frame dimensions.  Calibrated on 1280×960.

ROI_AMMO_REL      = (0.8852, 0.9583, 0.9492, 1.0000)   # ammo "30 3↑"
ROI_KILLS_REL     = (0.0625, 0.7240, 0.1680, 0.7865)   # AimBotz "X / 100" banner
ROI_BOT_COUNT_REL = (0.5063, 0.0604, 0.5234, 0.0771)   # T-side alive count
ROI_BOT_COUNT_ICON_FRAC = 0.227   # fraction of ROI width to blank (icon)
ROI_TOP_STRIP_REL = (0.3125, 0.0000, 0.7031, 0.1042)   # wide top-strip for verification


def resolve_roi(rel_roi, frame_w, frame_h):
    """Convert relative ROI fractions to absolute pixel coordinates."""
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * frame_w), round(y1f * frame_h),
            round(x2f * frame_w), round(y2f * frame_h))

# Sample timestamps (seconds) — all should be mid-challenge
TIMESTAMPS = list(range(2, 66, 2))  # 32 samples, every 2s across ~65s video

SCALE = 3   # upscale factor before OCR


# ── helpers ───────────────────────────────────────────────────────────────────

def crop(frame, roi):
    x1, y1, x2, y2 = roi
    return frame[y1:y2, x1:x2]


def preprocess_white_on_dark(patch):
    """Grayscale → invert → threshold → upscale. For white text on dark BG."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
    big  = cv2.resize(bw, (bw.shape[1] * SCALE, bw.shape[0] * SCALE),
                      interpolation=cv2.INTER_CUBIC)
    return big


def preprocess_dark_on_yellow(patch):
    """Grayscale → threshold → upscale. For dark text on yellow/bright BG."""
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
    big  = cv2.resize(bw, (bw.shape[1] * SCALE, bw.shape[0] * SCALE),
                      interpolation=cv2.INTER_CUBIC)
    return big


def preprocess_orange_on_dark(patch, icon_px=5):
    """
    OTSU threshold computed on digit-only columns (excluding forced-zero icon
    columns that previously skewed the histogram toward a near-zero split).
    Adapts per-frame to varying background brightness (~20 dark vs ~75 bright).
    """
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    digit_region = gray[:, icon_px:]
    otsu_val, _ = cv2.threshold(digit_region, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bw = cv2.threshold(gray, otsu_val, 255, cv2.THRESH_BINARY)
    bw[:, :icon_px] = 0
    big  = cv2.resize(bw, (bw.shape[1] * SCALE * 2, bw.shape[0] * SCALE * 2),
                      interpolation=cv2.INTER_NEAREST)
    return big


def ocr_bot_count(patch_processed):
    """
    OCR for the T-side bot count. PSM 8 (single word), no whitelist.
    Pixel-art "1" is misread as T/n/N/w/W; "0" as m/y/D/Q/O.
    Normalise then strip. Ordering conflicts ('01'→'10') are fixed in parse_bot_count.
    """
    raw = pytesseract.image_to_string(patch_processed, config="--psm 8").strip()
    norm = (raw
            .replace('l', '1').replace('I', '1').replace('|', '1').replace('T', '1')
            .replace('n', '1').replace('N', '1').replace('w', '1').replace('W', '1')
            .replace('p', '1').replace('P', '1').replace('B', '1').replace('E', '1')
            .replace('o', '0').replace('O', '0').replace('D', '0').replace('Q', '0')
            .replace('m', '0').replace('M', '0').replace('y', '0').replace('Y', '0')
            .replace('a', '0').replace('A', '0')
            .replace('R', '2').replace('z', '2').replace('Z', '2'))
    digits = re.sub(r'[^0-9]', '', norm)
    return raw, digits


def parse_bot_count(digits_str):
    """Extract bot count integer (0-12) from normalised digit string.
    '01' is always an ordering artefact (normalised '1' for left digit + '0' for right)
    — flip to '10'. Leading zeros otherwise don't appear in valid bot counts.
    """
    if digits_str:
        # '01' and '00' are ordering/collision artefacts from normalisation — both mean "10"
        if digits_str in ('01', '00'):
            digits_str = '10'
        val = int(digits_str) if len(digits_str) <= 2 else None
        return val if val is not None and 0 <= val <= 12 else None
    return None


def ocr_digits(patch_processed, config="--psm 7 -c tessedit_char_whitelist=0123456789/ "):
    # Note: bot-count OCR uses ocr_bot_count() instead (no whitelist + normalisation)

    text = pytesseract.image_to_string(patch_processed, config=config).strip()
    return text


def parse_ammo(text):
    """Extract the first integer from OCR text (the magazine count)."""
    m = re.search(r'\d+', text)
    return int(m.group()) if m else None


def parse_kills(text):
    """Extract the kill count (X) from 'X / 100' or similar."""
    m = re.search(r'(\d+)\s*/\s*100', text)
    if m:
        return int(m.group(1))
    # fallback: first integer
    m = re.search(r'\d+', text)
    return int(m.group()) if m else None


def seek_frame(cap, t_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
    ret, frame = cap.read()
    return frame if ret else None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO_PATH}", file=sys.stderr)
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS)
    total_s  = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    w        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Resolve relative ROIs to absolute pixels for this resolution
    roi_ammo      = resolve_roi(ROI_AMMO_REL, w, h)
    roi_kills     = resolve_roi(ROI_KILLS_REL, w, h)
    roi_bot_count = resolve_roi(ROI_BOT_COUNT_REL, w, h)
    roi_top_strip = resolve_roi(ROI_TOP_STRIP_REL, w, h)
    bot_icon_px   = max(1, int(ROI_BOT_COUNT_ICON_FRAC * (roi_bot_count[2] - roi_bot_count[0])))

    print(f"Video: {w}×{h}  {fps:.1f}fps  {total_s:.2f}s")
    print(f"Resolved ROIs: ammo={roi_ammo}  kills={roi_kills}  bot_count={roi_bot_count}  icon_px={bot_icon_px}")
    print()
    print(f"{'t(s)':>6}  {'ammo_raw':>12}  {'ammo_val':>8}  "
          f"{'kills_raw':>14}  {'kills_val':>9}  {'count_raw':>10}  {'count_val':>9}")
    print("-" * 80)

    for t in TIMESTAMPS:
        if t > total_s:
            print(f"{t:6.1f}  (beyond video duration {total_s:.1f}s)")
            continue

        frame = seek_frame(cap, t)
        if frame is None:
            print(f"{t:6.1f}  (frame read failed)")
            continue

        # ── ammo ──────────────────────────────────────────────────────────────
        ammo_patch = crop(frame, roi_ammo)
        ammo_proc  = preprocess_white_on_dark(ammo_patch)
        ammo_raw   = ocr_digits(ammo_proc)
        ammo_val   = parse_ammo(ammo_raw)

        # ── kills ─────────────────────────────────────────────────────────────
        kills_patch = crop(frame, roi_kills)
        kills_proc  = preprocess_dark_on_yellow(kills_patch)
        kills_raw   = ocr_digits(kills_proc)
        kills_val   = parse_kills(kills_raw)

        # ── bot count ─────────────────────────────────────────────────────────
        count_patch         = crop(frame, roi_bot_count)
        count_proc          = preprocess_orange_on_dark(count_patch, icon_px=bot_icon_px)
        count_raw, count_digits = ocr_bot_count(count_proc)
        count_val           = parse_bot_count(count_digits)

        print(f"{t:6.1f}  {repr(ammo_raw):>12}  {str(ammo_val):>8}  "
              f"{repr(kills_raw):>14}  {str(kills_val):>9}  "
              f"{repr(count_raw):>10}  {repr(count_digits):>6}  {str(count_val):>9}")

        # ── save crops ────────────────────────────────────────────────────────
        tag = f"t{t:02d}"
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_ammo_raw.jpg"),     ammo_patch)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_ammo_proc.jpg"),    ammo_proc)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_kills_raw.jpg"),    kills_patch)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_kills_proc.jpg"),   kills_proc)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_count_raw.jpg"),    count_patch)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_count_proc.jpg"),   count_proc)

        # Full frame with all ROI boxes + top-strip for bot-count verification
        vis = frame.copy()
        for roi, col, lbl in [
            (roi_ammo,      (0, 255, 0),   "ammo"),
            (roi_kills,     (0, 165, 255), "kills"),
            (roi_bot_count, (0, 100, 255), "bot_count"),
            (roi_top_strip, (200, 200, 0), "top_strip"),
        ]:
            cv2.rectangle(vis, (roi[0], roi[1]), (roi[2], roi[3]), col, 2)
            cv2.putText(vis, lbl, (roi[0], max(roi[1] - 4, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_full_annotated.jpg"), vis)

        # High-res top-strip crop — for visually confirming bot-count ROI position
        top_strip_patch = crop(frame, roi_top_strip)
        top_big   = cv2.resize(top_strip_patch,
                               (top_strip_patch.shape[1] * 3, top_strip_patch.shape[0] * 3),
                               interpolation=cv2.INTER_NEAREST)
        # Draw bot_count box on the enlarged strip
        x_off, y_off = roi_top_strip[0], roi_top_strip[1]
        bx1 = (roi_bot_count[0] - x_off) * 3
        by1 = (roi_bot_count[1] - y_off) * 3
        bx2 = (roi_bot_count[2] - x_off) * 3
        by2 = (roi_bot_count[3] - y_off) * 3
        cv2.rectangle(top_big, (bx1, by1), (bx2, by2), (0, 100, 255), 2)
        cv2.imwrite(os.path.join(OUT_DIR, f"{tag}_top_strip_annotated.jpg"), top_big)

    cap.release()
    print()
    print(f"Crops saved to {OUT_DIR}")
    print()
    print("Pass criteria:")
    print("  ammo_val  == 30 on >=4/6 frames")
    print("  kills_val is a plausible integer")
    print("  count_val in [0..12] — check *_top_strip_annotated.jpg to verify ROI position")


if __name__ == "__main__":
    main()
