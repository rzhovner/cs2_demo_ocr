"""
characterize_bot_count.py
=========================
Phase A of the bot-count benchmark pipeline.

Samples 100 frames evenly across a gameplay video, extracts the bot-count
HUD ROI from each, saves 8x upscaled PNG patches, then uses a Claude vision
agent to read and label each patch as the digit it sees (8-12 or null).

Output
------
  outputs/{stem}_benchmark_patches/   — 100 PNG crops (8x upscaled)
  outputs/{stem}_benchmark_labels.csv — agent-labeled ground truth

CSV columns
-----------
  frame_n, time_s, patch_path, agent_label, agent_confidence, agent_notes

Usage
-----
  python src/characterize_bot_count.py video.mp4
  python src/characterize_bot_count.py video.mp4 --n-samples 100 --out-dir outputs/
"""

import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd

# ── ROI constants (from calibrate_rois.py) ───────────────────────────────────
ROI_BOT_COUNT_REL       = (0.5063, 0.0604, 0.5234, 0.0771)
ROI_BOT_COUNT_ICON_FRAC = 0.227   # left fraction of ROI is player icon — excluded

PATCH_SCALE = 8          # upscale factor for saved patches (improves visibility)
N_SAMPLES   = 100        # default number of evenly-spaced sample frames


# ── helpers ───────────────────────────────────────────────────────────────────

def resolve_roi(rel_roi, w, h):
    x1f, y1f, x2f, y2f = rel_roi
    return (round(x1f * w), round(y1f * h), round(x2f * w), round(y2f * h))


def extract_patches(video_path, n_samples, out_dir):
    """
    Extract n_samples evenly-spaced bot-count ROI patches from video.
    Returns list of dicts: {frame_n, time_s, patch_path}.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: cannot open {video_path}", file=sys.stderr)
        sys.exit(1)

    fps        = cap.get(cv2.CAP_PROP_FPS)
    total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_f / fps

    print(f"Video: {w}×{h}  {fps:.1f}fps  {total_f} frames  ({duration_s:.1f}s)")

    roi_abs  = resolve_roi(ROI_BOT_COUNT_REL, w, h)
    x1, y1, x2, y2 = roi_abs
    icon_px  = max(1, round(ROI_BOT_COUNT_ICON_FRAC * (x2 - x1)))

    print(f"Bot-count ROI (abs): x={x1}-{x2}  y={y1}-{y2}  icon_px={icon_px}")

    # Evenly space sample frame indices across the full video
    sample_frames = [round(i * (total_f - 1) / (n_samples - 1)) for i in range(n_samples)]

    patches = []
    for i, fn in enumerate(sample_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
        ret, frame = cap.read()
        if not ret:
            print(f"  [warn] frame {fn} read failed — skipping")
            continue

        t = fn / fps
        patch = frame[y1:y2, x1:x2].copy()

        # Upscale with nearest-neighbour to preserve crisp pixel art digits
        big = cv2.resize(patch,
                         (patch.shape[1] * PATCH_SCALE, patch.shape[0] * PATCH_SCALE),
                         interpolation=cv2.INTER_NEAREST)

        fname = f"patch_{i:03d}_f{fn:06d}_t{t:.2f}s.png"
        fpath = os.path.join(out_dir, fname)
        cv2.imwrite(fpath, big)

        patches.append({"frame_n": fn, "time_s": round(t, 4), "patch_path": fpath})

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  {i+1}/{n_samples} patches extracted...")

    cap.release()
    print(f"Extracted {len(patches)} patches to {out_dir}")
    return patches


# ── agent vision labeling ────────────────────────────────────────────────────

def label_patches_with_agent(patches):
    """
    Use a Claude vision agent (this process itself, via the Read tool's image
    rendering capability) to read each patch and return labels.

    Each patch is a small upscaled crop of the bot-count HUD digit region
    (orange digits on dark background).  Valid values: 8, 9, 10, 11, 12.
    Unreadable / transition frames → None.

    Returns list of dicts with agent_label, agent_confidence, agent_notes
    added to each patch record.
    """
    import anthropic

    client = anthropic.Anthropic()

    labeled = []
    print(f"\nLabeling {len(patches)} patches with Claude vision...")

    for i, p in enumerate(patches):
        patch_path = p["patch_path"]

        # Read image bytes
        with open(patch_path, "rb") as f:
            img_bytes = f.read()

        import base64
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        prompt = (
            "This is a small crop (8× upscaled, nearest-neighbour) from a CS2 AimBotz "
            "gameplay HUD. It shows the T-side alive bot count — a 1 or 2 digit number "
            "in orange/amber pixel-art font on a near-black background. The valid values "
            "are 8, 9, 10, 11, or 12. There may be a partial player silhouette icon on "
            "the left edge — ignore it.\n\n"
            "Reply with ONLY a JSON object with these keys:\n"
            "  label: integer (8-12) or null if unreadable\n"
            "  confidence: \"high\", \"medium\", or \"low\"\n"
            "  notes: brief string if anything unusual (else empty string)\n\n"
            "Example: {\"label\": 11, \"confidence\": \"high\", \"notes\": \"\"}\n"
            "Example: {\"label\": null, \"confidence\": \"low\", \"notes\": \"transition blur\"}"
        )

        try:
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=100,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }],
            )

            raw = response.content[0].text.strip()

            # Parse JSON response
            import json, re
            # Strip any markdown code fences if present
            raw_clean = re.sub(r"```[a-z]*\n?", "", raw).strip()
            result = json.loads(raw_clean)

            agent_label      = result.get("label")
            agent_confidence = result.get("confidence", "low")
            agent_notes      = result.get("notes", "")

            # Validate label
            if agent_label is not None:
                try:
                    agent_label = int(agent_label)
                    if agent_label not in range(8, 13):
                        agent_notes = f"out-of-range ({agent_label}); " + agent_notes
                        agent_label = None
                        agent_confidence = "low"
                except (ValueError, TypeError):
                    agent_label = None
                    agent_confidence = "low"
                    agent_notes = f"parse error: {raw[:60]}"

        except Exception as e:
            agent_label      = None
            agent_confidence = "low"
            agent_notes      = f"api error: {str(e)[:80]}"

        labeled.append({
            **p,
            "agent_label":      agent_label,
            "agent_confidence": agent_confidence,
            "agent_notes":      agent_notes,
        })

        if (i + 1) % 10 == 0 or i == 0:
            readable = sum(1 for r in labeled if r["agent_label"] is not None)
            print(f"  {i+1}/{len(patches)} labeled  ({readable} readable so far)")

    return labeled


def print_summary(labeled):
    """Print label distribution and confidence breakdown."""
    from collections import Counter
    labels = [r["agent_label"] for r in labeled]
    confs  = [r["agent_confidence"] for r in labeled]

    label_counts = Counter(labels)
    conf_counts  = Counter(confs)

    print("\n── Label distribution ──────────────────────────")
    for v in sorted(k for k in label_counts if k is not None):
        print(f"  {v:>2}: {label_counts[v]:>3} frames")
    if label_counts[None]:
        print(f"  null (unreadable): {label_counts[None]}")

    print("\n── Confidence breakdown ────────────────────────")
    for c in ["high", "medium", "low"]:
        print(f"  {c:<6}: {conf_counts.get(c, 0):>3}")

    high_readable = sum(1 for r in labeled
                        if r["agent_label"] is not None and r["agent_confidence"] == "high")
    print(f"\n  High-confidence readable frames: {high_readable}/{len(labeled)}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Sample 100 frames, extract bot-count ROI patches, label with Claude vision")
    ap.add_argument("video", help="Path to gameplay .mp4")
    ap.add_argument("--n-samples", type=int, default=N_SAMPLES,
                    help=f"Number of frames to sample (default: {N_SAMPLES})")
    ap.add_argument("--out-dir", default="outputs",
                    help="Directory for patch images and output CSV (default: outputs/)")
    ap.add_argument("--skip-labeling", action="store_true",
                    help="Only extract patches, skip Claude API labeling")
    args = ap.parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    stem       = os.path.splitext(os.path.basename(args.video))[0]
    patch_dir  = os.path.join(args.out_dir, f"{stem}_benchmark_patches")
    os.makedirs(patch_dir, exist_ok=True)

    # ── Phase A1: extract patches ──────────────────────────────────────────
    patches = extract_patches(args.video, args.n_samples, patch_dir)

    if args.skip_labeling:
        print(f"\n--skip-labeling set. Patches saved to {patch_dir}")
        print("Run without --skip-labeling to invoke Claude vision labeling.")
        return

    # ── Phase A2: agent labeling ───────────────────────────────────────────
    labeled = label_patches_with_agent(patches)

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_csv = os.path.join(args.out_dir, f"{stem}_benchmark_labels.csv")
    df = pd.DataFrame(labeled, columns=[
        "frame_n", "time_s", "patch_path",
        "agent_label", "agent_confidence", "agent_notes",
    ])
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    print_summary(labeled)


if __name__ == "__main__":
    main()
