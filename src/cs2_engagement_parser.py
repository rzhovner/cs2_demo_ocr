"""
CS2 Micro-Duel Engagement Parser
=================================
Parses a CS2 .dem file from a 90° 100-kill bot challenge and produces
a full engagement map CSV with per-shot detail and derived KPIs.

Usage:
    python cs2_engagement_parser.py <path_to_demo.dem> [--output engagements.csv] [--tick-rate 64]

Requirements:
    pip install demoparser2 pandas

Output columns:
    engagement_id       — sequential engagement number
    kill_tick           — tick when the bot died
    kill_time_s         — kill time in seconds from demo start
    engagement_start_s  — estimated start (prev kill time or first shot, whichever proxy works)
    ttff_ms             — time to first fire (first shot - engagement start) in ms
    ttk_ms              — time to kill (kill tick - first shot tick) in ms
    total_duration_ms   — engagement start to kill in ms
    shots_fired         — number of weapon_fire events in this engagement
    shots_hit           — number of player_hurt events attributed to this engagement
    shots_missed        — shots_fired - shots_hit
    hit_rate            — shots_hit / shots_fired
    headshot            — whether the killing blow was a headshot (True/False)
    damage_dealt        — total damage across all hits in this engagement
    weapon              — weapon used
    hit_sequence        — per-shot Y/M pattern (e.g., "M,Y,Y")
    shot_1 .. shot_N    — individual shot timestamps (seconds from demo start)
    shot_1_hit .. shot_N_hit — Y or M for each shot
    inter_engagement_gap_ms — time between previous kill and this engagement's first shot
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

try:
    from demoparser2 import DemoParser
except ImportError:
    print("=" * 60)
    print("ERROR: demoparser2 is not installed.")
    print()
    print("Install it with:")
    print("  pip install demoparser2")
    print()
    print("Requires Python 3.8+ on Windows/macOS/Linux.")
    print("=" * 60)
    sys.exit(1)


# ─── Configuration ───────────────────────────────────────────

# Max ms between a weapon_fire and a player_hurt to count as
# "that shot hit." Accounts for bullet travel time / tick jitter.
HIT_MATCH_WINDOW_TICKS = 8

# Max ticks between last kill and next shot to consider them
# part of separate engagements (otherwise might be same burst)
ENGAGEMENT_GAP_TICKS = 32

# Max number of shot columns in the output CSV
MAX_SHOT_COLUMNS = 15


def parse_demo(demo_path, tick_rate=None):
    """Parse demo and return raw event DataFrames."""
    print(f"[1/5] Loading demo: {demo_path}")
    parser = DemoParser(demo_path)

    # Detect tick rate from header if not specified
    if tick_rate is None:
        try:
            header = parser.parse_header()
            tick_rate = header.get("playback_ticks", 64) / max(header.get("playback_time", 1), 1)
            tick_rate = round(tick_rate)
        except Exception:
            tick_rate = 64
    print(f"       Tick rate: {tick_rate} ticks/sec")

    print("[2/5] Extracting events...")
    # Parse the three core event streams
    fire_events = parser.parse_events(["weapon_fire"])
    hurt_events = parser.parse_events(["player_hurt"])
    death_events = parser.parse_events(["player_death"])

    # Convert to DataFrames — new demoparser2 returns list of (event_name, df) tuples
    def _to_df(result):
        if isinstance(result, pd.DataFrame):
            return result
        if isinstance(result, list) and result and isinstance(result[0], tuple):
            return result[0][1]
        return pd.DataFrame(result)

    fire_events = _to_df(fire_events)
    hurt_events = _to_df(hurt_events)
    death_events = _to_df(death_events)

    print(f"       weapon_fire events: {len(fire_events)}")
    print(f"       player_hurt events: {len(hurt_events)}")
    print(f"       player_death events: {len(death_events)}")

    return fire_events, hurt_events, death_events, tick_rate


def identify_player(fire_events, death_events):
    """
    Identify the human player (most weapon_fire events = the one practicing).
    In a bot challenge, the human fires vastly more than any single bot.
    """
    # Try user_steamid first, fall back to user_name or attacker columns
    for col in ["user_steamid", "user_name", "attacker_steamid", "attacker_name"]:
        if col in fire_events.columns:
            counts = fire_events[col].value_counts()
            if counts.empty:
                continue
            player_id = counts.index[0]
            id_col = col
            print(f"[3/5] Identified player: {player_id} (via {col}, {counts.iloc[0]} shots)")
            return player_id, id_col

    # If we can't identify, use all events
    print("[3/5] Could not isolate player — using all events")
    return None, None


def filter_player_events(fire_events, hurt_events, death_events, player_id, id_col):
    """Filter events to only those involving the human player as attacker."""
    if player_id is None:
        return fire_events, hurt_events, death_events

    # Map column names — demoparser2 uses different prefixes per event type
    fire_col = id_col  # weapon_fire uses user_*
    hurt_col = id_col.replace("user_", "attacker_")  # player_hurt uses attacker_*
    death_col = id_col.replace("user_", "attacker_")  # player_death uses attacker_*

    if fire_col in fire_events.columns:
        fire_events = fire_events[fire_events[fire_col] == player_id].copy()
    if hurt_col in hurt_events.columns:
        hurt_events = hurt_events[hurt_events[hurt_col] == player_id].copy()
    if death_col in death_events.columns:
        death_events = death_events[death_events[death_col] == player_id].copy()

    return fire_events, hurt_events, death_events


def match_shots_to_hits(fire_ticks, hurt_ticks_set, window=HIT_MATCH_WINDOW_TICKS):
    """
    For each fire tick, determine if a hit occurred within the matching window.
    Returns list of (tick, "Y"/"M") tuples.
    """
    results = []
    used_hurt_ticks = set()

    for ft in fire_ticks:
        matched = False
        for offset in range(0, window + 1):
            candidate = ft + offset
            if candidate in hurt_ticks_set and candidate not in used_hurt_ticks:
                used_hurt_ticks.add(candidate)
                matched = True
                break
        results.append((ft, "Y" if matched else "M"))

    return results


def build_engagement_map(fire_events, hurt_events, death_events, tick_rate):
    """
    Build the full engagement map:
    - Use kill events as engagement boundaries
    - Attribute shots (weapon_fire) and hits (player_hurt) to each engagement
    - Compute per-engagement metrics
    """
    print("[4/5] Building engagement map...")

    # Get tick column name
    tick_col = "tick"

    # Sort all events by tick
    fire_ticks = sorted(fire_events[tick_col].values)
    kill_ticks = sorted(death_events[tick_col].values)

    # Build a set of hurt ticks for fast matching
    hurt_ticks = sorted(hurt_events[tick_col].values)
    hurt_ticks_set = set(hurt_ticks)

    # Also extract per-hurt metadata (damage, hitgroup) indexed by tick
    hurt_meta = {}
    if len(hurt_events) > 0:
        for _, row in hurt_events.iterrows():
            t = row[tick_col]
            dmg = row.get("dmg_health", row.get("damage", 0))
            hitgroup = row.get("hitgroup", -1)
            if t not in hurt_meta:
                hurt_meta[t] = {"damage": 0, "hitgroup": hitgroup}
            hurt_meta[t]["damage"] += dmg

    # Extract kill metadata (headshot, weapon)
    kill_meta = {}
    for _, row in death_events.iterrows():
        t = row[tick_col]
        hs = row.get("headshot", False)
        weapon = row.get("weapon", "unknown")
        kill_meta[t] = {"headshot": hs, "weapon": weapon}

    # ── Assign shots to engagements ──────────────────────────
    # Strategy: each engagement ends at a kill tick.
    # Shots belonging to engagement N are those between kill N-1 and kill N.

    engagements = []
    fire_idx = 0  # pointer into fire_ticks

    for eng_num, kill_tick in enumerate(kill_ticks, start=1):
        # Collect all fire ticks up to and including this kill tick
        # (shots that contributed to this kill)
        eng_fire_ticks = []
        while fire_idx < len(fire_ticks) and fire_ticks[fire_idx] <= kill_tick:
            eng_fire_ticks.append(fire_ticks[fire_idx])
            fire_idx += 1

        if not eng_fire_ticks:
            # Kill with no detected shots (knife, grenade, etc.) — skip or flag
            continue

        # Match each shot to a hit
        shot_results = match_shots_to_hits(eng_fire_ticks, hurt_ticks_set)

        # Compute metrics
        first_shot_tick = eng_fire_ticks[0]
        shots_fired = len(eng_fire_ticks)
        shots_hit = sum(1 for _, h in shot_results if h == "Y")
        shots_missed = shots_fired - shots_hit

        # Engagement start estimate: for first engagement, use first shot tick.
        # For subsequent, use previous kill tick (the moment the last bot died,
        # new target becomes available).
        prev_kill_tick = kill_ticks[eng_num - 2] if eng_num > 1 else None

        if prev_kill_tick is not None:
            engagement_start_tick = prev_kill_tick
            inter_gap_ticks = first_shot_tick - prev_kill_tick
        else:
            engagement_start_tick = first_shot_tick
            inter_gap_ticks = None

        # Time conversions
        ticks_to_ms = lambda t: (t / tick_rate) * 1000
        tick_to_s = lambda t: t / tick_rate

        ttff_ms = ticks_to_ms(first_shot_tick - engagement_start_tick)
        ttk_ms = ticks_to_ms(kill_tick - first_shot_tick)
        total_duration_ms = ticks_to_ms(kill_tick - engagement_start_tick)
        inter_gap_ms = ticks_to_ms(inter_gap_ticks) if inter_gap_ticks is not None else None

        # Damage total for this engagement
        total_damage = 0
        for _, row_tick in enumerate(eng_fire_ticks):
            for offset in range(0, HIT_MATCH_WINDOW_TICKS + 1):
                if row_tick + offset in hurt_meta:
                    total_damage += hurt_meta[row_tick + offset]["damage"]
                    break

        km = kill_meta.get(kill_tick, {})

        eng_data = {
            "engagement_id": eng_num,
            "kill_tick": kill_tick,
            "kill_time_s": round(tick_to_s(kill_tick), 3),
            "engagement_start_s": round(tick_to_s(engagement_start_tick), 3),
            "ttff_ms": round(ttff_ms, 1),
            "ttk_ms": round(ttk_ms, 1),
            "total_duration_ms": round(total_duration_ms, 1),
            "shots_fired": shots_fired,
            "shots_hit": shots_hit,
            "shots_missed": shots_missed,
            "hit_rate": round(shots_hit / shots_fired, 3) if shots_fired else 0,
            "headshot": km.get("headshot", False),
            "damage_dealt": total_damage,
            "weapon": km.get("weapon", "unknown"),
            "hit_sequence": ",".join(h for _, h in shot_results),
            "inter_engagement_gap_ms": round(inter_gap_ms, 1) if inter_gap_ms is not None else "",
        }

        # Add individual shot columns
        for i, (shot_tick, hit_flag) in enumerate(shot_results):
            col_num = i + 1
            if col_num <= MAX_SHOT_COLUMNS:
                eng_data[f"shot_{col_num}_time_s"] = round(tick_to_s(shot_tick), 3)
                eng_data[f"shot_{col_num}_hit"] = hit_flag

        engagements.append(eng_data)

    # Handle any remaining shots after the last kill (unfinished engagement)
    remaining_shots = fire_ticks[fire_idx:]
    if remaining_shots:
        print(f"       Note: {len(remaining_shots)} shots after last kill (unfinished engagement, excluded)")

    print(f"       Engagements detected: {len(engagements)}")
    return engagements


def compute_summary(engagements):
    """Compute aggregate KPIs from the engagement map."""
    df = pd.DataFrame(engagements)

    if df.empty:
        return {}

    summary = {
        "total_engagements": len(df),
        "total_shots_fired": df["shots_fired"].sum(),
        "total_shots_hit": df["shots_hit"].sum(),
        "overall_hit_rate": round(df["shots_hit"].sum() / max(df["shots_fired"].sum(), 1), 3),
        "headshot_kills": df["headshot"].sum(),
        "headshot_rate": round(df["headshot"].sum() / max(len(df), 1), 3),
        "median_ttff_ms": round(df["ttff_ms"].median(), 1),
        "mean_ttff_ms": round(df["ttff_ms"].mean(), 1),
        "std_ttff_ms": round(df["ttff_ms"].std(), 1),
        "min_ttff_ms": round(df["ttff_ms"].min(), 1),
        "max_ttff_ms": round(df["ttff_ms"].max(), 1),
        "median_ttk_ms": round(df["ttk_ms"].median(), 1),
        "mean_ttk_ms": round(df["ttk_ms"].mean(), 1),
        "median_shots_to_kill": round(df["shots_fired"].median(), 1),
        "mean_shots_to_kill": round(df["shots_fired"].mean(), 2),
        "one_tap_rate": round(len(df[df["shots_fired"] == 1]) / max(len(df), 1), 3),
        "mean_engagement_duration_ms": round(df["total_duration_ms"].mean(), 1),
    }

    # Inter-engagement pacing
    gaps = pd.to_numeric(df["inter_engagement_gap_ms"], errors="coerce").dropna()
    if len(gaps):
        summary["median_pacing_gap_ms"] = round(gaps.median(), 1)
        summary["mean_pacing_gap_ms"] = round(gaps.mean(), 1)

    # Kill streaks (consecutive kills with 1 shot)
    streak = 0
    max_streak = 0
    for _, row in df.iterrows():
        if row["shots_fired"] == 1:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    summary["max_one_tap_streak"] = max_streak

    # Consistency: coefficient of variation of TTFF
    if summary["mean_ttff_ms"] > 0:
        summary["ttff_consistency_cv"] = round(summary["std_ttff_ms"] / summary["mean_ttff_ms"], 3)

    return summary


def main():
    global MAX_SHOT_COLUMNS
    parser = argparse.ArgumentParser(
        description="CS2 Micro-Duel Engagement Parser — 90° Bot Challenge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cs2_engagement_parser.py my_demo.dem
  python cs2_engagement_parser.py my_demo.dem --output results.csv --tick-rate 128
  python cs2_engagement_parser.py my_demo.dem --summary-only
        """,
    )
    parser.add_argument("demo", help="Path to CS2 .dem file")
    parser.add_argument("--output", "-o", default=None, help="Output CSV path (default: <demo_name>_engagements.csv)")
    parser.add_argument("--tick-rate", "-t", type=int, default=None, help="Override tick rate (default: auto-detect)")
    parser.add_argument("--summary-only", action="store_true", help="Print summary KPIs only, skip CSV")
    parser.add_argument("--max-shots", type=int, default=MAX_SHOT_COLUMNS, help=f"Max shot columns in CSV (default: {MAX_SHOT_COLUMNS})")

    args = parser.parse_args()

    demo_path = Path(args.demo)
    if not demo_path.exists():
        print(f"ERROR: Demo file not found: {demo_path}")
        sys.exit(1)

    MAX_SHOT_COLUMNS = args.max_shots

    # Parse
    fire_events, hurt_events, death_events, tick_rate = parse_demo(str(demo_path), args.tick_rate)

    if fire_events.empty:
        print("ERROR: No weapon_fire events found in demo.")
        sys.exit(1)

    # Identify human player
    player_id, id_col = identify_player(fire_events, death_events)

    # Filter to player
    fire_events, hurt_events, death_events = filter_player_events(
        fire_events, hurt_events, death_events, player_id, id_col
    )

    # Build engagement map
    engagements = build_engagement_map(fire_events, hurt_events, death_events, tick_rate)

    if not engagements:
        print("ERROR: No engagements detected. Check if the demo contains kills.")
        sys.exit(1)

    # Summary
    summary = compute_summary(engagements)
    print("\n" + "=" * 60)
    print("  PERFORMANCE SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<35} {v}")
    print("=" * 60)

    # CSV output
    if not args.summary_only:
        output_path = args.output or str(demo_path.stem) + "_engagements.csv"
        df = pd.DataFrame(engagements)
        df.to_csv(output_path, index=False)
        print(f"\n[5/5] Engagement map saved: {output_path}")
        print(f"       {len(df)} engagements × {len(df.columns)} columns")
    else:
        print("\n[5/5] Summary-only mode — no CSV written.")

    print("\nDone.")


if __name__ == "__main__":
    main()
