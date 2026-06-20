#!/usr/bin/env python3
"""
Feature Engineering: Historical Lookup Table for ASTRAM Traffic Events

This script answers the question:
    "For events like THIS one (same corridor, cause, time-of-day, weekend?),
     what TYPICALLY happened in the past?"

WHY we use historical matching instead of raw ML predictions alone:
  - Many groups in this dataset are sparse (< 5 confirmed events).
  - A black-box model cannot explain its estimate to a police officer.
  - A lookup-based approach is fully transparent: you can say
    "Based on N past events of type X on corridor Y at morning-weekend,
     the median resolution time was Z minutes."
  - The fallback hierarchy makes the uncertainty explicit: a fine-grain match
    (specific corridor + cause) is more trustworthy than a coarse match
    (cause only). The `match_level` column tells downstream callers which
    tier of confidence applies.

Match hierarchy (most specific → least specific):
  Level 1 (FINE):       (corridor, event_cause, is_weekend, hour_bucket) — N >= 5
  Level 2 (COARSE):     (event_cause, is_weekend, hour_bucket)           — N >= 5
  Level 3 (CAUSE_ONLY): (event_cause only, ignoring time/weekend)        — handled in
                          impact_model.py by querying cleaned_events.csv directly.
  Level 4 (NO_DATA):    Event cause never seen before — return None.

The reliability threshold (MIN_EVENTS = 5) is config-driven so it can be
tuned without changing logic.
"""

import os
from typing import Optional, Tuple
import pandas as pd

# ─── Configuration (thresholds, paths) ─────────────────────────────────────────
CLEANED_DATA_PATH = os.path.join("data", "processed", "cleaned_events.csv")
LOOKUP_OUTPUT_PATH = os.path.join("data", "processed", "historical_lookup.csv")

# Minimum events in a group before we trust its statistics.
# Below this, the sample is too small to produce a reliable estimate.
MIN_EVENTS: int = 5

# Hour band definitions: maps hour_of_day (0-23) to a named bucket.
# 4 bands chosen to balance granularity vs. group sizes.
HOUR_BANDS: list[tuple[int, int, str]] = [
    (0,  6,  "night"),       # 00:00 – 05:59  (low-traffic overnight)
    (6,  12, "morning"),     # 06:00 – 11:59  (morning rush)
    (12, 18, "afternoon"),   # 12:00 – 17:59  (midday / afternoon)
    (18, 24, "evening"),     # 18:00 – 23:59  (evening rush + night)
]


def assign_hour_bucket(hour: int) -> str:
    """
    Map an hour-of-day integer (0-23) to a named time band.

    Args:
        hour: Integer hour in [0, 23].

    Returns:
        One of 'night', 'morning', 'afternoon', 'evening'.
    """
    for start, end, label in HOUR_BANDS:
        if start <= hour < end:
            return label
    # Fallback: should never be reached for valid hour values.
    return "evening"


def _road_closure_rate(series: pd.Series) -> float:
    """
    Compute the fraction of rows where requires_road_closure is True.

    Uses the full group (including has_duration=False rows), because
    road closure status is logged at event creation — it does not depend
    on knowing the resolution time.

    Args:
        series: Boolean Series of requires_road_closure values.

    Returns:
        Float in [0.0, 1.0].
    """
    if len(series) == 0:
        return float("nan")
    return series.sum() / len(series)


def _priority_mode(series: pd.Series) -> str:
    """
    Return the most common priority value in a group.

    Falls back to 'Unknown' if the series is empty.

    Args:
        series: String Series of priority values.

    Returns:
        Most frequent priority string, e.g. 'High' or 'Low'.
    """
    if len(series) == 0:
        return "Unknown"
    return series.mode().iloc[0]


def _aggregate_group(
    dur_rows: pd.DataFrame,
    all_rows: pd.DataFrame,
) -> dict:
    """
    Compute aggregate statistics for a single histogram bucket.

    Duration stats (mean, median, count) use only rows where has_duration=True,
    because rows without an effective_end_datetime cannot contribute a valid
    time measurement and would bias the distribution.

    Non-duration stats (road_closure_rate, priority_mode) use all rows,
    since those fields are populated at event creation time, not at resolution.

    Args:
        dur_rows: Subset of the group where has_duration=True.
        all_rows: The full group (has_duration True and False).

    Returns:
        Dict with keys: event_count, duration_count, mean_duration,
        median_duration, road_closure_rate, priority_mode.
    """
    n_duration = len(dur_rows)
    n_total = len(all_rows)

    mean_dur = dur_rows["duration_minutes"].mean() if n_duration > 0 else float("nan")
    median_dur = dur_rows["duration_minutes"].median() if n_duration > 0 else float("nan")

    return {
        # Total events in this group (used for the N displayed to officers)
        "event_count": n_total,
        # Number of events with a known duration (used internally for reliability)
        "duration_count": n_duration,
        "mean_duration_minutes": round(mean_dur, 2),
        "median_duration_minutes": round(median_dur, 2),
        "road_closure_rate": round(
            _road_closure_rate(all_rows["requires_road_closure"]), 4
        ),
        "priority_mode": _priority_mode(all_rows["priority"]),
    }


def build_fine_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the FINE-grain lookup table grouped by
    (corridor, event_cause, is_weekend, hour_bucket).

    A group is included only if its duration_count >= MIN_EVENTS.
    This is the primary lookup tier — the most specific match.

    Args:
        df: Cleaned events DataFrame with `has_duration`, `hour_bucket` columns.

    Returns:
        DataFrame with one row per reliable fine-grain group.
    """
    dur_df = df[df["has_duration"]]

    # Aggregate duration stats from has_duration=True rows only
    dur_agg = (
        dur_df
        .groupby(["corridor", "event_cause", "is_weekend", "hour_bucket"])
        .agg(
            duration_count=("duration_minutes", "count"),
            mean_duration_minutes=("duration_minutes", "mean"),
            median_duration_minutes=("duration_minutes", "median"),
        )
        .reset_index()
    )

    # Aggregate non-duration stats from ALL rows (both has_duration values)
    all_agg = (
        df
        .groupby(["corridor", "event_cause", "is_weekend", "hour_bucket"])
        .agg(
            event_count=("id", "count"),
            road_closure_rate=("requires_road_closure", "mean"),   # mean of bool = fraction True
            priority_mode=("priority", lambda s: s.mode().iloc[0] if len(s) > 0 else "Unknown"),
        )
        .reset_index()
    )

    # Merge duration stats onto the full-count table
    fine = all_agg.merge(
        dur_agg,
        on=["corridor", "event_cause", "is_weekend", "hour_bucket"],
        how="left",
    )

    # Round for readability
    fine["mean_duration_minutes"] = fine["mean_duration_minutes"].round(2)
    fine["median_duration_minutes"] = fine["median_duration_minutes"].round(2)
    fine["road_closure_rate"] = fine["road_closure_rate"].round(4)

    # Keep only groups with enough historical duration observations
    # REASON: fewer than MIN_EVENTS duration observations means our mean/median
    # could be dominated by a single outlier.  We trust coarse fallback more.
    fine = fine[fine["duration_count"] >= MIN_EVENTS].copy()
    fine["match_level"] = "fine"

    return fine


def build_coarse_lookup(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the COARSE-grain lookup table grouped by
    (event_cause, is_weekend, hour_bucket).

    This is the fallback tier: corridor information is dropped, giving
    broader coverage at the cost of less specificity.

    Args:
        df: Cleaned events DataFrame with `has_duration`, `hour_bucket` columns.

    Returns:
        DataFrame with one row per coarse group (all groups, regardless of count).
        The `match_level` and `duration_count` columns allow callers to decide
        if the coarse match itself is reliable.
    """
    dur_df = df[df["has_duration"]]

    dur_agg = (
        dur_df
        .groupby(["event_cause", "is_weekend", "hour_bucket"])
        .agg(
            duration_count=("duration_minutes", "count"),
            mean_duration_minutes=("duration_minutes", "mean"),
            median_duration_minutes=("duration_minutes", "median"),
        )
        .reset_index()
    )

    all_agg = (
        df
        .groupby(["event_cause", "is_weekend", "hour_bucket"])
        .agg(
            event_count=("id", "count"),
            road_closure_rate=("requires_road_closure", "mean"),
            priority_mode=("priority", lambda s: s.mode().iloc[0] if len(s) > 0 else "Unknown"),
        )
        .reset_index()
    )

    coarse = all_agg.merge(
        dur_agg,
        on=["event_cause", "is_weekend", "hour_bucket"],
        how="left",
    )

    coarse["mean_duration_minutes"] = coarse["mean_duration_minutes"].round(2)
    coarse["median_duration_minutes"] = coarse["median_duration_minutes"].round(2)
    coarse["road_closure_rate"] = coarse["road_closure_rate"].round(4)

    # Mark all coarse rows — the lookup function will filter by MIN_EVENTS before returning
    coarse["match_level"] = "coarse"

    # Add corridor=None placeholder so fine and coarse can be stacked if needed
    coarse.insert(0, "corridor", None)

    return coarse


def build_lookup_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine fine and coarse lookup tables into a single saved CSV.

    The saved file contains both tiers. The `match_level` column distinguishes
    them. Downstream callers use `get_historical_match()` which applies the
    fallback logic at query time, so no hierarchical merging is needed here.

    Args:
        df: Cleaned events DataFrame.

    Returns:
        Combined DataFrame (fine rows followed by coarse rows).
    """
    # Build both tiers
    fine = build_fine_lookup(df)
    coarse = build_coarse_lookup(df)

    # Concatenate — keep both tiers in one file for transparency / auditability
    combined = pd.concat([fine, coarse], ignore_index=True)
    return combined


def get_historical_match(
    corridor: str,
    event_cause: str,
    is_weekend: bool,
    hour_bucket: str,
    lookup_df: pd.DataFrame,
) -> Tuple[Optional[pd.Series], str]:
    """
    Look up historical statistics for a given event profile.

    Fallback logic (most specific → least specific):
    ─────────────────────────────────────────────────
    1. Try FINE match: corridor + event_cause + is_weekend + hour_bucket,
       where duration_count >= MIN_EVENTS.
       → If found: return it with match_level='fine'.

    2. Try COARSE match: event_cause + is_weekend + hour_bucket (ignores corridor),
       where duration_count >= MIN_EVENTS.
       → If found: return it with match_level='coarse'.
       WHY COARSE? When a specific corridor has few historical events of that
       cause/time, we still know something useful from city-wide patterns.

    3. No reliable match at fine/coarse level: return (None, 'none').
       Callers (e.g. impact_model.py) should then attempt a cause_only
       fallback before giving up entirely.

    Args:
        corridor:    Corridor name, e.g. 'Mysore Road' or 'Non-corridor'.
        event_cause: Event cause string, e.g. 'vehicle_breakdown'.
        is_weekend:  True if the event starts on a Saturday or Sunday.
        hour_bucket: One of 'night', 'morning', 'afternoon', 'evening'.
        lookup_df:   The historical lookup table DataFrame.

    Returns:
        Tuple of (matched_row_or_None, match_level_string).
        match_level is one of: 'fine', 'coarse', 'none'.
        'none' means the caller should try deeper fallbacks (cause_only).
    """
    # ── Attempt 1: Fine match ─────────────────────────────────────────────────
    fine_mask = (
        (lookup_df["match_level"] == "fine") &
        (lookup_df["corridor"] == corridor) &
        (lookup_df["event_cause"] == event_cause) &
        (lookup_df["is_weekend"] == is_weekend) &
        (lookup_df["hour_bucket"] == hour_bucket)
    )
    fine_hits = lookup_df[fine_mask]

    if len(fine_hits) > 0:
        # Fine match found — the most trustworthy result
        return fine_hits.iloc[0], "fine"

    # ── Attempt 2: Coarse match ───────────────────────────────────────────────
    # We drop the corridor constraint because:
    #   (a) this corridor has < MIN_EVENTS for this cause/time combination, or
    #   (b) the corridor wasn't in the training data at all (new road).
    # City-wide statistics for the cause/time still provide useful priors.
    coarse_mask = (
        (lookup_df["match_level"] == "coarse") &
        (lookup_df["event_cause"] == event_cause) &
        (lookup_df["is_weekend"] == is_weekend) &
        (lookup_df["hour_bucket"] == hour_bucket) &
        (lookup_df["duration_count"] >= MIN_EVENTS)
    )
    coarse_hits = lookup_df[coarse_mask]

    if len(coarse_hits) > 0:
        return coarse_hits.iloc[0], "coarse"

    # ── Attempt 3: Fine/coarse lookup exhausted ───────────────────────────────
    # Signal 'none' so impact_model.py can attempt a cause_only fallback
    # by querying cleaned_events.csv directly (ignoring time/weekend/corridor).
    return None, "none"


def main() -> None:
    """
    Main function: load cleaned events, build the lookup table, save to CSV.
    """
    print("Starting feature engineering: building historical lookup table...")

    # Load cleaned events
    if not os.path.exists(CLEANED_DATA_PATH):
        raise FileNotFoundError(
            f"Cleaned events file not found at: {CLEANED_DATA_PATH}\n"
            "Run src/data_cleaning.py first."
        )

    df = pd.read_csv(CLEANED_DATA_PATH)
    print(f"Loaded {len(df)} cleaned events.")

    # Assign hour_bucket from the pre-computed hour_of_day column
    print("Assigning hour buckets (night/morning/afternoon/evening)...")
    df["hour_bucket"] = df["hour_of_day"].apply(assign_hour_bucket)

    # Coerce types that may have been serialised differently by CSV round-trip
    df["is_weekend"] = df["is_weekend"].astype(bool)
    df["has_duration"] = df["has_duration"].astype(bool)
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)

    # Build lookup table
    print(f"Building lookup table (reliability threshold: N >= {MIN_EVENTS} duration events per group)...")
    lookup_df = build_lookup_table(df)

    # Breakdown stats for transparency
    fine_count = (lookup_df["match_level"] == "fine").sum()
    coarse_count = (lookup_df["match_level"] == "coarse").sum()
    print(f"  Fine-grain groups (corridor-specific, N>={MIN_EVENTS}): {fine_count}")
    print(f"  Coarse groups     (cause-level, all counts stored)   : {coarse_count}")

    # Save
    os.makedirs(os.path.dirname(LOOKUP_OUTPUT_PATH), exist_ok=True)
    lookup_df.to_csv(LOOKUP_OUTPUT_PATH, index=False)
    print(f"Lookup table saved to: {LOOKUP_OUTPUT_PATH}")

    # Demonstrate the lookup function on a sample query
    print("\n--- Sample lookup query ---")
    sample_corridor = "Mysore Road"
    sample_cause = "vehicle_breakdown"
    sample_weekend = False
    sample_bucket = "morning"
    row, level = get_historical_match(
        sample_corridor, sample_cause, sample_weekend, sample_bucket, lookup_df
    )
    if row is not None:
        print(f"  Query: corridor={sample_corridor!r}, cause={sample_cause!r}, "
              f"weekend={sample_weekend}, hour_bucket={sample_bucket!r}")
        print(f"  Match level: {level}")
        print(f"  Event count (all):   {row['event_count']}")
        print(f"  Duration count:      {row['duration_count']}")
        print(f"  Median duration:     {row['median_duration_minutes']} min")
        print(f"  Road closure rate:   {row['road_closure_rate']*100:.1f}%")
        print(f"  Priority mode:       {row['priority_mode']}")
    else:
        print("  No reliable historical match found (match_level='none').")

    print("\nFeature engineering complete.")


if __name__ == "__main__":
    main()
