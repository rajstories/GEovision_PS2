#!/usr/bin/env python3
"""
Impact Model for ASTRAM Traffic Events

Given a NEW upcoming or just-reported event, this module predicts its likely
traffic impact: how long it will last, whether road closure is likely, and
an overall severity classification.

WHY THIS IS NOT A BLACK-BOX ML MODEL:
  Traffic police officers must explain their decisions to the public and
  department leadership. An estimate of "High severity — based on 80 past
  vehicle breakdowns on Mysore Road on weekday mornings, median resolution
  48 min" is actionable and auditable. A neural network output of 0.87 is not.

HOW THE FALLBACK CHAIN WORKS:
  We try increasingly broad historical buckets until we find enough data:

  Tier 1 — FINE      : exact (corridor + cause + is_weekend + hour_bucket)
  Tier 2 — COARSE    : city-wide (cause + is_weekend + hour_bucket), drop corridor
  Tier 3 — CAUSE_ONLY: drop time of day and weekend, just use cause type.
            WHY? High-stakes rare events (vip_movement, protest, procession)
            will almost never have enough data for tiers 1 or 2, but they
            exist in the dataset and we can still say *something* meaningful.
  Tier 4 — NO_DATA   : event cause has truly zero history. Return explicit
            "no_data" rather than fabricating a number.

SEVERITY SCORING — TWO MODES (both config-driven):
  When evidence_count >= RELIABLE_SAMPLE_THRESHOLD:
    → Duration-primary mode: use median duration thresholds (SEVERITY_CONFIG).
      Duration is a well-estimated statistic when we have many samples.
  When evidence_count < RELIABLE_SAMPLE_THRESHOLD (i.e. cause_only or thin data):
    → Closure/priority-primary mode: use LOW_SAMPLE_SEVERITY_CONFIG.
      With fewer than ~5 duration data points, median duration is highly
      volatile — one administrative-lag record (e.g. a VIP event marked
      "closed" 12 days later) can completely dominate the estimate.
      Road closure rate and priority mode are categorical signals that are
      robust even with 2–4 records: a single VIP event either closed a road
      or it didn't. So we weight those signals instead.
      We still display the raw duration figure for transparency, but flag it
      with duration_reliability="low" and explain this in the confidence note.
"""

import os
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple
import pandas as pd

# ─── File paths ───────────────────────────────────────────────────────────────
LOOKUP_PATH = os.path.join("data", "processed", "historical_lookup.csv")
CLEANED_PATH = os.path.join("data", "processed", "cleaned_events.csv")

# ─── Sample-size threshold: below this, duration is not trustworthy ───────────
# When fewer than this many duration records exist, we switch to the
# closure/priority-primary severity mode. 5 matches the MIN_EVENTS used
# in feature_engineering.py for the same reason (one outlier in N=2 is 50%
# of the distribution; in N=5 it's 20% — tolerable for a rough estimate).
RELIABLE_SAMPLE_THRESHOLD: int = 5

# ─── Duration-primary severity config (used when sample is reliable) ──────────
# Structure: list of (upper_bound_minutes_exclusive, tier_label).
# Thresholds derived from dataset duration percentiles:
#   p25 ≈ 22 min | p50 ≈ 50 min | p75 ≈ 100 min
SEVERITY_CONFIG: list[tuple[float, str]] = [
    (50.0,     "Low"),     # < 50 min   — short disruption, self-resolving
    (120.0,    "Medium"),  # 50–120 min — needs attention, likely manageable
    (math.inf, "High"),    # > 120 min  — significant, may need manpower/diversion
]

# ─── Closure/priority-primary severity config (used when sample is thin) ──────
# When duration is unreliable (< RELIABLE_SAMPLE_THRESHOLD records), we
# derive severity from two robust categorical signals instead:
#
#   road_closure_rate (0.0–1.0) × closure_weight
#   + priority_score  (0.0–1.0, derived from priority_mode) × priority_weight
#   → combined score in [0.0, 1.0] → mapped to tier via CLOSURE_SCORE_THRESHOLDS
#
# Weights: closure matters more than priority (closure has direct safety impact)
CLOSURE_WEIGHT: float = 0.7
PRIORITY_WEIGHT: float = 0.3

# Score → tier mapping (scores are in [0.0, 1.0])
CLOSURE_SCORE_THRESHOLDS: list[tuple[float, str]] = [
    (0.35, "Low"),     # low closure probability and low priority
    (0.65, "Medium"),  # moderate signal — some events of this type close roads
    (1.01, "High"),    # strong signal — majority require closure or are high priority
]

# Priority → numeric score mapping (for the combined score calculation)
PRIORITY_SCORE_MAP: dict[str, float] = {
    "High":    1.0,
    "Low":     0.2,
    "Unknown": 0.5,  # neutral when unknown
}

# Road closure bump: if the incoming event explicitly states requires_road_closure=True,
# bump severity up one tier regardless of computed score.
# (This applies in both scoring modes.)
CLOSURE_SEVERITY_BUMP: bool = True  # set False to disable

# ─── Confidence thresholds ────────────────────────────────────────────────────
HIGH_CONFIDENCE_MIN_COUNT: int = 10  # fine/coarse with >= this many events

# ─── Hour band definitions (must match feature_engineering.py) ────────────────
HOUR_BANDS: list[tuple[int, int, str]] = [
    (0,  6,  "night"),
    (6,  12, "morning"),
    (12, 18, "afternoon"),
    (18, 24, "evening"),
]


# ─── Output dataclass ─────────────────────────────────────────────────────────
@dataclass
class ImpactPrediction:
    """
    The structured output of predict_impact().

    All numeric fields are None if the match level is 'no_data'.
    severity_tier is 'Unknown' for no_data.

    Fields
    ------
    expected_duration_minutes : Median duration of similar past events (None = unknown).
    road_closure_probability  : Historical fraction of similar events with road closure.
    severity_tier             : 'Low' / 'Medium' / 'High' / 'Unknown'.
    evidence_count            : Total historical events this estimate is based on.
    duration_count            : Number of events WITH a known duration (used for
                                reliability assessment; may be < evidence_count).
    duration_reliability      : 'high' if duration_count >= RELIABLE_SAMPLE_THRESHOLD,
                                'low' otherwise. When 'low', severity is driven by
                                closure_rate + priority, not duration.
    match_level               : 'fine' | 'coarse' | 'cause_only' | 'no_data'.
    confidence_label          : Human-readable confidence explanation.
    severity_basis            : 'duration' or 'closure_priority' — explains which
                                signals drove the severity_tier classification.
    notes                     : List of extra context strings (e.g. bump reasons,
                                reliability warnings).
    """
    expected_duration_minutes: Optional[float]
    road_closure_probability: Optional[float]
    severity_tier: str
    evidence_count: int
    duration_count: int
    duration_reliability: str           # 'high' | 'low'
    match_level: str
    confidence_label: str
    severity_basis: str                 # 'duration' | 'closure_priority' | 'unknown'
    notes: list[str] = field(default_factory=list)

    def display(self) -> None:
        """Print the prediction in a clearly labelled human-readable format."""
        print("-" * 56)
        print(f"  Match level           : {self.match_level}")
        print(f"  Confidence            : {self.confidence_label}")
        print(f"  Evidence count        : {self.evidence_count} historical events")
        print(f"  Duration records      : {self.duration_count} "
              f"(reliability: {self.duration_reliability})")
        if self.expected_duration_minutes is not None:
            dur_flag = " [LOW RELIABILITY]" if self.duration_reliability == "low" else ""
            print(f"  Expected duration     : {self.expected_duration_minutes:.0f} min"
                  f"{dur_flag}")
        else:
            print("  Expected duration     : UNKNOWN")
        if self.road_closure_probability is not None:
            print(f"  Road closure prob.    : {self.road_closure_probability*100:.1f}%")
        else:
            print("  Road closure prob.    : UNKNOWN")
        print(f"  Severity tier         : {self.severity_tier}")
        print(f"  Severity basis        : {self.severity_basis}")
        for note in self.notes:
            print(f"  NOTE: {note}")
        print("-" * 56)


# ─── Helper functions ─────────────────────────────────────────────────────────

def assign_hour_bucket(hour: int) -> str:
    """Map hour 0-23 to a named time band (matches feature_engineering.py)."""
    for start, end, label in HOUR_BANDS:
        if start <= hour < end:
            return label
    return "evening"  # safety fallback


def _bump_severity(base_tier: str, notes: list[str]) -> str:
    """
    Elevate base_tier by one step if CLOSURE_SEVERITY_BUMP is enabled.

    This is used when the incoming event explicitly specifies road closure.
    Even a short road closure causes disproportionate disruption at junctions.

    Args:
        base_tier: Current tier before bump ('Low', 'Medium', 'High').
        notes:     Mutable list to append explanation to.

    Returns:
        Potentially elevated tier string.
    """
    tier_order = ["Low", "Medium", "High"]
    if base_tier not in tier_order:
        return base_tier
    idx = tier_order.index(base_tier)
    bumped = tier_order[min(idx + 1, len(tier_order) - 1)]
    if bumped != base_tier:
        notes.append(
            f"Severity bumped {base_tier} -> {bumped} "
            "because requires_road_closure=True was specified."
        )
    return bumped


def derive_severity_duration_primary(
    duration_minutes: float,
    requires_road_closure_input: Optional[bool],
) -> Tuple[str, str, list[str]]:
    """
    Duration-primary severity scoring (used when duration_count >= threshold).

    Looks up the duration in SEVERITY_CONFIG, then optionally bumps
    by one tier if the incoming event has explicit road closure.

    Args:
        duration_minutes:            Predicted median duration.
        requires_road_closure_input: Whether caller flagged road closure.

    Returns:
        Tuple of (severity_tier, severity_basis, notes_list).
    """
    notes: list[str] = []

    base_tier = "Unknown"
    for upper_bound, label in SEVERITY_CONFIG:
        if duration_minutes < upper_bound:
            base_tier = label
            break

    if CLOSURE_SEVERITY_BUMP and requires_road_closure_input is True:
        base_tier = _bump_severity(base_tier, notes)

    return base_tier, "duration", notes


def derive_severity_closure_priority(
    road_closure_prob: Optional[float],
    priority_mode: str,
    requires_road_closure_input: Optional[bool],
    duration_count: int,
) -> Tuple[str, str, list[str]]:
    """
    Closure/priority-primary severity scoring (used when sample is thin).

    WHY: When duration_count < RELIABLE_SAMPLE_THRESHOLD, the median duration
    is heavily influenced by even a single outlier. For example, 2 VIP movement
    records with administrative-lag closures of 12 and 15 days produce a median
    of ~13 days, which is not representative of actual road disruption.

    Instead, we use road_closure_rate and priority_mode, which are categorical
    signals present at event creation time and are far less distorted by
    administrative lag. A VIP movement either required road closure or it didn't.

    The combined score formula:
        score = closure_rate * CLOSURE_WEIGHT + priority_score * PRIORITY_WEIGHT

    Both inputs are in [0, 1], so the combined score is in [0, 1]. This is
    deliberately NOT a trained ML model — the formula and weights are visible
    and adjustable by any team member.

    Args:
        road_closure_prob:           Historical road closure fraction (0-1).
        priority_mode:               Most common priority label, e.g. 'High'.
        requires_road_closure_input: Whether caller flagged road closure.
        duration_count:              Number of duration records (used in note text).

    Returns:
        Tuple of (severity_tier, severity_basis, notes_list).
    """
    notes: list[str] = []

    # Explain why duration was deprioritised
    notes.append(
        f"Duration estimate is based on only {duration_count} historical "
        "record(s) and may be unreliable due to administrative-lag closures. "
        "Severity is based primarily on road-closure likelihood and event "
        "priority for this reason."
    )

    # If neither signal is available, fall back to Unknown
    if road_closure_prob is None:
        if CLOSURE_SEVERITY_BUMP and requires_road_closure_input is True:
            notes.append("Road closure confirmed by operator — defaulting to Medium.")
            return "Medium", "closure_priority", notes
        return "Unknown", "closure_priority", notes

    # Build composite score
    closure_score = float(road_closure_prob)
    priority_score = PRIORITY_SCORE_MAP.get(str(priority_mode), 0.5)
    combined = closure_score * CLOSURE_WEIGHT + priority_score * PRIORITY_WEIGHT

    # Map combined score to tier
    base_tier = "Unknown"
    for upper, label in CLOSURE_SCORE_THRESHOLDS:
        if combined < upper:
            base_tier = label
            break

    notes.append(
        f"Closure/priority score: {combined:.2f} "
        f"(closure_rate={closure_score:.0%}, priority={priority_mode})"
    )

    if CLOSURE_SEVERITY_BUMP and requires_road_closure_input is True:
        base_tier = _bump_severity(base_tier, notes)

    return base_tier, "closure_priority", notes


def build_confidence_label(
    match_level: str,
    evidence_count: int,
    duration_reliability: str,
) -> str:
    """
    Map match level, evidence count, and duration reliability to a confidence label.

    Args:
        match_level:          'fine' | 'coarse' | 'cause_only' | 'no_data'.
        evidence_count:       Total historical events used.
        duration_reliability: 'high' | 'low'.

    Returns:
        Human-readable confidence string.
    """
    if match_level == "no_data":
        return "Insufficient data — defer to officer judgment"

    if match_level == "cause_only":
        return (
            "Low confidence — cause-only match "
            f"({evidence_count} historical events; duration reliability: {duration_reliability})"
        )

    if match_level in ("fine", "coarse"):
        if evidence_count >= HIGH_CONFIDENCE_MIN_COUNT:
            return "High confidence"
        else:
            return "Moderate confidence"

    return "Unknown confidence"


def cause_only_fallback(
    event_cause: str,
    cleaned_df: pd.DataFrame,
) -> Tuple[Optional[float], Optional[float], int, int, str]:
    """
    Compute aggregate stats for event_cause, ignoring time/corridor/weekend.

    This is Tier 3 in the fallback chain. It is invoked when both the fine
    and coarse lookups return 'none' — typically for rare causes (vip_movement,
    protest, procession) that don't have enough rows in any single time bucket.

    Uses MEDIAN duration (not mean) throughout, consistent with feature_engineering.py.
    Median is more robust to the administrative-lag outliers described in the
    module docstring.

    Args:
        event_cause: The cause string to filter on.
        cleaned_df:  Full cleaned events DataFrame.

    Returns:
        Tuple of (median_duration_or_None, road_closure_rate_or_None,
                  total_event_count, duration_count, priority_mode).
        total_event_count is 0 if the cause has no historical rows at all.
    """
    cause_rows = cleaned_df[cleaned_df["event_cause"] == event_cause]
    n_total = len(cause_rows)

    if n_total == 0:
        return None, None, 0, 0, "Unknown"

    # Duration stats from has_duration=True rows only — same rule as feature_engineering.py
    dur_rows = cause_rows[cause_rows["has_duration"] == True]
    n_dur = len(dur_rows)
    # Median is chosen over mean: one 12-day administrative-lag record
    # would add ~17,000 minutes to a mean of 2 values, but would "only"
    # be the middle value of a 3-value sorted list, still bad but less explosive.
    median_dur = dur_rows["duration_minutes"].median() if n_dur > 0 else None

    # Road closure rate and priority mode from all rows
    rc_rate = cause_rows["requires_road_closure"].mean()
    priority_mode = (
        cause_rows["priority"].mode().iloc[0]
        if len(cause_rows) > 0 else "Unknown"
    )

    return (
        round(float(median_dur), 2) if median_dur is not None else None,
        round(float(rc_rate), 4),
        n_total,
        n_dur,
        str(priority_mode),
    )


# ─── Main prediction function ─────────────────────────────────────────────────

def predict_impact(
    event_cause: str,
    planned_start_datetime: datetime,
    corridor: Optional[str] = None,
    requires_road_closure: Optional[bool] = None,
    lookup_df: Optional[pd.DataFrame] = None,
    cleaned_df: Optional[pd.DataFrame] = None,
) -> ImpactPrediction:
    """
    Predict the traffic impact of a new or upcoming event.

    Parameters
    ----------
    event_cause : str
        The type of event (e.g. 'vehicle_breakdown', 'vip_movement').
    planned_start_datetime : datetime
        When the event is expected to start. Naive datetimes are assumed IST.
    corridor : str, optional
        Named road corridor, e.g. 'Mysore Road'. Pass None or 'Non-corridor'
        if unknown.
    requires_road_closure : bool, optional
        Whether the event requires road closure. None means unknown at this time.
    lookup_df : pd.DataFrame, optional
        Pre-loaded historical lookup table. Loaded from LOOKUP_PATH if None.
    cleaned_df : pd.DataFrame, optional
        Pre-loaded cleaned events. Loaded from CLEANED_PATH if None
        (only needed for cause_only fallback).

    Returns
    -------
    ImpactPrediction dataclass with all output fields populated.
    """
    # ── Lazy-load DataFrames (callers can pre-load and pass them for performance)
    if lookup_df is None:
        if not os.path.exists(LOOKUP_PATH):
            raise FileNotFoundError(
                f"Lookup table not found at {LOOKUP_PATH}. "
                "Run src/feature_engineering.py first."
            )
        lookup_df = pd.read_csv(LOOKUP_PATH)

    if cleaned_df is None:
        if not os.path.exists(CLEANED_PATH):
            raise FileNotFoundError(
                f"Cleaned events not found at {CLEANED_PATH}. "
                "Run src/data_cleaning.py first."
            )
        cleaned_df = pd.read_csv(CLEANED_PATH)
        cleaned_df["has_duration"] = cleaned_df["has_duration"].astype(bool)
        cleaned_df["requires_road_closure"] = cleaned_df["requires_road_closure"].astype(bool)

    # ── Derive time features from start datetime ──────────────────────────────
    # Convert to IST (UTC+5:30) consistent with data_cleaning.py derivations.
    ist_offset = pd.Timedelta("5h30min")
    if planned_start_datetime.tzinfo is None:
        start_ist = planned_start_datetime  # assume IST if naive
    else:
        start_utc = planned_start_datetime.astimezone(timezone.utc)
        start_ist = pd.Timestamp(start_utc) + ist_offset

    hour_of_day = start_ist.hour
    day_of_week = start_ist.weekday()  # 0=Mon, 6=Sun
    is_weekend = day_of_week in (5, 6)
    hour_bucket = assign_hour_bucket(hour_of_day)

    effective_corridor = corridor if corridor else "Non-corridor"

    from feature_engineering import get_historical_match

    # ── Tiers 1 & 2: Fine and coarse lookup ───────────────────────────────────
    row, match_level = get_historical_match(
        effective_corridor, event_cause, is_weekend, hour_bucket, lookup_df
    )

    # ── Tier 3: Cause-only fallback ───────────────────────────────────────────
    # Invoked when fine and coarse both return 'none' (typically rare causes).
    if match_level == "none":
        median_dur, rc_rate, n_total, n_dur, pmode = cause_only_fallback(
            event_cause, cleaned_df
        )

        if n_total == 0:
            # ── Tier 4: No data at all ────────────────────────────────────────
            # We refuse to fabricate. An honest "unknown" is more defensible
            # in front of a review board than a hallucinated confidence interval.
            return ImpactPrediction(
                expected_duration_minutes=None,
                road_closure_probability=None,
                severity_tier="Unknown",
                evidence_count=0,
                duration_count=0,
                duration_reliability="low",
                match_level="no_data",
                confidence_label=build_confidence_label("no_data", 0, "low"),
                severity_basis="unknown",
                notes=["Event cause has no historical records. Officer judgment required."],
            )

        # Build a row-like structure consistent with the lookup table format
        row = pd.Series({
            "median_duration_minutes": median_dur,
            "road_closure_rate": rc_rate,
            "event_count": n_total,
            "duration_count": n_dur,
            "priority_mode": pmode,
        })
        match_level = "cause_only"

    # ── Extract stats from the matched row ────────────────────────────────────
    median_dur_val: Optional[float] = (
        float(row["median_duration_minutes"])
        if pd.notna(row.get("median_duration_minutes"))
        else None
    )
    rc_rate_val: Optional[float] = (
        float(row["road_closure_rate"])
        if pd.notna(row.get("road_closure_rate"))
        else None
    )
    evidence_count: int = int(row.get("event_count", 0))
    # duration_count may be NaN in lookup rows that had no duration data at all
    raw_dur_count = row.get("duration_count")
    duration_count: int = int(raw_dur_count) if pd.notna(raw_dur_count) else 0
    priority_mode: str = str(row.get("priority_mode", "Unknown"))

    # ── Decide duration reliability and scoring mode ──────────────────────────
    # We use duration_count (not total evidence_count) because it is the number
    # of records that actually contributed to the median_duration estimate.
    # A group can have 20 total events but only 3 with resolved datetimes —
    # those 3 drive the duration figure, and 3 is too few to be trustworthy.
    duration_reliability = (
        "high" if duration_count >= RELIABLE_SAMPLE_THRESHOLD else "low"
    )

    # ── Derive severity using the appropriate mode ────────────────────────────
    if duration_reliability == "high" and median_dur_val is not None:
        # Sufficient duration data — let median duration drive the tier
        severity, sev_basis, sev_notes = derive_severity_duration_primary(
            median_dur_val, requires_road_closure
        )
    else:
        # Too few duration records — switch to closure/priority primary
        # This is the key design decision: avoid letting 2 admin-lag records
        # project a 6-day severity estimate to the judge.
        severity, sev_basis, sev_notes = derive_severity_closure_priority(
            rc_rate_val, priority_mode, requires_road_closure, duration_count
        )

    confidence = build_confidence_label(match_level, evidence_count, duration_reliability)

    return ImpactPrediction(
        expected_duration_minutes=median_dur_val,
        road_closure_probability=rc_rate_val,
        severity_tier=severity,
        evidence_count=evidence_count,
        duration_count=duration_count,
        duration_reliability=duration_reliability,
        match_level=match_level,
        confidence_label=confidence,
        severity_basis=sev_basis,
        notes=sev_notes,
    )


# ─── Demo / __main__ block ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Pre-load DataFrames once — all 4 examples share the same objects
    print("Loading lookup table and cleaned events...")
    _lookup = pd.read_csv(LOOKUP_PATH)
    _cleaned = pd.read_csv(CLEANED_PATH)
    _cleaned["has_duration"] = _cleaned["has_duration"].astype(bool)
    _cleaned["requires_road_closure"] = _cleaned["requires_road_closure"].astype(bool)

    print("=" * 56)
    print("IMPACT MODEL DEMO -- 4 example predictions")
    print("=" * 56)

    # ── [1] Fine match ────────────────────────────────────────────────────────
    # vehicle_breakdown on Mysore Road, weekday morning.
    # 80 duration records → reliable sample → duration-primary severity.
    print("\n[1] Fine match | vehicle_breakdown | Mysore Road | Mon 08:00")
    r1 = predict_impact(
        event_cause="vehicle_breakdown",
        planned_start_datetime=datetime(2024, 4, 1, 8, 0, 0),   # Monday IST
        corridor="Mysore Road",
        requires_road_closure=False,
        lookup_df=_lookup,
        cleaned_df=_cleaned,
    )
    r1.display()

    # ── [2] Coarse match ──────────────────────────────────────────────────────
    # vehicle_breakdown on a brand-new road not in the dataset.
    # Falls through to coarse (cause + weekend + hour bucket), still large N.
    print("\n[2] Coarse match | vehicle_breakdown | New Peripheral Road | Sat 17:00")
    r2 = predict_impact(
        event_cause="vehicle_breakdown",
        planned_start_datetime=datetime(2024, 4, 6, 17, 0, 0),  # Saturday IST
        corridor="New Peripheral Ring Road",
        requires_road_closure=False,
        lookup_df=_lookup,
        cleaned_df=_cleaned,
    )
    r2.display()

    # ── [3] Cause-only fallback — VIP movement ────────────────────────────────
    # vip_movement on a weekend night.
    # Only 4 total records; 2 have 12-15 day durations (administrative lag).
    # duration_count=4 < RELIABLE_SAMPLE_THRESHOLD → closure/priority scoring.
    # Severity should NOT be driven by the noisy 8684-min median.
    # road_closure_rate = 0.0%, priority = High → expect Low or Medium.
    print("\n[3] Cause-only | vip_movement | CBD 2 | Sat 23:00 | road_closure=True")
    r3 = predict_impact(
        event_cause="vip_movement",
        planned_start_datetime=datetime(2024, 4, 6, 23, 0, 0),  # Sat IST
        corridor="CBD 2",
        requires_road_closure=True,
        lookup_df=_lookup,
        cleaned_df=_cleaned,
    )
    r3.display()

    # ── [4] Cause-only — protest with road closure ────────────────────────────
    # protest near Cubbon Park — 13 total records, 7 with duration.
    # duration_count=7 >= RELIABLE_SAMPLE_THRESHOLD → duration-primary.
    # road_closure=True should bump severity one step.
    print("\n[4] Cause-only | protest | Non-corridor | Tue 11:00 | road_closure=True")
    r4 = predict_impact(
        event_cause="protest",
        planned_start_datetime=datetime(2024, 4, 2, 11, 0, 0),  # Tuesday IST
        corridor="Non-corridor",
        requires_road_closure=True,
        lookup_df=_lookup,
        cleaned_df=_cleaned,
    )
    r4.display()
