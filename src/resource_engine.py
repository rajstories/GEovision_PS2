#!/usr/bin/env python3
"""
Resource Engine: Deployment Recommendations for Traffic Police

This module converts an ImpactPrediction (from src/impact_model.py) into a
concrete, actionable deployment recommendation: how many officers to deploy,
whether barricading is needed, and a plain-English rationale.

DESIGN PRINCIPLES:
  1. Config-driven, not hardcoded. Every threshold a commander might want
     to adjust lives in a named dict or constant at the top of this file —
     not buried in an if-else chain. A non-programmer can read and tune it.

  2. Rationale is audience-aware. When severity was assessed from closure/
     priority signals (because duration data is thin), the rationale says so
     explicitly. It never implies more certainty than the upstream model
     actually has.

  3. Honest about diversion limits. The system does not have a real-time
     road network graph. Rather than pretend to generate a diversion route,
     it states precisely what data would be needed and provides a hint
     toward standard BTP diversion practice for the corridor/cause type.

  4. Barricade hinting uses the best available location field. Junction >
     address > "near event start location" (graceful degradation).
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime

import pandas as pd

# ─── Personnel lookup table (config) ─────────────────────────────────────────
# Structure: keyed by (severity_tier, closure_required: bool)
# Values: base officer count.
# Edit these rows — do NOT change the function that reads them.
#
# Rationale for defaults:
#   Low / no closure    → 2 officers: one on scene, one for traffic direction
#   Low / closure       → 4 officers: extra hands needed for physical closure
#   Medium / no closure → 4 officers: longer event, shift handover likely
#   Medium / closure    → 6 officers: closure + active flow management
#   High / no closure   → 6 officers: prolonged disruption, manpower rotation
#   High / closure      → 10 officers: full road closure management protocol
#   Unknown / *         → 4 officers: safe middle-ground when uncertain
#
PERSONNEL_TABLE: dict[tuple[str, bool], int] = {
    ("Low",     False):  2,
    ("Low",     True):   4,
    ("Medium",  False):  4,
    ("Medium",  True):   6,
    ("High",    False):  6,
    ("High",    True):  10,
    ("Unknown", False):  4,
    ("Unknown", True):   6,
}

# Additional officers per point above this road-closure probability threshold.
# If historical closure rate is high but the caller did not explicitly specify
# requires_road_closure, we still nudge the count up a little.
CLOSURE_PROB_SOFT_THRESHOLD: float = 0.40  # 40% historical closure rate
CLOSURE_PROB_SOFT_EXTRA: int = 2           # extra officers added when exceeded

# Barricade is recommended whenever severity is Medium or above, OR when
# road closure probability exceeds this threshold.
BARRICADE_PROBABILITY_THRESHOLD: float = 0.25  # 25% historical closure rate
BARRICADE_SEVERITY_TIERS: set[str] = {"Medium", "High"}

# ─── Diversion disclaimer (config) ───────────────────────────────────────────
# The system honestly does not have live road-network data. These strings
# appear verbatim in every deployment note and rationale. Edit wording here.
DIVERSION_DISCLAIMER = (
    "A precise diversion route requires a live road-network graph with "
    "current traffic volumes, which is not available in this system. "
    "The recommended approach is to consult BTP's Standard Diversion Plan "
    "for the identified corridor and coordinate with the nearest traffic "
    "police station for real-time alternatives."
)

# NOTE ON DIVERSION ROUTING (dataset-compliance):
# We intentionally do NOT hardcode specific alternate-route suggestions.
# The ASTRAM dataset contains no road-network graph, adjacency, or
# alternate-route field — only the corridor name of the event itself.
# Asserting "road X is a viable diversion for road Y" would require outside
# geographic knowledge not present in the provided data, which this
# dataset-only submission must avoid. Diversion guidance is therefore limited
# to the honest disclaimer below, which points officers to BTP's own
# Standard Diversion Plan for the corridor.


# ─── Output dataclass ─────────────────────────────────────────────────────────
@dataclass
class ResourceRecommendation:
    """
    Structured deployment recommendation for a single event.

    Fields
    ------
    recommended_personnel_count : int
        Minimum officer count based on severity and closure signals.
    barricade_recommended : bool
        True when physical barricading should be prepared.
    barricade_location_hint : str
        Best available location for barricade placement (junction > address
        > generic fallback).
    diversion_suggestion : str
        Honest, corridor-aware diversion guidance including a clear disclaimer
        about the absence of live road-network data in this system.
    rationale : str
        Plain-English paragraph for a police officer. Mentions the data basis
        (duration vs. closure/priority) and the evidence count, so the reader
        knows exactly how certain the recommendation is.
    severity_tier : str
        Passed through from ImpactPrediction for convenience.
    confidence_label : str
        Passed through from ImpactPrediction for convenience.
    notes : list[str]
        Supplementary notes (e.g., soft closure probability bump, barricade
        reasoning).
    """
    recommended_personnel_count: int
    barricade_recommended: bool
    barricade_location_hint: str
    diversion_suggestion: str
    rationale: str
    severity_tier: str
    confidence_label: str
    notes: list[str] = field(default_factory=list)

    def display(self) -> None:
        """Print the recommendation in a clearly labelled format."""
        width = 60
        print("=" * width)
        print(f"  Severity tier     : {self.severity_tier}")
        print(f"  Confidence        : {self.confidence_label}")
        print(f"  Recommended staff : {self.recommended_personnel_count} officers")
        print(f"  Barricade needed  : {'YES' if self.barricade_recommended else 'No'}")
        print(f"  Barricade at      : {self.barricade_location_hint}")
        print(f"  Diversion         : {self.diversion_suggestion}")
        print()
        print("  RATIONALE:")
        # Word-wrap rationale at ~56 chars for terminal readability
        words = self.rationale.split()
        line, lines = [], []
        for w in words:
            if sum(len(x) + 1 for x in line) + len(w) > 56:
                lines.append("    " + " ".join(line))
                line = [w]
            else:
                line.append(w)
        if line:
            lines.append("    " + " ".join(line))
        print("\n".join(lines))
        if self.notes:
            print()
            for n in self.notes:
                print(f"  * {n}")
        print("=" * width)


# ─── Core functions ───────────────────────────────────────────────────────────

def _resolve_barricade_location(
    junction: Optional[str],
    address: Optional[str],
    corridor: Optional[str],
) -> str:
    """
    Pick the best available location string for the barricade hint.

    Priority: junction (most specific) > address > corridor > generic.
    'Unknown' junction value (the cleaned-data fill value) is treated as absent.

    Args:
        junction:  Junction name from the event record, or None/'Unknown'.
        address:   Address string from the event record, or None.
        corridor:  Corridor name, or None/'Non-corridor'.

    Returns:
        A short descriptive location string.
    """
    if junction and junction.strip().lower() not in ("unknown", ""):
        return f"{junction} junction"
    if address and address.strip():
        return f"near {address[:80]}"  # truncate very long addresses
    if corridor and corridor.strip().lower() not in ("non-corridor", ""):
        return f"along {corridor}"
    return "near event start location (coordinates available in ASTRAM record)"


def _build_diversion_suggestion(corridor: Optional[str]) -> str:
    """
    Return honest diversion guidance.

    We deliberately do NOT prescribe specific alternate roads: the ASTRAM
    dataset has no road-network/alternate-route data, so any concrete
    diversion route would be outside knowledge not traceable to the data.
    We return only the disclaimer, which directs officers to BTP's own
    Standard Diversion Plan for the corridor.

    Args:
        corridor: Named corridor string, or None. Accepted for API
            compatibility; not used to fabricate a route.

    Returns:
        The diversion disclaimer string.
    """
    return DIVERSION_DISCLAIMER


def _count_personnel(
    severity_tier: str,
    road_closure_probability: Optional[float],
    requires_road_closure: Optional[bool],
    notes: list[str],
) -> int:
    """
    Look up the recommended officer count from PERSONNEL_TABLE.

    Hard closure signal (requires_road_closure=True from the caller) takes
    precedence. If absent, a high historical closure_probability triggers a
    soft bump: officers are still looked up under closure=False but
    CLOSURE_PROB_SOFT_EXTRA additional officers are added, with an explanation.

    Args:
        severity_tier:            'Low' / 'Medium' / 'High' / 'Unknown'.
        road_closure_probability: Historical closure fraction (0–1) or None.
        requires_road_closure:    Caller-specified closure flag or None.
        notes:                    Mutable list to append explanations to.

    Returns:
        Integer officer count.
    """
    # Determine whether to look up the "closure=True" row
    hard_closure = bool(requires_road_closure)
    count = PERSONNEL_TABLE.get((severity_tier, hard_closure),
                                PERSONNEL_TABLE.get(("Unknown", hard_closure), 4))

    # Soft bump: high historical closure rate even if caller didn't specify
    if (
        not hard_closure
        and road_closure_probability is not None
        and road_closure_probability >= CLOSURE_PROB_SOFT_THRESHOLD
    ):
        count += CLOSURE_PROB_SOFT_EXTRA
        notes.append(
            f"Personnel count increased by {CLOSURE_PROB_SOFT_EXTRA} because historical "
            f"road closure rate ({road_closure_probability*100:.0f}%) exceeds "
            f"the {CLOSURE_PROB_SOFT_THRESHOLD*100:.0f}% soft threshold."
        )

    return count


def _build_rationale(
    event_cause: str,
    corridor: Optional[str],
    severity_tier: str,
    severity_basis: str,
    evidence_count: int,
    duration_count: int,
    duration_reliability: str,
    expected_duration_minutes: Optional[float],
    road_closure_probability: Optional[float],
    confidence_label: str,
    requires_road_closure: Optional[bool],
    personnel_count: int,
    barricade_recommended: bool,
) -> str:
    """
    Build a plain-English rationale paragraph for a police officer.

    CRITICAL RULE: the rationale must never imply more certainty than the
    upstream model's confidence_label and severity_basis actually support.
    Specifically:
      - If severity_basis == 'closure_priority', the rationale MUST state that
        severity was driven by closure rate and priority (not duration), and
        explain WHY (thin sample → unreliable duration).
      - If severity_basis == 'duration', quote the median duration and the
        evidence count normally.
      - In both cases, reproduce the confidence_label verbatim so the officer
        knows the overall trust level.

    Args:
        (All named parameters mirror the fields from ImpactPrediction and
        the computed recommendation values.)

    Returns:
        A single paragraph string.
    """
    corridor_str = (
        f" on {corridor}" if corridor and corridor.lower() not in ("non-corridor", "none", "")
        else ""
    )
    cause_str = event_cause.replace("_", " ")

    # ── Opening: what event, what corridor ───────────────────────────────────
    intro = (
        f"This recommendation is for a reported {cause_str} event"
        f"{corridor_str}. "
    )

    # ── Evidence basis ────────────────────────────────────────────────────────
    if severity_basis == "duration":
        # Duration was the primary signal — report it with evidence count
        dur_str = (
            f"{expected_duration_minutes:.0f} minutes"
            if expected_duration_minutes is not None
            else "an unknown duration"
        )
        basis_text = (
            f"Based on {evidence_count} similar historical events "
            f"({duration_count} with recorded resolution times), "
            f"the typical event of this type lasts approximately {dur_str}. "
            f"This duration estimate drove the {severity_tier} severity classification. "
        )

    else:
        # severity_basis == 'closure_priority' (or 'unknown' for no_data)
        # MANDATORY: explicitly state why duration was NOT the primary driver.
        rc_str = (
            f"{road_closure_probability*100:.0f}%"
            if road_closure_probability is not None
            else "unknown"
        )
        dur_note = ""
        if expected_duration_minutes is not None:
            dur_note = (
                f" A raw duration figure of {expected_duration_minutes:.0f} minutes "
                f"is on record but is based on only {duration_count} historical "
                f"records and is likely distorted by administrative-lag closures "
                f"(events marked closed days after the road actually cleared). "
                f"It has NOT been used to determine severity."
            )
        basis_text = (
            f"Severity was assessed from historical road closure rate "
            f"({rc_str}) and event priority, rather than from duration, "
            f"because only {duration_count} historical record(s) with timing data "
            f"exist for this event type — too few to produce a reliable duration "
            f"estimate at this sample size."
            f"{dur_note} "
        )

    # ── Confidence qualification ──────────────────────────────────────────────
    confidence_text = (
        f"Overall data confidence for this prediction: '{confidence_label}'. "
    )

    # ── Deployment rationale ──────────────────────────────────────────────────
    closure_note = ""
    if requires_road_closure is True:
        closure_note = (
            "Road closure has been confirmed for this event, "
            "which increases both staffing and barricade requirements. "
        )
    elif road_closure_probability is not None and road_closure_probability >= CLOSURE_PROB_SOFT_THRESHOLD:
        closure_note = (
            f"Historical data shows a {road_closure_probability*100:.0f}% chance "
            f"of road closure for similar events, so a precautionary staffing "
            f"uplift has been applied. "
        )

    barricade_note = (
        "Physical barricading is recommended. "
        if barricade_recommended
        else "Barricading is not indicated at this severity level, "
             "but can be escalated if conditions change on site. "
    )

    deployment_text = (
        f"{closure_note}"
        f"A minimum of {personnel_count} officers is recommended. "
        f"{barricade_note}"
    )

    return intro + basis_text + confidence_text + deployment_text


# ─── Main entry point ─────────────────────────────────────────────────────────

def generate_recommendation(
    impact,                              # ImpactPrediction instance
    event_cause: str,
    corridor: Optional[str] = None,
    junction: Optional[str] = None,
    address: Optional[str] = None,
    requires_road_closure: Optional[bool] = None,
) -> ResourceRecommendation:
    """
    Convert an ImpactPrediction into a ResourceRecommendation.

    Parameters
    ----------
    impact : ImpactPrediction
        Output from impact_model.predict_impact().
    event_cause : str
        Event type string (e.g. 'vehicle_breakdown'). Used in rationale text.
    corridor : str, optional
        Named road corridor. Drives diversion hint selection.
    junction : str, optional
        Junction name from the event record. Used for barricade location hint.
    address : str, optional
        Street address from the event record. Fallback for barricade hint.
    requires_road_closure : bool, optional
        Whether the event requires road closure. Passed through from the
        original event; None means unknown.

    Returns
    -------
    ResourceRecommendation dataclass.
    """
    notes: list[str] = []

    # ── Personnel count ───────────────────────────────────────────────────────
    personnel = _count_personnel(
        severity_tier=impact.severity_tier,
        road_closure_probability=impact.road_closure_probability,
        requires_road_closure=requires_road_closure,
        notes=notes,
    )

    # ── Barricade decision ────────────────────────────────────────────────────
    # Barricade is needed when:
    #   a) severity is Medium or High, OR
    #   b) historical road closure probability exceeds the threshold, OR
    #   c) caller explicitly confirmed road closure
    barricade_needed = (
        impact.severity_tier in BARRICADE_SEVERITY_TIERS
        or (
            impact.road_closure_probability is not None
            and impact.road_closure_probability >= BARRICADE_PROBABILITY_THRESHOLD
        )
        or (requires_road_closure is True)
    )

    # ── Location hint ─────────────────────────────────────────────────────────
    barricade_loc = _resolve_barricade_location(junction, address, corridor)

    # ── Diversion suggestion ──────────────────────────────────────────────────
    diversion = _build_diversion_suggestion(corridor)

    # ── Rationale ─────────────────────────────────────────────────────────────
    rationale = _build_rationale(
        event_cause=event_cause,
        corridor=corridor,
        severity_tier=impact.severity_tier,
        severity_basis=impact.severity_basis,
        evidence_count=impact.evidence_count,
        duration_count=impact.duration_count,
        duration_reliability=impact.duration_reliability,
        expected_duration_minutes=impact.expected_duration_minutes,
        road_closure_probability=impact.road_closure_probability,
        confidence_label=impact.confidence_label,
        requires_road_closure=requires_road_closure,
        personnel_count=personnel,
        barricade_recommended=barricade_needed,
    )

    return ResourceRecommendation(
        recommended_personnel_count=personnel,
        barricade_recommended=barricade_needed,
        barricade_location_hint=barricade_loc,
        diversion_suggestion=diversion,
        rationale=rationale,
        severity_tier=impact.severity_tier,
        confidence_label=impact.confidence_label,
        notes=notes,
    )


# ─── Demo / __main__ block ────────────────────────────────────────────────────

if __name__ == "__main__":
    # Add src/ to path so sibling imports work when run from project root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

    from impact_model import predict_impact

    LOOKUP_PATH  = os.path.join("data", "processed", "historical_lookup.csv")
    CLEANED_PATH = os.path.join("data", "processed", "cleaned_events.csv")

    print("Loading data...")
    _lookup  = pd.read_csv(LOOKUP_PATH)
    _cleaned = pd.read_csv(CLEANED_PATH)
    _cleaned["has_duration"]           = _cleaned["has_duration"].astype(bool)
    _cleaned["requires_road_closure"]  = _cleaned["requires_road_closure"].astype(bool)

    # ── 4 demo cases — mirrors impact_model.py's __main__ exactly ─────────────
    DEMO_CASES = [
        {
            "label":               "[1] Fine match — vehicle breakdown, Mysore Road, weekday morning",
            "event_cause":         "vehicle_breakdown",
            "planned_start":       datetime(2024, 4, 1, 8, 0, 0),
            "corridor":            "Mysore Road",
            "junction":            "Unknown",           # typical cleaned-data value
            "address":             "Mysore Road, Bengaluru",
            "requires_road_closure": False,
        },
        {
            "label":               "[2] Coarse match — vehicle breakdown, unknown new road, Sat afternoon",
            "event_cause":         "vehicle_breakdown",
            "planned_start":       datetime(2024, 4, 6, 17, 0, 0),
            "corridor":            "New Peripheral Ring Road",
            "junction":            None,
            "address":             "Peripheral Ring Road, Bengaluru",
            "requires_road_closure": False,
        },
        {
            "label":               "[3] Cause-only — VIP movement, CBD 2, Sat night, road closure=True",
            "event_cause":         "vip_movement",
            "planned_start":       datetime(2024, 4, 6, 23, 0, 0),
            "corridor":            "CBD 2",
            "junction":            "AyyappaTempleJunc",
            "address":             "MG Road, CBD Bengaluru",
            "requires_road_closure": True,
        },
        {
            "label":               "[4] Cause-only — protest, no corridor, Tue morning, road closure=True",
            "event_cause":         "protest",
            "planned_start":       datetime(2024, 4, 2, 11, 0, 0),
            "corridor":            None,
            "junction":            None,
            "address":             "Cubbon Park Road, Bengaluru",
            "requires_road_closure": True,
        },
    ]

    print()
    print("#" * 60)
    print("  FULL PIPELINE DEMO: Impact Model -> Resource Engine")
    print("#" * 60)

    for case in DEMO_CASES:
        print(f"\n{case['label']}")

        # Step 1: predict impact
        impact = predict_impact(
            event_cause=case["event_cause"],
            planned_start_datetime=case["planned_start"],
            corridor=case["corridor"],
            requires_road_closure=case["requires_road_closure"],
            lookup_df=_lookup,
            cleaned_df=_cleaned,
        )

        # Step 2: generate resource recommendation
        rec = generate_recommendation(
            impact=impact,
            event_cause=case["event_cause"],
            corridor=case["corridor"],
            junction=case.get("junction"),
            address=case.get("address"),
            requires_road_closure=case["requires_road_closure"],
        )

        rec.display()
