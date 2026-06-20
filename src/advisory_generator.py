#!/usr/bin/env python3
"""
Traffic Advisory Generator for ASTRAM Traffic Events

This module converts a ResourceRecommendation, an ImpactPrediction, and event metadata
into a formatted public traffic advisory in the style typically published by the Bengaluru
Traffic Police (BTP).
"""

from datetime import datetime
from typing import Optional
import pandas as pd
from resource_engine import CORRIDOR_DIVERSION_HINTS

# VMS concise alternates mapping
VMS_ALTS = {
    "mysore road": "USE KANAKAPURA RD / CHORD RD",
    "tumkur road": "USE HESARAGHATTA RD / RING RD",
    "bellary road 1": "USE THANISANDRA RD",
    "bellary road 2": "USE HENNUR RD",
    "nh 44": "USE NICE RD",
    "old madras road": "USE WHITEFIELD RD",
    "hosur road": "USE BANNERGHATTA RD / SARJAPUR RD",
    "sarjapur road": "USE CARMELARAM RD",
    "outer ring road": "EXPECT DELAYS"
}

def generate_traffic_advisory(
    event_cause: str,
    planned_start_datetime: datetime,
    corridor: Optional[str],
    requires_road_closure: bool,
    impact, # ImpactPrediction instance
    rec, # ResourceRecommendation instance
    cleaned_df: Optional[pd.DataFrame] = None
) -> str:
    """
    Generate a publishable BTP-formatted traffic advisory text block.

    Parameters
    ----------
    event_cause : str
        The type of event (e.g. 'vehicle_breakdown').
    planned_start_datetime : datetime
        Event start datetime.
    corridor : str, optional
        Corridor name.
    requires_road_closure : bool
        True if road closure is expected.
    impact : ImpactPrediction
        Output from impact_model.predict_impact().
    rec : ResourceRecommendation
        Output from resource_engine.generate_recommendation().
    cleaned_df : pd.DataFrame, optional
        Used to resolve which police station typically manages this corridor.

    Returns
    -------
    str
        Formatted traffic advisory draft block.
    """
    # ── Resolve Police Station beat ──────────────────────────────────────────
    police_station = "local Traffic Police Station"
    if corridor and corridor != "Non-corridor" and cleaned_df is not None:
        matches = cleaned_df[cleaned_df["corridor"] == corridor]
        if len(matches) > 0:
            station_mode = matches["police_station"].mode()
            if not station_mode.empty:
                police_station = f"{station_mode.iloc[0]} Traffic Police Station"

    # ── Time Range calculations ──────────────────────────────────────────────
    start_str = planned_start_datetime.strftime("%I:%M %p")
    
    if impact.expected_duration_minutes and not pd.isna(impact.expected_duration_minutes):
        duration_delta = pd.Timedelta(minutes=impact.expected_duration_minutes)
        end_time = planned_start_datetime + duration_delta
        end_str = end_time.strftime("%I:%M %p")
        time_range = f"{start_str} to {end_str}"
    else:
        time_range = f"starting at {start_str}"

    # ── Closure Statement ────────────────────────────────────────────────────
    location_str = corridor if corridor else "Non-corridor"
    
    if rec.barricade_recommended or requires_road_closure:
        closure_statement = (
            f"Vehicular movement on {location_str} between {time_range} is expected to "
            f"be restricted. A full road closure/diversion is recommended based on historical pattern."
        )
    else:
        closure_statement = (
            f"Vehicular movement on {location_str} between {time_range} is expected to "
            f"be slow-moving but unrestricted. Slow down and expect minor delays."
        )

    # ── Diversion Statement ──────────────────────────────────────────────────
    hint = None
    if corridor:
        hint = CORRIDOR_DIVERSION_HINTS.get(corridor.strip().lower())
        
    if hint:
        diversion_line = f"Diversion Route: {hint}"
    else:
        diversion_line = "Diversion Route: No specific alternate route is designated. Commuters are advised to seek alternative roads or expect delays."

    # ── VMS Signboard text (concise, under 100 characters) ───────────────────
    corr_upper = corridor.upper() if corridor else "ROAD"
    cause_upper = event_cause.replace("_", " ").upper()
    
    # Concise time range
    if impact.expected_duration_minutes and not pd.isna(impact.expected_duration_minutes):
        end_time = planned_start_datetime + pd.Timedelta(minutes=impact.expected_duration_minutes)
        vms_time = f"{planned_start_datetime.strftime('%I:%M%p')}-{end_time.strftime('%I:%M%p')}"
    else:
        vms_time = f"FROM {planned_start_datetime.strftime('%I:%M%p')}"

    alt_route = "EXPECT DELAYS"
    if corridor:
        alt_route = VMS_ALTS.get(corridor.lower().strip(), "EXPECT DELAYS")

    if rec.barricade_recommended or requires_road_closure:
        vms_text = f"DIVERSION: {corr_upper} CLOSED {vms_time}. {alt_route}"
    else:
        vms_text = f"ALERT: {cause_upper} ON {corr_upper} {vms_time}. EXPECT DELAYS."
        
    # Strictly enforce 100 character VMS constraint
    if len(vms_text) > 100:
        vms_text = vms_text[:97] + "..."

    # ── Personnel coordinated alert ──────────────────────────────────────────
    personnel_line = (
        f"Recommend coordinating with {police_station} beat officers; "
        f"estimated additional personnel: {rec.recommended_personnel_count}."
    )

    # ── Compile text block ───────────────────────────────────────────────────
    date_str = planned_start_datetime.strftime("%Y-%m-%d")
    cause_title = event_cause.replace("_", " ").title()
    loc_title = corridor if corridor else "Non-corridor"
    
    advisory = []
    advisory.append(f"Traffic Advisory: {cause_title} at {loc_title} — {date_str}")
    advisory.append("=" * len(advisory[0]))
    advisory.append("")
    advisory.append(closure_statement)
    advisory.append("")
    advisory.append(diversion_line)
    advisory.append("")
    advisory.append(f"VMS SIGNBOARD MESSAGE:\n[ {vms_text.upper()} ]")
    advisory.append("")
    advisory.append(personnel_line)
    advisory.append("")
    advisory.append("*" * 52)
    advisory.append("*** DRAFT — officer review and sign-off required ***")
    advisory.append("*" * 52)
    
    return "\n".join(advisory)
