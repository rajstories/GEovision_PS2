#!/usr/bin/env python3
"""
BTP Action Brief export.

This is a formatting layer over the existing impact, resource, and advisory
modules. It does not add prediction logic or diversion routing.
"""

import html
import re
from datetime import datetime
from typing import Optional

import pandas as pd


def _fmt_minutes(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{float(value):.0f} minutes"


def _fmt_percent(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{float(value) * 100:.1f}%"


def extract_vms_message(advisory_text: str) -> str:
    """Extract the VMS line produced by advisory_generator.py."""
    match = re.search(r"VMS SIGNBOARD MESSAGE:\s*\n\[\s*(.*?)\s*\]", advisory_text, re.S)
    return match.group(1).strip() if match else "See advisory text"


def _historical_rows_html(evidence_df: Optional[pd.DataFrame], limit: int = 8) -> str:
    if evidence_df is None or evidence_df.empty:
        return "<p>No specific historical rows available for this evidence tier.</p>"

    cols = [
        "start_datetime_ist",
        "event_cause",
        "corridor",
        "priority",
        "requires_road_closure",
        "duration_minutes",
        "police_station",
        "junction",
    ]
    available = [c for c in cols if c in evidence_df.columns]
    display = evidence_df[available].copy().head(limit)
    if "duration_minutes" in display.columns:
        display["duration_minutes"] = pd.to_numeric(
            display["duration_minutes"], errors="coerce"
        ).map(lambda x: "" if pd.isna(x) else f"{x:.0f}")

    header = "".join(f"<th>{html.escape(c.replace('_', ' ').title())}</th>" for c in available)
    body_rows = []
    for _, row in display.iterrows():
        body_rows.append(
            "<tr>"
            + "".join(
                f"<td>{html.escape(str(row.get(c, '')))}</td>"
                for c in available
            )
            + "</tr>"
        )

    return (
        "<table>"
        f"<thead><tr>{header}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def generate_action_brief_html(
    *,
    incident_id: str,
    generated_at: datetime,
    event_cause: str,
    planned_start_datetime: datetime,
    corridor: Optional[str],
    requires_road_closure: bool,
    impact,
    rec,
    advisory_text: str,
    evidence_df: Optional[pd.DataFrame] = None,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> str:
    """
    Generate a printable HTML action brief for a traffic officer.
    """
    location = corridor if corridor else "Non-corridor"
    coordinates = (
        f"{float(latitude):.6f}, {float(longitude):.6f}"
        if latitude is not None and longitude is not None
        else "Not available in event form"
    )
    cause_title = event_cause.replace("_", " ").title()
    vms_message = extract_vms_message(advisory_text)
    evidence_count = getattr(impact, "evidence_count", 0)
    duration_count = getattr(impact, "duration_count", 0)
    match_level = getattr(impact, "match_level", "unknown")

    escaped_advisory = html.escape(advisory_text)
    historical_rows = _historical_rows_html(evidence_df)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>BTP Action Brief - {html.escape(incident_id)}</title>
  <style>
    body {{ font-family: Arial, sans-serif; color: #1f2933; margin: 28px; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 26px; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; }}
    .meta {{ color: #52606d; font-size: 13px; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
    .box {{ border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px 14px; }}
    .label {{ color: #52606d; font-size: 12px; text-transform: uppercase; font-weight: 700; }}
    .value {{ font-size: 17px; font-weight: 700; margin-top: 4px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 6px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 12px; }}
    .signoff {{ margin-top: 28px; display: grid; grid-template-columns: 1fr 1fr; gap: 28px; }}
    .line {{ border-bottom: 1px solid #1f2933; height: 34px; }}
  </style>
</head>
<body>
  <h1>BTP Action Brief</h1>
  <div class="meta">Incident ID: <strong>{html.escape(incident_id)}</strong> | Generated: {generated_at:%Y-%m-%d %H:%M}</div>

  <h2>1. Location and Event Details</h2>
  <div class="grid">
    <div class="box"><div class="label">Corridor</div><div class="value">{html.escape(location)}</div></div>
    <div class="box"><div class="label">Cause</div><div class="value">{html.escape(cause_title)}</div></div>
    <div class="box"><div class="label">Event Timestamp</div><div class="value">{planned_start_datetime:%Y-%m-%d %H:%M}</div></div>
    <div class="box"><div class="label">Coordinates</div><div class="value">{html.escape(coordinates)}</div></div>
  </div>

  <h2>2. Predicted Impact</h2>
  <div class="grid">
    <div class="box"><div class="label">Severity</div><div class="value">{html.escape(impact.severity_tier)}</div></div>
    <div class="box"><div class="label">Expected Duration</div><div class="value">{_fmt_minutes(impact.expected_duration_minutes)}</div></div>
    <div class="box"><div class="label">Closure Probability</div><div class="value">{_fmt_percent(impact.road_closure_probability)}</div></div>
    <div class="box"><div class="label">Confidence</div><div class="value">{html.escape(impact.confidence_label)}</div></div>
  </div>

  <h2>3. Supporting Historical Evidence</h2>
  <p>Evidence count: <strong>{evidence_count}</strong> total similar events; duration records: <strong>{duration_count}</strong>; match tier: <strong>{html.escape(match_level)}</strong>.</p>
  {historical_rows}

  <h2>4. Officer Requirement</h2>
  <p><strong>{rec.recommended_personnel_count} officers</strong> recommended.</p>
  <p>{html.escape(rec.rationale)}</p>

  <h2>5. Barricade Placement</h2>
  <p>{'Recommended' if rec.barricade_recommended else 'Not indicated'}: {html.escape(rec.barricade_location_hint)}</p>

  <h2>6. Diversion Decision</h2>
  <p>Road closure expected by operator: <strong>{'Yes' if requires_road_closure else 'No'}</strong>.</p>
  <p>{html.escape(rec.diversion_suggestion)}</p>

  <h2>7. Public Advisory and VMS Message</h2>
  <p><strong>Short VMS sign-board message:</strong> {html.escape(vms_message)}</p>
  <pre>{escaped_advisory}</pre>

  <h2>8. Officer Sign-Off</h2>
  <div class="signoff">
    <div><div class="label">Name / ID</div><div class="line"></div></div>
    <div><div class="label">Signature</div><div class="line"></div></div>
    <div><div class="label">Sign-Off Timestamp</div><div class="line"></div></div>
    <div><div class="label">Remarks</div><div class="line"></div></div>
  </div>
</body>
</html>
"""
