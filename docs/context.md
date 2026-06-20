# Project Context — Gridlock Hackathon: Event-Driven Congestion

## Problem
Forecast event-related traffic impact (planned & unplanned) and recommend
manpower, barricading, and diversion plans, for Bengaluru Traffic Police.

## Dataset: data/raw/astram_event_data.csv (~8,173 rows, real BTP ASTRAM logs)
Key columns:
- event_type: "planned" or "unplanned"
- event_cause: vehicle_breakdown, construction, water_logging, accident,
  tree_fall, congestion, public_event, procession, vip_movement, protest,
  pot_holes, road_conditions, others
- requires_road_closure: True/False
- start_datetime, end_datetime, closed_datetime, resolved_datetime (UTC, ISO format)
  -> end_datetime is mostly null; use closed_datetime, fallback resolved_datetime
- status: closed, active, resolved
- priority: High, Low
- corridor: named road corridor or "Non-corridor" (~70 unique values)
- zone: Central/North/South/East/West Zone 1 & 2 (many nulls)
- police_station: fully populated, ~location-level unit
- junction: named junction, sparsely populated
- latitude, longitude, address: location data
- Known data quality issues: some durations are negative or absurdly large
  (100,000+ minutes) — these are bad records, must be filtered, not imputed silently

## Design constraints (non-negotiable)
- Must be explainable to a non-technical police officer — no black-box models
  as the primary output. Show "based on N historical similar events" evidence.
- Lightweight stack only: Python, pandas, scikit-learn, SQLite or CSV storage,
  Streamlit for UI. No cloud-only dependencies, no GPU requirement.
- Config-driven thresholds (severity cutoffs etc.), not hardcoded magic numbers.
- Code must be modular: one responsibility per file, clear function docstrings,
  type hints, and inline comments explaining *why*, not just *what*.
- Must handle missing/dirty data gracefully and log what was dropped/imputed.
