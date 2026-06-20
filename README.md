# Gridlock — Event-Driven Congestion Forecasting for Bengaluru Traffic Police

A lightweight, explainable decision-support system for the Bengaluru Traffic Police (BTP) that predicts traffic impact from planned and unplanned road events and generates actionable deployment recommendations — grounded entirely in real historical ASTRAM data.

---

## Problem

Bengaluru's traffic police respond to hundreds of road events daily: vehicle breakdowns, construction, VIP movements, protests, waterlogging, and more. Response quality depends on how quickly officers can answer three questions:

1. How long will this disrupt traffic?
2. Will a road closure be needed?
3. How many officers and barricades should we deploy?

Today, those decisions rely on individual officer experience and are not systematically informed by past event data. This system changes that.

---

## Solution Overview

The pipeline has four stages:

| Stage | File | Description |
|---|---|---|
| Data Cleaning | `src/data_cleaning.py` | Cleans ASTRAM logs, fixes datetime issues, computes duration |
| Feature Engineering | `src/feature_engineering.py` | Builds historical lookup table from cleaned events |
| Impact Model | `src/impact_model.py` | Predicts severity, duration, and road closure probability |
| Resource Engine | `src/resource_engine.py` | Converts impact into a deployment recommendation |
| Dashboard | `app/dashboard.py` | Streamlit UI for officers to query the system |

---

## Running the Dashboard

From the project root:

```bash
py -m streamlit run app/dashboard.py
```

Requires: `streamlit`, `pandas` (and their dependencies). Processed data files must exist at `data/processed/`. If not, run `src/data_cleaning.py` then `src/feature_engineering.py` first.

---

## How It Works

### Fallback Hierarchy

The system never fabricates data. When historical coverage is thin, it falls back gracefully rather than projecting an unreliable number with false confidence:

1. **Fine match** — exact corridor + cause + time-of-day + weekend flag
2. **Coarse match** — city-wide cause + time-of-day + weekend flag (corridor dropped)
3. **Cause-only** — all historical events of this type, regardless of time or location
4. **No data** — honest "Insufficient data — defer to officer judgment"

Every prediction shown to an officer states which tier it came from and how many historical events back it up.

### Severity Scoring

- **When sample is reliable (≥ 5 duration records):** severity is driven by historical median duration.
- **When sample is thin:** severity is driven by historical road-closure rate and event priority instead. Raw duration figures are shown in a muted style with a `(low reliability)` flag so they are not mistaken for confident estimates. This design specifically prevents administrative-lag outliers (e.g., a VIP event marked "closed" 12 days after the road actually cleared) from producing misleading severity projections.

---

## Known Limitations & Scope

### Personnel Recommendations and VIP Events

Personnel recommendations are calibrated from **routine traffic-management response**, not high-security VIP protocols, which follow separate force-deployment chains (SPG/escort-driven coordination outside the ASTRAM event-logging system). This system augments, not replaces, that process for VIP-classified events.

In practice, for `vip_movement` events, officers should treat the personnel count as the *traffic-management component* of the response only, and coordinate the security component through the appropriate protocol channels independently.

### Diversion Routes

The system does not have access to a live road-network graph. Diversion suggestions are corridor-aware hints based on BTP's established Standard Diversion Plans, not dynamically computed routes. Officers should validate suggestions against current conditions.

### Duration Reliability

Only ~38% of historical ASTRAM records have a computable duration (i.e., a resolved or closed timestamp). Duration estimates are always displayed alongside the number of records they are based on. Treat low-count estimates as directional guidance rather than precise forecasts.

---

## Data

Source: BTP ASTRAM traffic event logs (~8,173 rows). After cleaning: **7,331 events** retained. See `docs/context.md` for full schema documentation and data quality notes.

---

## Design Principles

- **Explainable by default.** Every output says "based on N historical events" and names the match tier. No black-box scores.
- **Config-driven thresholds.** Severity cutoffs, personnel counts, and barricade probability thresholds are named constants in each source file — editable without touching logic.
- **Honest about uncertainty.** Low-reliability signals are labelled as such in the UI. Missing data is never silently imputed in a way that inflates confidence.
- **Lightweight stack.** Python + pandas + Streamlit. No cloud-only dependencies, no GPU requirement, deployable on a station laptop.
