#!/usr/bin/env python3
"""
Verified Outcome Refresh for ASTRAM Traffic Events.

This module captures officer-verified event outcomes and rebuilds the
historical lookup table in memory. It does not mutate the supplied ASTRAM
dataset files; verified feedback is appended to a separate feedback log and
combined with cleaned_events.csv only for a derived, refreshed lookup.
"""

import csv
import hashlib
import os
from datetime import datetime
from typing import Optional

import pandas as pd

from feature_engineering import assign_hour_bucket, build_lookup_table

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "data", "processed", "feedback_log.csv")
CLEANED_EVENTS_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_events.csv")

FEEDBACK_SCHEMA = [
    "timestamp",
    "event_datetime",
    "event_cause",
    "corridor",
    "actual_duration_minutes",
    "actual_requires_road_closure",
    "priority",
    "verified_by",
    "verification_status",
]

VALID_VERIFICATION_STATUSES = {"verified", "unverified", "disputed"}


def _normalise_corridor(corridor: Optional[str]) -> str:
    if corridor is None or str(corridor).strip() == "":
        return "Non-corridor"
    if str(corridor).strip().lower() == "unknown":
        return "Non-corridor"
    return str(corridor).strip()


def _normalise_status(status: object) -> str:
    normalised = str(status or "unverified").strip().lower()
    if normalised not in VALID_VERIFICATION_STATUSES:
        return "unverified"
    return normalised


def _coerce_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "y", "closed"}


def _feedback_key(row: pd.Series) -> str:
    """
    Build a stable event-outcome key used to ignore exact duplicate submissions.

    There is no incident ID in the feedback form, so the dedupe key uses the
    operational outcome fields that identify the verified event itself. The
    verifier and audit timestamp are intentionally excluded: two clicks by the
    same or different officers should not create two historical events.
    """
    event_dt = pd.to_datetime(row.get("event_datetime"), errors="coerce")
    event_dt_key = event_dt.isoformat() if pd.notna(event_dt) else ""
    duration = pd.to_numeric(row.get("actual_duration_minutes"), errors="coerce")
    duration_key = "" if pd.isna(duration) else f"{float(duration):.4f}"
    raw = "|".join(
        [
            event_dt_key,
            str(row.get("event_cause", "")).strip().lower(),
            _normalise_corridor(row.get("corridor")).lower(),
            duration_key,
            str(_coerce_bool(row.get("actual_requires_road_closure"))),
            str(row.get("priority", "Unknown")).strip().lower(),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_feedback_log(feedback_log_path: str = FEEDBACK_LOG_PATH) -> pd.DataFrame:
    """
    Read feedback rows and normalise old/new schemas to FEEDBACK_SCHEMA.

    Older logs used `requires_road_closure` and omitted event_datetime,
    priority, verified_by, and verification_status. Those rows are still
    readable; missing verification status is treated as unverified so only
    explicitly verified outcomes enter the refresh.
    """
    if not os.path.exists(feedback_log_path):
        return pd.DataFrame(columns=FEEDBACK_SCHEMA)

    try:
        df = pd.read_csv(feedback_log_path)
    except pd.errors.ParserError:
        # Fallback for mixed-schema files caused by mode="a" appends
        with open(feedback_log_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                header = []
            rows = []
            for row in reader:
                if not row:
                    continue
                # If row matches FEEDBACK_SCHEMA length but header doesn't, it's a newly appended row
                if len(row) == len(FEEDBACK_SCHEMA) and len(header) != len(FEEDBACK_SCHEMA):
                    rows.append(dict(zip(FEEDBACK_SCHEMA, row)))
                else:
                    rows.append(dict(zip(header, row)))
        df = pd.DataFrame(rows)

    if "actual_requires_road_closure" not in df.columns and "requires_road_closure" in df.columns:
        df["actual_requires_road_closure"] = df["requires_road_closure"]
    if "event_datetime" not in df.columns and "timestamp" in df.columns:
        df["event_datetime"] = df["timestamp"]

    for col in FEEDBACK_SCHEMA:
        if col not in df.columns:
            if col == "verification_status":
                df[col] = "unverified"
            elif col == "priority":
                df[col] = "Unknown"
            elif col == "verified_by":
                df[col] = ""
            elif col == "actual_requires_road_closure":
                df[col] = False
            else:
                df[col] = pd.NA

    df = df[FEEDBACK_SCHEMA].copy()
    df["corridor"] = df["corridor"].apply(_normalise_corridor)
    df["actual_requires_road_closure"] = df["actual_requires_road_closure"].apply(_coerce_bool)
    df["verification_status"] = df["verification_status"].apply(_normalise_status)
    df["priority"] = df["priority"].fillna("Unknown").astype(str)
    df["verified_by"] = df["verified_by"].fillna("").astype(str)
    df["event_datetime"] = pd.to_datetime(df["event_datetime"], errors="coerce")
    df["actual_duration_minutes"] = pd.to_numeric(
        df["actual_duration_minutes"], errors="coerce"
    )
    return df


def log_actual_outcome(
    event_cause: str,
    corridor: Optional[str],
    actual_duration_minutes: object,
    requires_road_closure: Optional[bool] = None,
    *,
    event_datetime: Optional[datetime] = None,
    actual_requires_road_closure: Optional[bool] = None,
    priority: str = "Unknown",
    verified_by: str = "",
    verification_status: str = "unverified",
    feedback_log_path: str = FEEDBACK_LOG_PATH,
) -> bool:
    """
    Append an actual event outcome to the feedback log.

    The positional `requires_road_closure` argument is preserved for backward
    compatibility with the original dashboard call. New callers should pass
    `actual_requires_road_closure` explicitly.

    Returns True when a row is appended and False when the same event outcome
    already exists in the feedback log.
    """
    closure_value = (
        actual_requires_road_closure
        if actual_requires_road_closure is not None
        else requires_road_closure
    )
    event_dt = event_datetime or datetime.now()
    new_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_datetime": pd.Timestamp(event_dt).strftime("%Y-%m-%d %H:%M:%S"),
        "event_cause": event_cause,
        "corridor": _normalise_corridor(corridor),
        "actual_duration_minutes": actual_duration_minutes,
        "actual_requires_road_closure": _coerce_bool(closure_value),
        "priority": priority if priority else "Unknown",
        "verified_by": verified_by.strip() if verified_by else "",
        "verification_status": _normalise_status(verification_status),
    }

    existing = read_feedback_log(feedback_log_path)
    new_key = _feedback_key(pd.Series(new_record))
    if len(existing) > 0:
        existing_keys = existing.apply(_feedback_key, axis=1)
        if new_key in set(existing_keys):
            return False

    df_new = pd.DataFrame([new_record], columns=FEEDBACK_SCHEMA)
    os.makedirs(os.path.dirname(feedback_log_path), exist_ok=True)
    
    # Write the entire combined dataframe to ensure schema consistency
    combined = pd.concat([existing, df_new], ignore_index=True) if len(existing) > 0 else df_new
    combined.to_csv(feedback_log_path, index=False)
    return True


def feedback_rows_to_events(feedback_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert explicitly verified feedback rows into cleaned-event-shaped rows.

    Invalid or missing durations are kept as events with has_duration=False, so
    they can inform closure-rate statistics without entering duration medians.
    """
    if feedback_df.empty:
        return pd.DataFrame()

    verified = feedback_df[
        feedback_df["verification_status"].astype(str).str.lower() == "verified"
    ].copy()
    if verified.empty:
        return pd.DataFrame()

    verified["feedback_key"] = verified.apply(_feedback_key, axis=1)
    verified = verified.drop_duplicates(subset=["feedback_key"], keep="first").copy()
    verified = verified.dropna(subset=["event_datetime", "event_cause"])
    if verified.empty:
        return pd.DataFrame()

    durations = pd.to_numeric(verified["actual_duration_minutes"], errors="coerce")
    has_duration = durations.notna() & (durations >= 0)
    start_dt = pd.to_datetime(verified["event_datetime"], errors="coerce")
    hour_of_day = start_dt.dt.hour
    day_of_week = start_dt.dt.dayofweek

    events = pd.DataFrame(
        {
            "id": "feedback:" + verified["feedback_key"],
            "event_type": "verified_feedback",
            "event_cause": verified["event_cause"].astype(str),
            "corridor": verified["corridor"].apply(_normalise_corridor),
            "priority": verified["priority"].fillna("Unknown").astype(str),
            "requires_road_closure": verified["actual_requires_road_closure"].apply(_coerce_bool),
            "start_datetime": start_dt,
            "start_datetime_ist": start_dt,
            "duration_minutes": durations.where(has_duration, pd.NA),
            "has_duration": has_duration,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "is_weekend": day_of_week.isin([5, 6]),
        }
    )
    events["hour_bucket"] = events["hour_of_day"].apply(assign_hour_bucket)
    return events


def build_combined_events_with_feedback(
    cleaned_events_path: str = CLEANED_EVENTS_PATH,
    feedback_log_path: str = FEEDBACK_LOG_PATH,
) -> pd.DataFrame:
    """
    Return original cleaned events plus verified feedback rows in memory only.
    """
    cleaned_df = pd.read_csv(cleaned_events_path)
    cleaned_df["has_duration"] = cleaned_df["has_duration"].astype(bool)
    cleaned_df["requires_road_closure"] = cleaned_df["requires_road_closure"].astype(bool)
    if "hour_bucket" not in cleaned_df.columns:
        cleaned_df["hour_bucket"] = cleaned_df["hour_of_day"].apply(assign_hour_bucket)

    feedback_df = read_feedback_log(feedback_log_path)
    feedback_events = feedback_rows_to_events(feedback_df)
    if feedback_events.empty:
        return cleaned_df

    for col in cleaned_df.columns:
        if col not in feedback_events.columns:
            feedback_events[col] = pd.NA
    feedback_events = feedback_events[cleaned_df.columns]
    combined = pd.concat([cleaned_df, feedback_events], ignore_index=True)
    combined["has_duration"] = combined["has_duration"].astype(bool)
    combined["requires_road_closure"] = combined["requires_road_closure"].astype(bool)
    return combined


def refresh_lookup_with_feedback(
    cleaned_events_path: str = CLEANED_EVENTS_PATH,
    feedback_log_path: str = FEEDBACK_LOG_PATH,
) -> pd.DataFrame:
    """
    Build a refreshed historical lookup with verified feedback included.

    The original cleaned_events.csv file is read but never modified. The
    returned lookup can be passed directly to predict_impact() for the next
    dashboard prediction in the same Streamlit session.
    """
    combined = build_combined_events_with_feedback(
        cleaned_events_path=cleaned_events_path,
        feedback_log_path=feedback_log_path,
    )
    return build_lookup_table(combined)
