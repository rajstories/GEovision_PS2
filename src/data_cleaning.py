#!/usr/bin/env python3
"""
Data Cleaning Script for ASTRAM Traffic Event Log Dataset

This script:
1. Loads raw event logs from 'data/raw/astram_event_data.csv'.
2. Parses datetime columns into timezone-aware UTC datetime objects.
3. Computes the local IST time of the event starts.
4. Derives the effective end datetime using a fallback hierarchy (closed -> resolved -> end).
5. Computes event duration and flags invalid records (missing start_datetime, negative, 
   or exceeding type-specific implausibility thresholds).
6. Keeps rows with missing end times in cleaned_events.csv, setting has_duration=False and 
   duration_minutes=NaN.
7. Handles missing categorical fields (corridors, zones, junctions).
8. Adds time-based features (hour of day, day of week, weekend flag) in IST.
9. Exports cleaned data and a detailed log of dropped rows.
"""

import os
from typing import List
import pandas as pd

# Define paths relative to project root
RAW_DATA_PATH = os.path.join("data", "raw", "astram_event_data.csv")
PROCESSED_DIR = os.path.join("data", "processed")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DIR, "cleaned_events.csv")
DROPPED_LOG_PATH = os.path.join(PROCESSED_DIR, "dropped_rows_log.csv")

def main() -> None:
    """
    Main function to execute the data cleaning pipeline.
    """
    print("Starting ASTRAM event data cleaning pipeline...")
    
    # 1. Load data
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data file not found at: {RAW_DATA_PATH}")
    
    # Read CSV, treating 'NULL' as NaN values explicitly
    print(f"Loading raw data from: {RAW_DATA_PATH}")
    df = pd.read_csv(RAW_DATA_PATH, na_values=["NULL", "null", ""])
    initial_row_count = len(df)
    print(f"Loaded {initial_row_count} rows.")

    # 2. Parse Datetimes
    # Convert datetimes to UTC. Invalid formats are coerced to NaT.
    datetime_cols = ["start_datetime", "end_datetime", "closed_datetime", "resolved_datetime"]
    print("Parsing datetime columns as UTC...")
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
    
    # Add Asia/Kolkata version of start_datetime (IST) for local traffic patterns analysis
    print("Converting start_datetime to Asia/Kolkata (IST)...")
    # For rows where start_datetime is null, tz_convert will output NaT, which is expected.
    df["start_datetime_ist"] = df["start_datetime"].dt.tz_convert("Asia/Kolkata")

    # 3. Create effective_end_datetime
    # Hierarchy: closed_datetime -> fallback resolved_datetime -> fallback end_datetime
    print("Computing effective_end_datetime using fallback hierarchy (closed -> resolved -> end)...")
    df["effective_end_datetime"] = (
        df["closed_datetime"]
        .fillna(df["resolved_datetime"])
        .fillna(df["end_datetime"])
    )

    # 4. Compute duration in minutes
    print("Computing duration in minutes where end times are available...")
    df["duration_minutes"] = (
        df["effective_end_datetime"] - df["start_datetime"]
    ).dt.total_seconds() / 60.0

    # 5. Flag and remove invalid garbage/negative rows
    print("Identifying and logging rows to drop...")
    # Rows with missing start_datetime (we cannot perform time-based analyses on these)
    is_missing_start = df["start_datetime"].isna()
    
    # Rows with negative duration (always invalid/garbage)
    is_negative = ~df["duration_minutes"].isna() & (df["duration_minutes"] < 0)
    
    # Rows with implausible durations based on event type:
    # - unplanned: must be between 0 and 1440 minutes (24 hours)
    # - planned: must be between 0 and 43200 minutes (30 days)
    is_implausible_unplanned = (
        (df["event_type"] == "unplanned") & 
        (~df["duration_minutes"].isna()) & 
        (df["duration_minutes"] > 1440)
    )
    is_implausible_planned = (
        (df["event_type"] == "planned") & 
        (~df["duration_minutes"].isna()) & 
        (df["duration_minutes"] > 43200)
    )

    # Combine masks for dropping
    drop_mask = is_missing_start | is_negative | is_implausible_unplanned | is_implausible_planned

    # Generate reasons for drop log
    reasons = pd.Series(index=df.index, dtype="object")
    reasons[is_missing_start] = "Missing start_datetime"
    reasons[is_negative] = df.loc[is_negative, "duration_minutes"].apply(
        lambda x: f"Negative duration ({x:.2f} mins)"
    )
    reasons[is_implausible_unplanned] = df.loc[is_implausible_unplanned, "duration_minutes"].apply(
        lambda x: f"Unplanned duration exceeds 24 hours ({x:.2f} mins)"
    )
    reasons[is_implausible_planned] = df.loc[is_implausible_planned, "duration_minutes"].apply(
        lambda x: f"Planned duration exceeds 30 days ({x:.2f} mins)"
    )

    # Create dropped rows log
    dropped_df = df[drop_mask].copy()
    dropped_df["reason"] = reasons[drop_mask]

    # Cleaned dataframe contains all valid rows (including those with missing effective_end_datetime)
    cleaned_df = df[~drop_mask].copy()

    # Add has_duration boolean column
    cleaned_df["has_duration"] = ~cleaned_df["duration_minutes"].isna()

    # 6. Fill missing categorical variables
    # Fill missing corridor with "Non-corridor"
    print("Filling missing corridors with 'Non-corridor'...")
    cleaned_df["corridor"] = cleaned_df["corridor"].replace("NULL", "Non-corridor").fillna("Non-corridor")
    
    # Leave zone/junction nulls as explicit "Unknown"
    print("Filling missing zones and junctions with 'Unknown'...")
    cleaned_df["zone"] = cleaned_df["zone"].replace("NULL", "Unknown").fillna("Unknown")
    cleaned_df["junction"] = cleaned_df["junction"].replace("NULL", "Unknown").fillna("Unknown")

    # 7. Add derived columns based on local IST time
    print("Adding derived time features (hour of day, day of week, weekend flag) based on IST...")
    cleaned_df["hour_of_day"] = cleaned_df["start_datetime_ist"].dt.hour
    cleaned_df["day_of_week"] = cleaned_df["start_datetime_ist"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    cleaned_df["is_weekend"] = cleaned_df["day_of_week"].isin([5, 6])

    # Ensure processed directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 8. Save dataframes
    print(f"Saving cleaned dataset to: {CLEANED_DATA_PATH}")
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)

    print(f"Saving dropped rows log to: {DROPPED_LOG_PATH}")
    dropped_df.to_csv(DROPPED_LOG_PATH, index=False)

    # 9. Print Data Quality Summary
    rows_kept = len(cleaned_df)
    rows_dropped = len(dropped_df)
    
    has_duration_true = cleaned_df["has_duration"].sum()
    has_duration_false = (~cleaned_df["has_duration"]).sum()
    
    # Breakdown of has_duration=True by event_type
    duration_true_by_type = cleaned_df[cleaned_df["has_duration"]].groupby("event_type").size()
    
    print("\n" + "="*50)
    print("DATA QUALITY SUMMARY")
    print("="*50)
    print(f"Initial row count             : {initial_row_count}")
    print(f"Total rows retained (cleaned) : {rows_kept} ({rows_kept/initial_row_count*100:.2f}%)")
    print(f"Total rows dropped            : {rows_dropped} ({rows_dropped/initial_row_count*100:.2f}%)")
    
    print("\nBreakdown of dropped rows:")
    print(f"  - Missing start datetime    : {is_missing_start.sum()}")
    print(f"  - Negative duration         : {is_negative.sum()}")
    print(f"  - Implausible unplanned (>24h): {is_implausible_unplanned.sum()}")
    print(f"  - Implausible planned (>30d)  : {is_implausible_planned.sum()}")
    
    print("\nDuration details in retained rows:")
    print(f"  - has_duration = True       : {has_duration_true} ({has_duration_true/rows_kept*100:.2f}%)")
    print(f"  - has_duration = False      : {has_duration_false} ({has_duration_false/rows_kept*100:.2f}%)")
    
    print("\nBreakdown of has_duration=True rows by event_type:")
    for etype, count in duration_true_by_type.items():
        print(f"  - {etype:<25}   : {count}")
    
    print("\nNull percentage per key column in cleaned dataset:")
    key_columns = [
        "event_type", "event_cause", "requires_road_closure", "start_datetime",
        "effective_end_datetime", "priority", "corridor", "zone",
        "police_station", "junction", "latitude", "longitude"
    ]
    
    for col in key_columns:
        if col in cleaned_df.columns:
            null_count = cleaned_df[col].isna().sum()
            null_pct = (null_count / rows_kept) * 100
            print(f"  - {col:<27}: {null_pct:.2f}% ({null_count} nulls)")
        else:
            print(f"  - {col:<27}: NOT FOUND")
    
    print("="*50)
    print("Data cleaning completed successfully.")

if __name__ == "__main__":
    main()
