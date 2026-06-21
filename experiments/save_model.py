#!/usr/bin/env python3
"""
Trained Model Serialization Script

This script trains the advanced RandomForestClassifier on the entire labeled dataset
and serializes the model and its preprocessing metadata (top 15 corridors list,
feature column layout) into 'download/' and 'downloads/' folders in the project structure.
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np
from datetime import datetime

# Add src/ to python path to import project modules
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from impact_model import (
    predict_impact,
    derive_severity_duration_primary,
    derive_severity_closure_priority,
    LOOKUP_PATH,
    CLEANED_PATH
)
from feature_engineering import (
    assign_hour_bucket
)
from sklearn.ensemble import RandomForestClassifier

def main():
    print("Loading data for model training...")
    df = pd.read_csv(CLEANED_PATH)
    
    # Coerce boolean types
    df["is_weekend"] = df["is_weekend"].astype(bool)
    df["has_duration"] = df["has_duration"].astype(bool)
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"])
    
    # Filter to has_duration=True rows
    df_dur = df[df["has_duration"] == True].copy()
    print(f"Total rows with duration: {len(df_dur)}")
    
    # Load historical lookup for the ground-truth labeling process
    overall_lookup = pd.read_csv(LOOKUP_PATH)
    
    # Compute ground-truth labels
    print("Computing ground truth severity labels...")
    true_severities = []
    
    for idx, row in df_dur.iterrows():
        pred = predict_impact(
            event_cause=row["event_cause"],
            planned_start_datetime=row["start_datetime"],
            corridor=row["corridor"],
            requires_road_closure=row["requires_road_closure"],
            lookup_df=overall_lookup,
            cleaned_df=df
        )
        
        if pred.duration_reliability == "high" and pd.notna(row["duration_minutes"]):
            sev, _, _ = derive_severity_duration_primary(
                row["duration_minutes"],
                row["requires_road_closure"]
            )
        else:
            true_closure_prob = 1.0 if row["requires_road_closure"] else 0.0
            sev, _, _ = derive_severity_closure_priority(
                true_closure_prob,
                row["priority"],
                row["requires_road_closure"],
                pred.duration_count
            )
        true_severities.append(sev)
        
    df_dur["true_severity"] = true_severities
    
    # Drop "Unknown" tier rows if any
    df_dur = df_dur[df_dur["true_severity"] != "Unknown"].copy()
    
    # Determine top 15 corridors on the full dataset to build the final model
    top_15_corridors = df_dur["corridor"].value_counts().index[:15].tolist()
    
    def clean_corridor(c):
        return c if c in top_15_corridors else "Other"
        
    df_dur["corridor_clean"] = df_dur["corridor"].apply(clean_corridor)
    df_dur["hour_bucket"] = df_dur["hour_of_day"].apply(assign_hour_bucket)
    
    # Define features
    feature_cols = ["event_cause", "corridor_clean", "hour_bucket", "is_weekend", "requires_road_closure"]
    
    y = df_dur["true_severity"]
    X_raw = df_dur[feature_cols].copy()
    
    # One-hot encode categorical features
    categorical_cols = ["event_cause", "corridor_clean", "hour_bucket"]
    X = pd.get_dummies(X_raw, columns=categorical_cols)
    X = X.astype(float)
    
    feature_columns_list = X.columns.tolist()
    
    # Train the final RandomForestClassifier
    print("Training final RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X, y)
    
    # Compile model package metadata
    metadata = {
        "top_15_corridors": top_15_corridors,
        "feature_columns": feature_columns_list,
        "categorical_cols": categorical_cols,
        "feature_cols_raw": feature_cols,
        "trained_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Create target directories
    os.makedirs("download", exist_ok=True)
    os.makedirs("downloads", exist_ok=True)
    
    # Serialize to download/
    print("Saving model files to 'download/'...")
    with open(os.path.join("download", "advanced_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join("download", "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    # Serialize to downloads/ (just in case)
    print("Saving model files to 'downloads/'...")
    with open(os.path.join("downloads", "advanced_model.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join("downloads", "model_metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("Trained model and preprocessing metadata successfully saved!")

if __name__ == "__main__":
    main()
