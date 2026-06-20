#!/usr/bin/env python3
"""
Model Validation Script for ASTRAM Traffic Event Severity Classification

This script:
1. Builds a ground-truth labeled dataset from cleaned_events.csv (has_duration=True).
   True severity is determined using actual duration, actual requires_road_closure,
   and priority, reusing the logic from impact_model.py.
2. Formulates a feature matrix using only pre-event-known features.
3. Chronologically splits data (80% train / 20% test) by start_datetime.
4. Trains a standard RandomForestClassifier (n_estimators=100, max_depth=8).
5. Trains a balanced RandomForestClassifier (n_estimators=100, max_depth=8, class_weight='balanced').
6. Evaluates both RandomForest models (accuracy, per-class metrics, confusion matrices).
7. Evaluates the existing rule-based system on the identical test set.
8. Compares all approaches against a majority-class baseline.
9. Writes a markdown report to docs/model_validation_results.md and prints it.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add src/ to python path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))

from impact_model import (
    predict_impact,
    derive_severity_duration_primary,
    derive_severity_closure_priority,
    LOOKUP_PATH,
    CLEANED_PATH
)
from feature_engineering import (
    assign_hour_bucket,
    build_lookup_table
)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

def main():
    # 1. Load data
    print("Loading cleaned event data...")
    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(f"Cleaned events file not found at: {CLEANED_PATH}")
    
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
    if not os.path.exists(LOOKUP_PATH):
        raise FileNotFoundError(f"Historical lookup not found at: {LOOKUP_PATH}")
    overall_lookup = pd.read_csv(LOOKUP_PATH)
    
    # Compute ground-truth labels using the exact severity logic from impact_model.py
    # applied to actual outcomes.
    print("Computing ground truth severity labels...")
    true_severities = []
    
    for idx, row in df_dur.iterrows():
        # Query impact_model to see if the category is high/low reliability in history
        pred = predict_impact(
            event_cause=row["event_cause"],
            planned_start_datetime=row["start_datetime"],
            corridor=row["corridor"],
            requires_road_closure=row["requires_road_closure"],
            lookup_df=overall_lookup,
            cleaned_df=df
        )
        
        # Substitutue predictions with actual outcomes in the scoring logic
        if pred.duration_reliability == "high" and pd.notna(row["duration_minutes"]):
            sev, _, _ = derive_severity_duration_primary(
                row["duration_minutes"],
                row["requires_road_closure"]
            )
        else:
            # Low reliability: override with closure/priority-primary logic,
            # using true outcomes:
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
    initial_dur_len = len(df_dur)
    df_dur = df_dur[df_dur["true_severity"] != "Unknown"].copy()
    dropped_unknown = initial_dur_len - len(df_dur)
    if dropped_unknown > 0:
        print(f"Dropped {dropped_unknown} rows with 'Unknown' severity.")
    
    # Sort chronologically by start_datetime
    df_dur = df_dur.sort_values("start_datetime").reset_index(drop=True)
    
    # Chronological split (80% train, 20% test)
    total_samples = len(df_dur)
    split_idx = int(total_samples * 0.8)
    
    train_df = df_dur.iloc[:split_idx].copy()
    test_df = df_dur.iloc[split_idx:].copy()
    
    start_train_date = train_df["start_datetime"].min()
    end_train_date = train_df["start_datetime"].max()
    start_test_date = test_df["start_datetime"].min()
    end_test_date = test_df["start_datetime"].max()
    
    print(f"Split completed:")
    print(f"  Training set size : {len(train_df)} ({start_train_date} to {end_train_date})")
    print(f"  Testing set size  : {len(test_df)} ({start_test_date} to {end_test_date})")
    print(f"  Date cutoff used  : {end_train_date}")
    
    # ── Feature Engineering (Pre-event known fields only) ───────────────────────
    # Determine top 15 corridors on training set ONLY to prevent leakage
    top_15_corridors = train_df["corridor"].value_counts().index[:15].tolist()
    
    def clean_corridor(c):
        return c if c in top_15_corridors else "Other"
        
    # Map corridors
    train_df["corridor_clean"] = train_df["corridor"].apply(clean_corridor)
    test_df["corridor_clean"] = test_df["corridor"].apply(clean_corridor)
    
    # Add hour_bucket
    train_df["hour_bucket"] = train_df["hour_of_day"].apply(assign_hour_bucket)
    test_df["hour_bucket"] = test_df["hour_of_day"].apply(assign_hour_bucket)
    
    # Define features
    feature_cols = ["event_cause", "corridor_clean", "hour_bucket", "is_weekend", "requires_road_closure"]
    
    # Format target and features
    y_train = train_df["true_severity"]
    y_test = test_df["true_severity"]
    
    X_train_raw = train_df[feature_cols].copy()
    X_test_raw = test_df[feature_cols].copy()
    
    # One-hot encode categorical features
    categorical_cols = ["event_cause", "corridor_clean", "hour_bucket"]
    X_train = pd.get_dummies(X_train_raw, columns=categorical_cols)
    X_test = pd.get_dummies(X_test_raw, columns=categorical_cols)
    
    # Align test set columns to match training set exactly
    missing_cols = set(X_train.columns) - set(X_test.columns)
    for col in missing_cols:
        X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    # Ensure numeric types for ML model
    X_train = X_train.astype(float)
    X_test = X_test.astype(float)
    
    # ── Train RandomForestClassifier (Standard) ───────────────────────────────
    print("Training standard RandomForestClassifier...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    
    # Evaluate Standard RandomForest
    rf_acc = accuracy_score(y_test, rf_preds)
    rf_f1 = f1_score(y_test, rf_preds, average="macro")
    
    # ── Train RandomForestClassifier (Balanced) ───────────────────────────────
    print("Training balanced RandomForestClassifier...")
    rf_bal = RandomForestClassifier(n_estimators=100, max_depth=8, class_weight="balanced", random_state=42)
    rf_bal.fit(X_train, y_train)
    rf_bal_preds = rf_bal.predict(X_test)
    
    # Evaluate Balanced RandomForest
    rf_bal_acc = accuracy_score(y_test, rf_bal_preds)
    rf_bal_f1 = f1_score(y_test, rf_bal_preds, average="macro")
    
    # Calculate Medium class recall comparison
    classes = ["Low", "Medium", "High"]
    report_std = classification_report(y_test, rf_preds, labels=classes, output_dict=True)
    report_bal = classification_report(y_test, rf_bal_preds, labels=classes, output_dict=True)
    
    std_med_recall = report_std["Medium"]["recall"]
    bal_med_recall = report_bal["Medium"]["recall"]
    
    # ── Train-Only Lookup Table for Rule-Based System Evaluation ────────────────
    # Build a lookup table from the training set only to prevent target/test leakage
    print("Building training-only historical lookup table for rule-based model evaluation...")
    train_lookup = build_lookup_table(train_df)
    
    # ── Evaluate Existing Rule-Based System on Test Set ───────────────────────
    print("Evaluating existing rule-based system on test set...")
    rule_preds = []
    
    for idx, row in test_df.iterrows():
        # Evaluate using pre-event features, ignoring actual test-set duration
        pred = predict_impact(
            event_cause=row["event_cause"],
            planned_start_datetime=row["start_datetime"],
            corridor=row["corridor"],
            requires_road_closure=row["requires_road_closure"],
            lookup_df=train_lookup,
            cleaned_df=train_df
        )
        rule_preds.append(pred.severity_tier)
        
    rule_preds = np.array(rule_preds)
    rule_acc = accuracy_score(y_test, rule_preds)
    rule_f1 = f1_score(y_test, rule_preds, average="macro")
    
    # ── Evaluate Majority Baseline ────────────────────────────────────────────
    majority_class = train_df["true_severity"].mode()[0]
    baseline_preds = np.array([majority_class] * len(test_df))
    baseline_acc = accuracy_score(y_test, baseline_preds)
    baseline_f1 = f1_score(y_test, baseline_preds, average="macro")
    
    # ── Compile Markdown Report ───────────────────────────────────────────────
    report_lines = []
    report_lines.append("# Model Validation Report: Severity Prediction")
    report_lines.append("")
    report_lines.append("This document evaluates the ASTRAM event severity prediction models. It compares the **Majority Class Baseline**, the **Existing Rule-Based Fallback System**, and two configurations of the **RandomForestClassifier** (Standard vs Balanced Class Weights) on an identical, chronologically split validation set.")
    report_lines.append("")
    
    report_lines.append("## Dataset and Split Details")
    report_lines.append(f"- **Total Rows with Duration**: {total_samples}")
    report_lines.append(f"- **Training Set (First 80%)**: {len(train_df)} rows ({start_train_date.strftime('%Y-%m-%d %H:%M')} to {end_train_date.strftime('%Y-%m-%d %H:%M')})")
    report_lines.append(f"- **Testing Set (Recent 20%)**: {len(test_df)} rows ({start_test_date.strftime('%Y-%m-%d %H:%M')} to {end_test_date.strftime('%Y-%m-%d %H:%M')})")
    report_lines.append(f"- **Date Cutoff Used for Split**: `{end_train_date.strftime('%Y-%m-%d %H:%M')}`")
    report_lines.append("")
    
    # Summary Table
    report_lines.append("## Performance Summary")
    report_lines.append("")
    report_lines.append("| Model | Accuracy | Macro F1-Score | Lift over Baseline (Acc) |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    report_lines.append(f"| **Majority Baseline** | {baseline_acc:.2%} | {baseline_f1:.4f} | -- |")
    report_lines.append(f"| **Rule-Based System** | {rule_acc:.2%} | {rule_f1:.4f} | {rule_acc - baseline_acc:+.2%} |")
    report_lines.append(f"| **RandomForest (standard)** | {rf_acc:.2%} | {rf_f1:.4f} | {rf_acc - baseline_acc:+.2%} |")
    report_lines.append(f"| **RandomForest (balanced)** | {rf_bal_acc:.2%} | {rf_bal_f1:.4f} | {rf_bal_acc - baseline_acc:+.2%} |")
    report_lines.append("")
    
    # Balancing note
    report_lines.append("### Balancing Impact Analysis")
    if bal_med_recall > std_med_recall:
        recall_diff = bal_med_recall - std_med_recall
        f1_diff = rf_bal_f1 - rf_f1
        acc_diff = rf_bal_acc - rf_acc
        report_lines.append(
            f"Balancing class weights improved the Medium-class recall from `{std_med_recall:.1%}` to `{bal_med_recall:.1%}` (a lift of `+{recall_diff:.1%}`), "
            f"closing the recall gap but causing the overall accuracy to drop by `{acc_diff:+.1%}` while changing Macro-F1 by `{f1_diff:+.4f}`. "
            "This highlights a classic trade-off where the balanced model predicts Medium-severity items more often but incurs more false positives, reducing overall accuracy."
        )
    else:
        report_lines.append(
            f"Balancing class weights did not significantly fix the Medium-class recall gap (Standard: `{std_med_recall:.1%}`, Balanced: `{bal_med_recall:.1%}`). "
            "This suggests that Medium-severity events are genuinely harder to characterize and represent a fundamental property of the underlying data distribution, rather than a simple model bias quirk."
        )
    report_lines.append("")
    
    # Detailed Evaluation Reports
    report_lines.append("## RandomForest (standard) Detailed Evaluation")
    report_lines.append("```")
    report_lines.append(classification_report(y_test, rf_preds, labels=classes))
    report_lines.append("```")
    report_lines.append("")
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, rf_preds, labels=classes)
    report_lines.append("### RandomForest (standard) Confusion Matrix")
    report_lines.append("")
    report_lines.append("| Actual \\ Predicted | Low | Medium | High |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for i, row_label in enumerate(classes):
        report_lines.append(f"| **{row_label}** | {cm_rf[i, 0]} | {cm_rf[i, 1]} | {cm_rf[i, 2]} |")
    report_lines.append("")
    
    # Balanced RF Detailed Evaluation
    report_lines.append("## RandomForest (balanced) Detailed Evaluation")
    report_lines.append("```")
    report_lines.append(classification_report(y_test, rf_bal_preds, labels=classes))
    report_lines.append("```")
    report_lines.append("")
    
    # Balanced RF Confusion Matrix
    cm_rf_bal = confusion_matrix(y_test, rf_bal_preds, labels=classes)
    report_lines.append("### RandomForest (balanced) Confusion Matrix")
    report_lines.append("")
    report_lines.append("| Actual \\ Predicted | Low | Medium | High |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for i, row_label in enumerate(classes):
        report_lines.append(f"| **{row_label}** | {cm_rf_bal[i, 0]} | {cm_rf_bal[i, 1]} | {cm_rf_bal[i, 2]} |")
    report_lines.append("")
    
    # Rule-Based Detailed Evaluation
    report_lines.append("## Rule-Based System Detailed Evaluation")
    report_lines.append("```")
    report_lines.append(classification_report(y_test, rule_preds, labels=classes))
    report_lines.append("```")
    report_lines.append("")
    
    # Rule-Based Confusion Matrix
    cm_rule = confusion_matrix(y_test, rule_preds, labels=classes)
    report_lines.append("### Rule-Based Confusion Matrix")
    report_lines.append("")
    report_lines.append("| Actual \\ Predicted | Low | Medium | High |")
    report_lines.append("| :--- | :---: | :---: | :---: |")
    for i, row_label in enumerate(classes):
        report_lines.append(f"| **{row_label}** | {cm_rule[i, 0]} | {cm_rule[i, 1]} | {cm_rule[i, 2]} |")
    report_lines.append("")
    
    # Feature Importances
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    report_lines.append("## RandomForest (standard) Feature Importances")
    report_lines.append("Top features driving the standard RandomForest predictions (sorted descending):")
    report_lines.append("")
    report_lines.append("| Feature | Importance |")
    report_lines.append("| :--- | :---: |")
    for idx in indices:
        if importances[idx] > 0.0001:
            report_lines.append(f"| `{X_train.columns[idx]}` | {importances[idx]:.4f} |")
    report_lines.append("")
    
    report_content = "\n".join(report_lines)
    
    # Write to docs/model_validation_results.md
    docs_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
    os.makedirs(docs_dir, exist_ok=True)
    report_path = os.path.join(docs_dir, "model_validation_results.md")
    
    with open(report_path, "w") as f:
        f.write(report_content)
        
    # Print report to standard output
    print("\n" + "="*80)
    print("MODEL VALIDATION RESULTS")
    print("="*80)
    print(report_content)
    print("="*80)
    print(f"Results report successfully written to: {report_path}")

if __name__ == "__main__":
    main()
