#!/usr/bin/env python3
"""
Binary Road-Closure Benchmark (supporting evidence, NOT the headline result)

This is a SECOND, supporting benchmark. The primary, headline result remains the
3-class severity benchmark in docs/model_validation_results.md
(Rule-Based 52.04% acc / 0.4906 macro-F1). This script evaluates a cleaner,
narrower question:

    "Given only what is known BEFORE an event resolves, can we predict whether
     it will require a road closure?"

WHY THIS TARGET (and not the abandoned experiment's target):
  The abandoned experiment used `severe_impact = (requires_road_closure==1) OR
  (duration_minutes >= 60)`, which conflates two different operational decisions
  and silently labels the ~61% of events without a duration value as "not severe"
  unless closure is True — injecting heavy false-negative label noise. We instead
  use `requires_road_closure` ALONE:
    - it is the actual decision that drives barricading and manpower deployment;
    - it is known as a label without needing duration data, so we can use the
      full cleaned dataset (7,331 rows), not just the 2,822 with durations;
    - it does not conflate two distinct decisions.

WHY NO DURATION FEATURE:
  Duration is only known AFTER an event resolves, so using it to "forecast"
  closure would be leakage. We use only pre-event-known fields:
    event_cause, corridor (reduced to a train-only top-15 vocabulary),
    event_type, priority, hour_bucket, is_weekend.
  Because duration is neither the target nor a feature, we do NOT restrict to
  has_duration==True rows.

METHODOLOGY (leakage discipline, mirrors src/model_validation.py):
  - CHRONOLOGICAL 70/15/15 split (train = oldest, test = newest). Never random.
  - All preprocessing (corridor vocabulary, one-hot columns) is fit on TRAIN only.
  - The decision threshold is tuned on the VALIDATION fold, never on test.
  - The class is heavily imbalanced (~6.8% positive), so we report PR-AUC,
    balanced accuracy, and per-class precision/recall/F1 — not just accuracy
    (a trivial "always predict no-closure" model already scores ~93%).

BASELINES:
  1. Majority-class baseline: always predict "no closure".
  2. Cause-rate baseline: predict the historical closure RATE for that
     event_cause (computed on TRAIN only); threshold tuned on validation.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from feature_engineering import assign_hour_bucket  # noqa: E402

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLEANED_PATH = os.path.join("data", "processed", "cleaned_events.csv")
DOCS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
REPORT_PATH = os.path.join(DOCS_DIR, "binary_closure_benchmark_results.md")

RANDOM_STATE = 42
TOP_K_CORRIDORS = 15
FEATURE_COLS = ["event_cause", "corridor_clean", "event_type", "priority",
                "hour_bucket", "is_weekend"]
CATEGORICAL_COLS = ["event_cause", "corridor_clean", "event_type", "priority",
                    "hour_bucket"]


def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pick the probability threshold that maximizes positive-class F1."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.02, 0.99, 0.01):
        preds = (scores >= t).astype(int)
        f = f1_score(y_true, preds, zero_division=0)
        if f > best_f1:
            best_f1, best_t = f, t
    return float(best_t)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score=None) -> dict:
    """Compute the imbalance-aware metric bundle for a single model."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": (average_precision_score(y_true, y_score)
                   if y_score is not None else float("nan")),
    }


def main() -> None:
    # Windows consoles default to cp1252 and choke on unicode (→, —) in the
    # printed report; the file itself is always written as UTF-8.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    if not os.path.exists(CLEANED_PATH):
        raise FileNotFoundError(
            f"Cleaned events file not found at: {CLEANED_PATH}. "
            "Run src/data_cleaning.py first."
        )

    print("Loading cleaned events...")
    df = pd.read_csv(CLEANED_PATH)
    df["requires_road_closure"] = df["requires_road_closure"].astype(bool)
    df["is_weekend"] = df["is_weekend"].astype(bool)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df = df[df["start_datetime"].notna()].copy()

    # Target
    df["y"] = df["requires_road_closure"].astype(int)
    df["hour_bucket"] = df["hour_of_day"].apply(assign_hour_bucket)

    # ── Chronological 70/15/15 split ──────────────────────────────────────────
    df = df.sort_values("start_datetime").reset_index(drop=True)
    n = len(df)
    i_train = int(n * 0.70)
    i_val = int(n * 0.85)
    train_df = df.iloc[:i_train].copy()
    val_df = df.iloc[i_train:i_val].copy()
    test_df = df.iloc[i_val:].copy()

    # ── Train-only corridor vocabulary ────────────────────────────────────────
    top_corridors = train_df["corridor"].value_counts().index[:TOP_K_CORRIDORS].tolist()

    def clean_corridor(c):
        return c if c in top_corridors else "Other"

    for frame in (train_df, val_df, test_df):
        frame["corridor_clean"] = frame["corridor"].apply(clean_corridor)

    # ── One-hot encode, aligning all folds to the TRAIN columns ───────────────
    X_train = pd.get_dummies(train_df[FEATURE_COLS], columns=CATEGORICAL_COLS).astype(float)

    def encode(frame: pd.DataFrame) -> pd.DataFrame:
        X = pd.get_dummies(frame[FEATURE_COLS], columns=CATEGORICAL_COLS)
        for col in set(X_train.columns) - set(X.columns):
            X[col] = 0
        return X[X_train.columns].astype(float)

    X_val, X_test = encode(val_df), encode(test_df)
    y_train, y_val, y_test = train_df["y"].values, val_df["y"].values, test_df["y"].values

    # ── Model: class-weighted RandomForest (imbalance-aware) ──────────────────
    print("Training RandomForest (class_weight='balanced')...")
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    val_scores = rf.predict_proba(X_val)[:, 1]
    test_scores = rf.predict_proba(X_test)[:, 1]
    rf_threshold = _best_f1_threshold(y_val, val_scores)
    rf_pred = (test_scores >= rf_threshold).astype(int)
    rf_metrics = _metrics(y_test, rf_pred, test_scores)

    # ── Baseline 1: majority class (always "no closure") ──────────────────────
    maj_pred = np.zeros_like(y_test)
    maj_metrics = _metrics(y_test, maj_pred, np.zeros_like(y_test, dtype=float))

    # ── Baseline 2: per-cause historical closure rate (train-only) ────────────
    cause_rate = train_df.groupby("event_cause")["y"].mean()
    global_rate = float(train_df["y"].mean())
    val_cause_scores = val_df["event_cause"].map(cause_rate).fillna(global_rate).values
    test_cause_scores = test_df["event_cause"].map(cause_rate).fillna(global_rate).values
    cause_threshold = _best_f1_threshold(y_val, val_cause_scores)
    cause_pred = (test_cause_scores >= cause_threshold).astype(int)
    cause_metrics = _metrics(y_test, cause_pred, test_cause_scores)

    # ── Build markdown report ─────────────────────────────────────────────────
    pos_rate_overall = df["y"].mean()
    lines: list[str] = []
    lines.append("# Binary Road-Closure Benchmark (Supporting)")
    lines.append("")
    lines.append("> **This is a secondary, supporting benchmark — NOT the headline result.** "
                 "The primary result is the 3-class severity benchmark in "
                 "[`model_validation_results.md`](model_validation_results.md). "
                 "This page answers a narrower, cleaner question: *can we predict, from "
                 "pre-event information only, whether an event will require a road closure?*")
    lines.append("")
    lines.append("## Methodology")
    lines.append("- **Target:** `requires_road_closure` alone (the actual barricading/manpower "
                 "trigger). We deliberately do **not** use the abandoned experiment's "
                 "`closure OR duration>=60min` conflation.")
    lines.append("- **Features (pre-event only):** `event_cause`, `corridor` (train-only top-15 "
                 "vocabulary, rest → `Other`), `event_type`, `priority`, `hour_bucket`, "
                 "`is_weekend`. Duration is **not** used (it is unknown before resolution), so "
                 "all 7,331 cleaned rows are usable — not just the 2,822 with durations.")
    lines.append("- **Split:** chronological 70/15/15 (train = oldest, test = newest). No random "
                 "split. All preprocessing fit on **train only**; threshold tuned on **validation**.")
    lines.append(f"- **Class balance:** the positive class (`requires_road_closure=True`) is only "
                 f"**{pos_rate_overall*100:.1f}%** of events. A trivial \"always predict no-closure\" "
                 f"model scores ~{(1-pos_rate_overall)*100:.0f}% accuracy, so accuracy alone is "
                 f"misleading — we lead with **PR-AUC, balanced accuracy, and F1**.")
    lines.append("")
    lines.append("## Dataset and Split Details")
    lines.append(f"- **Total rows:** {n}")
    lines.append(f"- **Train:** {len(train_df)} rows "
                 f"({train_df['start_datetime'].min():%Y-%m-%d %H:%M} to "
                 f"{train_df['start_datetime'].max():%Y-%m-%d %H:%M}), "
                 f"positives {y_train.mean()*100:.1f}%")
    lines.append(f"- **Validation:** {len(val_df)} rows "
                 f"({val_df['start_datetime'].min():%Y-%m-%d %H:%M} to "
                 f"{val_df['start_datetime'].max():%Y-%m-%d %H:%M}), "
                 f"positives {y_val.mean()*100:.1f}%")
    lines.append(f"- **Test:** {len(test_df)} rows "
                 f"({test_df['start_datetime'].min():%Y-%m-%d %H:%M} to "
                 f"{test_df['start_datetime'].max():%Y-%m-%d %H:%M}), "
                 f"positives {y_test.mean()*100:.1f}%")
    lines.append(f"- **Tuned thresholds (on validation):** RandomForest = {rf_threshold:.2f}, "
                 f"cause-rate baseline = {cause_threshold:.2f}")
    lines.append("")
    lines.append("## Performance Summary (test fold)")
    lines.append("")
    lines.append("| Model | PR-AUC | Balanced Acc | Precision | Recall | F1 | Accuracy |")
    lines.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")

    def row(name, m):
        prauc = f"{m['pr_auc']:.4f}" if not np.isnan(m["pr_auc"]) else "n/a"
        return (f"| {name} | {prauc} | {m['balanced_accuracy']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | "
                f"{m['accuracy']:.4f} |")

    lines.append(row("Majority baseline (always no-closure)", maj_metrics))
    lines.append(row("Cause-rate baseline", cause_metrics))
    lines.append(row("**RandomForest (ours)**", rf_metrics))
    lines.append("")
    lines.append(f"- **Positive-class prevalence (test):** {y_test.mean():.4f} — this is the "
                 f"PR-AUC of a random classifier, the honest floor to beat.")
    lines.append("")
    lines.append("### RandomForest — Detailed Classification Report (test)")
    lines.append("```")
    lines.append(classification_report(y_test, rf_pred,
                                       target_names=["no_closure", "closure"],
                                       zero_division=0))
    lines.append("```")
    lines.append("")
    cm = confusion_matrix(y_test, rf_pred)
    lines.append("### RandomForest — Confusion Matrix (test)")
    lines.append("")
    lines.append("| Actual \\ Predicted | No closure | Closure |")
    lines.append("| :--- | :---: | :---: |")
    lines.append(f"| **No closure** | {cm[0, 0]} | {cm[0, 1]} |")
    lines.append(f"| **Closure** | {cm[1, 0]} | {cm[1, 1]} |")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("Road closures are rare and driven heavily by `event_cause` "
                 "(e.g. `tree_fall`, `construction`, `procession` close roads far more often "
                 "than `vehicle_breakdown`). The cause-rate baseline is therefore strong.")
    f1_lift = rf_metrics["f1"] - cause_metrics["f1"]
    prauc_lift = rf_metrics["pr_auc"] - cause_metrics["pr_auc"]
    if f1_lift <= 0.01 and prauc_lift <= 0.01:
        lines.append(
            f"**Honest finding:** the RandomForest does **not** clearly beat the simple "
            f"cause-rate heuristic (F1 lift `{f1_lift:+.4f}`, PR-AUC lift `{prauc_lift:+.4f}`). "
            "In other words, most of the predictable structure in closure decisions is already "
            "captured by knowing the event cause. This *reinforces* the project's core thesis: "
            "for this data, a transparent, explainable signal is competitive with a black-box "
            "model — so we ship the explainable system and present this as supporting evidence, "
            "not as a headline ML win."
        )
    else:
        lines.append(
            f"The RandomForest adds measurable lift over the cause-rate baseline "
            f"(F1 `{f1_lift:+.4f}`, PR-AUC `{prauc_lift:+.4f}`). We still treat this as "
            "supporting evidence, not the headline result, and report it without accuracy "
            "inflation."
        )
    lines.append("")

    os.makedirs(DOCS_DIR, exist_ok=True)
    report = "\n".join(lines)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 70)
    print(report)
    print("=" * 70)
    print(f"\nReport written to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
