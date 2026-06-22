# Model Validation Report: Severity Prediction

This document evaluates the ASTRAM event severity prediction models. It compares the **Majority Class Baseline**, the **Existing Rule-Based Fallback System**, and two configurations of the **RandomForestClassifier** (Standard vs Balanced Class Weights) on an identical, chronologically split validation set.

> **Methodology note (read this first).** Safeguards that make these numbers
> trustworthy, plus one honest caveat:
> - **Leakage-free, chronological split.** Models are trained on the earliest
>   80% of events and tested on the most recent 20%. The rule-based system's
>   historical lookup and the RandomForest's corridor vocabulary are built from
>   the **training period only**, never the test rows.
> - **Labels are train-only too.** Ground-truth severity labels are derived
>   from each event's *actual* outcome (true duration, true closure, priority).
>   The only lookup-dependent input — the duration-reliability tier decision —
>   is drawn from the **train-only** lookup, so no test-period statistics enter
>   the labels.
> - **Honest caveat.** The labelling *function* is the system's own severity
>   definition applied to real outcomes, so the comparison modestly favours the
>   rule-based system. We disclose this rather than present it as fully
>   model-agnostic.

## Dataset and Split Details
- **Total Rows with Duration**: 2822
- **Training Set (First 80%)**: 2257 rows (2023-11-09 19:24 to 2024-03-14 01:28)
- **Testing Set (Recent 20%)**: 565 rows (2024-03-14 01:34 to 2024-04-08 11:18)
- **Date Cutoff Used for Split**: `2024-03-14 01:28`

## Performance Summary

| Model | Accuracy | Macro F1-Score | Lift over Baseline (Acc) |
| :--- | :---: | :---: | :---: |
| **Majority Baseline** | 47.61% | 0.2150 | -- |
| **Rule-Based System** | 52.04% | 0.4906 | +4.42% |
| **RandomForest (standard)** | 58.23% | 0.4719 | +10.62% |
| **RandomForest (balanced)** | 52.39% | 0.4471 | +4.78% |

### Balancing Impact Analysis
Balancing class weights improved the Medium-class recall from `6.4%` to `8.3%` (a lift of `+1.9%`), closing the recall gap but causing the overall accuracy to drop by `-5.8%` while changing Macro-F1 by `-0.0248`. This highlights a classic trade-off where the balanced model predicts Medium-severity items more often but incurs more false positives, reducing overall accuracy.

## RandomForest (standard) Detailed Evaluation
```
              precision    recall  f1-score   support

         Low       0.60      0.84      0.70       269
      Medium       0.48      0.06      0.11       157
        High       0.55      0.66      0.60       139

    accuracy                           0.58       565
   macro avg       0.54      0.52      0.47       565
weighted avg       0.55      0.58      0.51       565

```

### RandomForest (standard) Confusion Matrix

| Actual \ Predicted | Low | Medium | High |
| :--- | :---: | :---: | :---: |
| **Low** | 227 | 3 | 39 |
| **Medium** | 110 | 10 | 37 |
| **High** | 39 | 8 | 92 |

## RandomForest (balanced) Detailed Evaluation
```
              precision    recall  f1-score   support

         Low       0.60      0.65      0.63       269
      Medium       0.28      0.08      0.13       157
        High       0.47      0.78      0.59       139

    accuracy                           0.52       565
   macro avg       0.45      0.50      0.45       565
weighted avg       0.48      0.52      0.48       565

```

### RandomForest (balanced) Confusion Matrix

| Actual \ Predicted | Low | Medium | High |
| :--- | :---: | :---: | :---: |
| **Low** | 175 | 26 | 68 |
| **Medium** | 91 | 13 | 53 |
| **High** | 24 | 7 | 108 |

## Rule-Based System Detailed Evaluation
```
              precision    recall  f1-score   support

         Low       0.60      0.65      0.62       269
      Medium       0.30      0.26      0.28       157
        High       0.57      0.57      0.57       139

    accuracy                           0.52       565
   macro avg       0.49      0.49      0.49       565
weighted avg       0.51      0.52      0.51       565

```

### Rule-Based Confusion Matrix

| Actual \ Predicted | Low | Medium | High |
| :--- | :---: | :---: | :---: |
| **Low** | 174 | 64 | 31 |
| **Medium** | 88 | 41 | 28 |
| **High** | 28 | 32 | 79 |

## RandomForest (standard) Feature Importances
Top features driving the standard RandomForest predictions (sorted descending):

| Feature | Importance |
| :--- | :---: |
| `event_cause_vehicle_breakdown` | 0.2190 |
| `event_cause_construction` | 0.1793 |
| `requires_road_closure` | 0.1717 |
| `corridor_clean_ORR East 2` | 0.0692 |
| `is_weekend` | 0.0325 |
| `event_cause_others` | 0.0260 |
| `event_cause_accident` | 0.0255 |
| `hour_bucket_morning` | 0.0253 |
| `event_cause_tree_fall` | 0.0224 |
| `hour_bucket_night` | 0.0220 |
| `event_cause_public_event` | 0.0196 |
| `event_cause_water_logging` | 0.0183 |
| `corridor_clean_Tumkur Road` | 0.0142 |
| `hour_bucket_afternoon` | 0.0139 |
| `corridor_clean_Non-corridor` | 0.0139 |
| `event_cause_road_conditions` | 0.0132 |
| `corridor_clean_Other` | 0.0112 |
| `corridor_clean_ORR East 1` | 0.0110 |
| `event_cause_congestion` | 0.0101 |
| `corridor_clean_Mysore Road` | 0.0089 |
| `hour_bucket_evening` | 0.0084 |
| `corridor_clean_ORR West 1` | 0.0079 |
| `corridor_clean_Bellary Road 1` | 0.0073 |
| `corridor_clean_Bellary Road 2` | 0.0069 |
| `corridor_clean_Old Madras Road` | 0.0060 |
| `corridor_clean_ORR North 1` | 0.0058 |
| `corridor_clean_West of Chord Road` | 0.0049 |
| `corridor_clean_Hosur Road` | 0.0046 |
| `event_cause_pot_holes` | 0.0039 |
| `corridor_clean_Magadi Road` | 0.0036 |
| `corridor_clean_Old Airport Road` | 0.0031 |
| `event_cause_vip_movement` | 0.0028 |
| `event_cause_protest` | 0.0027 |
| `corridor_clean_ORR North 2` | 0.0022 |
| `event_cause_procession` | 0.0022 |
| `event_cause_test_demo` | 0.0005 |
