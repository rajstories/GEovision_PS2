# Binary Road-Closure Benchmark (Supporting)

> **This is a secondary, supporting benchmark — NOT the headline result.** The primary result is the 3-class severity benchmark in [`model_validation_results.md`](model_validation_results.md). This page answers a narrower, cleaner question: *can we predict, from pre-event information only, whether an event will require a road closure?*

## Methodology
- **Target:** `requires_road_closure` alone (the actual barricading/manpower trigger). We deliberately do **not** use the abandoned experiment's `closure OR duration>=60min` conflation.
- **Features (pre-event only):** `event_cause`, `corridor` (train-only top-15 vocabulary, rest → `Other`), `event_type`, `priority`, `hour_bucket`, `is_weekend`. Duration is **not** used (it is unknown before resolution), so all 7,331 cleaned rows are usable — not just the 2,822 with durations.
- **Split:** chronological 70/15/15 (train = oldest, test = newest). No random split. All preprocessing fit on **train only**; threshold tuned on **validation**.
- **Class balance:** the positive class (`requires_road_closure=True`) is only **6.8%** of events. A trivial "always predict no-closure" model scores ~93% accuracy, so accuracy alone is misleading — we lead with **PR-AUC, balanced accuracy, and F1**.

## Dataset and Split Details
- **Total rows:** 7331
- **Train:** 5131 rows (2023-11-09 19:24 to 2024-03-03 20:50), positives 6.1%
- **Validation:** 1100 rows (2024-03-03 21:50 to 2024-03-24 04:40), positives 9.3%
- **Test:** 1100 rows (2024-03-24 04:58 to 2024-04-08 17:11), positives 7.7%
- **Tuned thresholds (on validation):** RandomForest = 0.59, cause-rate baseline = 0.18

## Performance Summary (test fold)

| Model | PR-AUC | Balanced Acc | Precision | Recall | F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Majority baseline (always no-closure) | 0.0773 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.9227 |
| Cause-rate baseline | 0.2346 | 0.6998 | 0.3280 | 0.4824 | 0.3905 | 0.8836 |
| **RandomForest (ours)** | 0.2403 | 0.6762 | 0.2836 | 0.4471 | 0.3470 | 0.8700 |

- **Positive-class prevalence (test):** 0.0773 — this is the PR-AUC of a random classifier, the honest floor to beat.

### RandomForest — Detailed Classification Report (test)
```
              precision    recall  f1-score   support

  no_closure       0.95      0.91      0.93      1015
     closure       0.28      0.45      0.35        85

    accuracy                           0.87      1100
   macro avg       0.62      0.68      0.64      1100
weighted avg       0.90      0.87      0.88      1100

```

### RandomForest — Confusion Matrix (test)

| Actual \ Predicted | No closure | Closure |
| :--- | :---: | :---: |
| **No closure** | 919 | 96 |
| **Closure** | 47 | 38 |

### Interpretation
Road closures are rare and driven heavily by `event_cause` (e.g. `tree_fall`, `construction`, `procession` close roads far more often than `vehicle_breakdown`). The cause-rate baseline is therefore strong.
**Honest finding:** the RandomForest does **not** clearly beat the simple cause-rate heuristic (F1 lift `-0.0434`, PR-AUC lift `+0.0058`). In other words, most of the predictable structure in closure decisions is already captured by knowing the event cause. This *reinforces* the project's core thesis: for this data, a transparent, explainable signal is competitive with a black-box model — so we ship the explainable system and present this as supporting evidence, not as a headline ML win.
