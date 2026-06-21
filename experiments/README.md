# Experiments — approaches we tried but did NOT ship

This folder preserves modelling experiments that are **not** part of the
production system in [`../src/`](../src). They are kept for transparency: they
document what we investigated and why we deliberately shipped the simpler,
explainable rule-based pipeline instead.

> The production path is, and only is:
> `data_cleaning → feature_engineering → impact_model → resource_engine →
> advisory_generator → dashboard`, with `feedback_loop` closing the loop.
> Nothing in this folder is imported or loaded by the running app.

## `train_advanced_ml_model.py` — high-capacity stacking/voting ensemble

A much heavier model: XGBoost + LightGBM + CatBoost stacked/voted, with
sentence-embedding text features, geo-clustering, target encoding, SMOTE, and
Optuna tuning.

**Why we didn't ship it:**

1. **It did not earn its complexity.** On a fair, leakage-free, *chronological*
   split (train on the past, test on the most recent events — see
   [`../src/model_validation.py`](../src/model_validation.py)), a transparent
   rule-based lookup scored competitively on macro-F1 and, unlike the ensemble,
   never silently ignored the Medium-severity class.
2. **Its own evaluation had target leakage.** This script fits its target
   encoder, geo-clusters, and embeddings on the *full* dataset **before** the
   train/test split, and uses a *random* (not chronological) split. Both inflate
   its reported scores — the encoder effectively sees the test fold's targets,
   and a random split lets it "forecast" events using information from later
   ones. Catching this is precisely what motivated the clean, train-only,
   chronological validation harness we shipped.
3. **Explainability.** A police officer must be able to justify a deployment.
   "Based on N similar past events, median resolution 48 min" is defensible;
   a stacked-ensemble probability of 0.87 is not.
4. **Dataset scope.** It pulls a pretrained sentence-embedding model
   (`all-MiniLM-L6-v2`) and contains a placeholder `rainfall_mm` weather feature
   — neither is part of the provided ASTRAM dataset. Another reason it stays out
   of the shipped, dataset-only solution.

## `save_model.py` — full-data model export

Trains the validation RandomForest on the entire labelled dataset and pickles
it (plus preprocessing metadata) to disk. Useful as a portable artifact, but the
running app does not load a pickled model — it computes from the historical
lookup tables at request time — so this is not part of the shipped path either.

## Running these (optional)

They expect the project's `src/` on the import path and the heavy libraries
listed at the bottom of [`../requirements.txt`](../requirements.txt):

```bash
PYTHONPATH=src python experiments/train_advanced_ml_model.py
```

Their outputs (model `.pkl` files, metrics JSON) are git-ignored.
