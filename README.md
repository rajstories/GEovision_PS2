<div align="center">

# 🚦 BTP Event-Driven Congestion Planner

### AI-assisted forecasting & resource deployment for planned & unplanned traffic events

**Gridlock Hackathon 2.0 — Round 2 · Theme: Event-Driven Congestion (Planned & Unplanned)**
**Hosted by Flipkart · In partnership with Bengaluru Traffic Police (BTP)**

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Prototype-yellow)
![Dataset](https://img.shields.io/badge/Dataset-ASTRAM%20%28BTP%29%20only-blue)

</div>

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Our Solution at a Glance](#-our-solution-at-a-glance)
- [Architecture Overview](#-architecture-overview)
- [Problem → Solution Mapping](#-problem--solution-mapping)
- [Key Design Philosophy](#-key-design-philosophy-explainability-over-false-precision)
- [Model Validation & Benchmarking](#-model-validation--benchmarking)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [How to Run](#-how-to-run)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [Dataset Compliance](#-dataset-compliance)

---

## 🧩 Problem Statement

> *Political rallies, festivals, sports events, construction activities, and sudden gatherings create localized traffic breakdowns. Event impact is not quantified in advance, resource deployment is experience-driven, and there is no post-event learning system.*
>
> **How can historical and real-time data be used to forecast event-related traffic impact and recommend optimal manpower, barricading, and diversion plans?**

Urban traffic networks face sudden, severe congestion from planned and unplanned events, yet traditional traffic models often rely on high-precision predictions that fail silently on rare or low-sample incidents. This project aggregates historical traffic incident data to forecast the impact of upcoming events and recommend concrete police resource deployments — while being explicit about when its evidence is thin.

## 🎯 Our Solution at a Glance

A pipeline that takes a real BTP traffic-event log, learns how different event types historically behave, and turns that into an interactive, explainable recommendation tool for traffic officers — including a feedback form that records actual outcomes to refine the historical baseline on the next data refresh.

<img width="1774" height="887" alt="image" src="https://github.com/user-attachments/assets/5a30dedd-0cbc-422d-a6a7-af7da43449db" />


## 🏗️ Architecture Overview

<img width="1693" height="929" alt="Architecture diagram" src="https://github.com/user-attachments/assets/dbb61cb7-40f6-449b-98e5-dac2cf3f9400" />

| Module | File | Responsibility |
|---|---|---|
| **Data Cleaning** | `src/data_cleaning.py` | Cleans raw ASTRAM event logs, applies type-aware duration caps (24h for unplanned events, 30 days for planned events), and flags rows lacking valid durations for categorical-only analysis |
| **Historical Aggregation** | `src/feature_engineering.py` | Groups cleaned events by spatial, temporal, and causal features; computes closure rates and duration statistics at fine and coarse resolution |
| **Impact Prediction** | `src/impact_model.py` | Cascades through historical match tiers to estimate event duration and road-closure probability, returning a structured confidence label |
| **Resource Recommendation** | `src/resource_engine.py` | Converts predicted severity into personnel counts, barricade placement, and diversion guidance, all via editable config tables and rules |
| **Traffic Advisory Generator** | `src/advisory_generator.py` | Auto-drafts a publishable BTP-style advisory, a beat-level (`police_station`) coordination note, and a short VMS-board message — the manual, experience-driven drafting step officers do by hand today |
| **Dashboard** | `app/dashboard.py` | Interactive Streamlit interface for submitting events, viewing recommendations, inspecting explainability flags, exporting the draft advisory, and reviewing supporting historical evidence |
| **Feedback Loop** | `src/feedback_loop.py` | Captures real-world outcomes from officers and logs them, to be merged into the historical baseline on the next data refresh |
| **Model Validation** | `src/model_validation.py` | Benchmarks the rule-based system against a RandomForest classifier on an identical, leakage-free, chronologically-split test set |

## 🔗 Problem → Solution Mapping

| Pain Point (from problem statement) | Module that Solves It |
|---|---|
| "Event impact is not quantified in advance" | **Impact Prediction** — 4-tier historical lookup, never silent, always returns a confidence-labeled estimate |
| "Resource deployment is experience-driven" | **Resource Recommendation + Advisory Generator** — auditable, rule-based personnel/barricade/diversion logic, plus an auto-drafted advisory that replaces the manual, experience-written advisory officers produce today |
| "No post-event learning system" | **Feedback Loop** — officers log actual outcomes, which are merged into the historical baseline on the next data refresh |

## 🧠 Key Design Philosophy: Explainability over False Precision

Rather than forcing a complex model to output a confident-looking number when data is scarce, this system prioritizes **explainable, rule-based reasoning with structured fallbacks** — every prediction tells you *how* confident it is and *why*.

### 4-Tier Fallback Mechanism

The system never returns an error on a new or rare scenario — it cascades:

1. **`fine`** — exact match on corridor + event cause + weekend status + hour bucket (≥5 historical events)
2. **`coarse`** — match on event cause + weekend status + hour bucket (drops corridor specificity)
3. **`cause_only`** — match solely on event cause (drops time and location)
4. **`no_data`** — explicitly returns "Unknown," never fabricates a number

### Closure-Rate-Priority Override (guarding against noisy data)

In low-sample categories (`cause_only` tier, <5 records), raw duration is often skewed by **administrative lag** — e.g., a VIP movement that physically lasted an hour but was marked "closed" in the system days later. Below this threshold, the model stops trusting duration entirely and instead derives severity from the historical road-closure rate and the event's native priority — both far more stable signals at small sample sizes.

> **Example — VIP Movement:** Historical logs show an average duration of ~6 days, almost certainly an administrative-cleanup artifact from only 4 ever-recorded events. Instead of outputting a false "High Severity," the override detects the low sample size, evaluates the historical closure rate and priority instead, and correctly drives a calibrated **Medium-severity** deployment recommendation — with the unreliable duration figure still displayed, clearly flagged, never hidden.

## 📊 Model Validation & Benchmarking

To verify this design choice isn't just philosophy but actually holds up, we benchmarked the rule-based system against a standard ML classifier on an **identical, leakage-free, chronologically split** test set (trained on the earliest 80% of events, tested on the most recent 20% — simulating real forecasting of unseen future events).

| Model | Accuracy | Macro F1-Score | Notes |
|---|:---:|:---:|---|
| Majority Baseline | 47.61% | 0.2150 | Always predicts the most common class |
| **Rule-Based System (ours)** | 52.04% | **0.4906** | Balanced across all severity classes |
| RandomForestClassifier | **58.23%** | 0.4719 | Higher raw accuracy, but only **6% recall** on the Medium class |

**Why this matters:** the RandomForest's higher accuracy is partly an artifact of class imbalance — it achieves it by almost completely ignoring the Medium-severity class. Our rule-based system's higher **macro-F1** shows it performs more evenly across all severity levels, which matters operationally: a model that quietly fails to flag medium-severity events is a worse deployment tool than one that's marginally less accurate but has no blind spots. Full methodology and results: [`docs/model_validation_results.md`](docs/model_validation_results.md).

### Why these numbers are trustworthy: leakage caught and corrected

An earlier high-capacity ensemble we prototyped (now preserved in
[`experiments/`](experiments/)) evaluated itself with **target leakage** — it
fit preprocessing on the full dataset before splitting and used a *random*
split, both of which inflate scores. We caught this and rebuilt the validation
harness to be leakage-free:

- **Chronological split, not random** — train on the earliest 80% of events,
  test on the most recent 20%, so the test set is genuinely "the future."
- **Train-only lookup tables** — the rule-based system is evaluated using a
  historical lookup built *only* from the training period, never the test rows.
- **Train-only corridor vocabulary** — the "top-15 corridors" feature for the
  RandomForest is learned from the training set alone.

**Honest caveat:** the severity labels used as ground truth are derived by
applying the system's own severity definition to each event's *actual* duration
and closure outcome. They therefore encode real post-event information the
pre-event predictor cannot see (so the benchmark is not circular), but the
labelling *function* is shared with the rule-based system, which may modestly
favour it. We report this openly rather than present the comparison as fully
model-agnostic.

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Data processing:** pandas
- **Modeling / validation:** scikit-learn
- **Interface:** Streamlit
- **Storage:** Flat-file CSV (zero infrastructure dependency — runs anywhere, no cloud lock-in)

## 📁 Project Structure

```
GEovision_PS2/
├── data/
│   ├── raw/                      # Original ASTRAM dataset (provided)
│   └── processed/                # Cleaned events, historical lookup (committed)
│                                 #   models/ is git-ignored (regenerated artifacts)
├── src/                          # Production pipeline (shipped solution)
│   ├── data_cleaning.py
│   ├── feature_engineering.py
│   ├── impact_model.py
│   ├── resource_engine.py
│   ├── advisory_generator.py
│   ├── feedback_loop.py
│   └── model_validation.py
├── app/
│   └── dashboard.py
├── experiments/                  # Approaches we tried but did NOT ship (see its README)
│   ├── README.md
│   ├── train_advanced_ml_model.py
│   └── save_model.py
├── docs/
│   ├── context.md
│   └── model_validation_results.md
├── requirements.txt
└── README.md
```

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/rajstories/GEovision_PS2.git
cd GEovision_PS2

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the dashboard
streamlit run app/dashboard.py
```

The app opens at `http://localhost:8501`. No API keys, no GPU required. The processed data files (`cleaned_events.csv`, `historical_lookup.csv`) are committed, so the dashboard runs out of the box. To regenerate them from the raw dataset:

```bash
python src/data_cleaning.py        # raw ASTRAM log -> cleaned_events.csv
python src/feature_engineering.py  # cleaned events -> historical_lookup.csv
```

## ⚠️ Known Limitations

- **Routine vs. high-security deployments** — Personnel recommendations are calibrated from routine traffic-management response patterns, not high-security VIP protocols (e.g., SPG or specialized convoy security), which follow independent force-deployment chains. This system is designed to *augment*, not replace, those processes for VIP-classified events.
- **Diversion routing** — The ASTRAM dataset contains no road-network graph or alternate-route data, so the system deliberately does **not** prescribe specific diversion routes (doing so would require outside geographic knowledge not in the data). It flags when a diversion is likely needed and points officers to BTP's Standard Diversion Plan for the corridor; the concrete route is left to on-ground judgment.
- **Data sparsity for rare causes** — Categories like `vip_movement` and `protest` have very few historical records; the system is explicitly transparent about this via confidence labels rather than disguising it.

## 🔮 Future Work

- **Live ASTRAM feed integration** — stream active incidents directly from the control room for real-time forecasting
- **Automated feedback retraining** — auto-rebuild the historical lookup tables whenever new feedback entries are logged
- **Road-network-aware routing** — integrate open routing services (e.g., OSRM) for dynamic, topology-aware diversion plans

## ✅ Dataset Compliance

The **shipped pipeline** (`src/` + `app/`) uses **only** the ASTRAM event dataset provided by HackerEarth/BTP for this round. No external datasets, APIs, or scraped data sources are read at any stage. Diversion guidance is intentionally limited to what the dataset supports — we do not hardcode named alternate routes, because road-network topology is not in the data.

> For full transparency: the abandoned experiments in [`experiments/`](experiments/) include a pretrained sentence-embedding model and a placeholder weather feature. These are **not part of the shipped solution** and are excluded precisely because they reach beyond the provided dataset.

---

<div align="center">

**Built for Gridlock Hackathon 2.0 · Round 2 · Theme: Event-Driven Congestion**

</div>
