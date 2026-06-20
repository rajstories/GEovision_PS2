# BTP Event-Driven Congestion Planner

## 1. Problem Statement
Urban traffic networks face sudden, severe congestion from planned and unplanned events, yet traditional traffic models often rely on high-precision predictions that fail on rare or low-sample incidents. This system aggregates historical traffic incident data to predict the impact of upcoming events and recommend police resource deployments. By utilizing structured fallbacks and prioritizing explainable metrics, the planner helps traffic authorities make robust, defensible deployment decisions even with sparse baseline data.

## 2. Architecture Overview
<img width="1693" height="929" alt="image" src="https://github.com/user-attachments/assets/dbb61cb7-40f6-449b-98e5-dac2cf3f9400" />

* **Data Cleaning ([data_cleaning.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/src/data_cleaning.py))**: Cleans raw event logs, applies type-specific duration caps (24 hours for unplanned events, 30 days for planned events), and flags rows without valid durations for partial categorical analysis.
* **Historical Aggregation ([feature_engineering.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/src/feature_engineering.py))**: Groups cleaned events by spatial, temporal, and cause features, computing closure rates and duration statistics at fine and coarse resolution tiers.
* **Impact Prediction ([impact_model.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/src/impact_model.py))**: Cascades through historical levels to estimate event duration and road closure probability for an upcoming event, returning structured confidence labels.
* **Resource Recommendation ([resource_engine.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/src/resource_engine.py))**: Translates predicted severity and closure likelihood into personnel counts, barricade placements, and diversion route suggestions using editable configurations.
* **Dashboard ([dashboard.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/app/dashboard.py))**: Provides an interactive Streamlit interface for traffic managers to submit new events, inspect recommendations, view explainability flags, and review historical evidence.
* **Feedback Loop ([feedback_loop.py](file:///c:/Users/thera/Downloads/flipkart_Gridlock/src/feedback_loop.py))**: Captures real-world event outcomes from officers on the ground and appends them to the cleaned baseline dataset to continuously refine predictions.

## 3. Key Design Philosophy: Explainability over False Precision
Rather than forcing a complex machine learning model to output potentially inaccurate numbers when data is scarce, this project prioritizes **explainable, rule-based reasoning and structured fallback mechanisms**.

* **4-Tier Fallback Mechanism**: To ensure the system always provides a useful prediction instead of returning an error on new or rare scenarios, the lookup cascades through:
  1. `fine`: Exact match on corridor + event cause + weekend status + hour bucket (requires $\ge$ 5 historical events).
  2. `coarse`: Match on event cause + weekend status + hour bucket (ignores specific corridor).
  3. `cause_only`: Match solely on event cause (ignores time and location).
  4. `no_data`: Default baseline fallback when no historical record of the event cause exists.

* **Closure-Rate-Priority Override**: In low-sample categories (specifically the `cause_only` tier with $< 5$ samples), duration estimates are often skewed by administrative lag (e.g., a short VIP movement marked as completed days late in the database). Under this override, the model ignores the unreliable duration statistics and instead determines severity based on the historical road closure rate and the event's native priority.
  * *Example (VIP Movement)*: A VIP movement might show an average duration of 6 days in historical logs due to administrative cleanup delays. Instead of outputting a false "High Severity" based on this duration, the override detects the low sample size, evaluates the high closure rate and high native priority, and correctly classifies the event under a rule-based logic that triggers a structured, moderate-severity personnel deployment.

## 4. How to Run
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Start the interactive dashboard:
   ```bash
   streamlit run app/dashboard.py
   ```

## 5. Known Limitations
* **Routine vs. High-Security Deployments**: Personnel recommendations are calibrated from routine traffic management responses. They do not reflect high-security VIP protocols (such as SPG or specialized security convoys), which follow independent force-deployment chains. This system is designed to augment, not replace, those processes for VIP-classified events.
* **Corridor-Level Diversions**: Diversion suggestions are corridor-level hints rather than precise road-network routing. This is due to a lack of live network topology and real-time congestion data within the current dataset.

## 6. Future Work
* **Live ASTRAM Feed Integration**: Automate real-time event ingestion by streaming active incidents directly from the ASTRAM control room API.
* **Automated Feedback Retraining**: Set up a pipeline to automatically re-run feature engineering and rebuild the lookup tables whenever new feedback entries are logged.
* **Road-Network-Aware Routing**: Integrate open routing services (e.g., OSRM) to compute dynamic diversion routes based on active street networks and live congestion bottlenecks.
