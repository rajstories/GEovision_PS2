import streamlit as st
import pandas as pd
from datetime import datetime, time
import sys
import os

# Add src folder to python path to resolve sibling imports cleanly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from impact_model import predict_impact
from resource_engine import generate_recommendation
from feedback_loop import log_actual_outcome
from advisory_generator import generate_traffic_advisory

# Page configuration
st.set_page_config(
    page_title="BTP Event-Driven Congestion Planner",
    page_icon="🚦",
    layout="wide"
)

# Inject minimal, clean custom CSS for premium look
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap');
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    body, p, div, label {
        font-family: 'Inter', sans-serif;
    }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 600;
        color: #212529;
    }
    div[data-testid="metric-container"] {
        background-color: #f8f9fa;
        padding: 12px 18px;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data
def load_data():
    """Load and prepare cleaned events and lookup table with caching."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    cleaned_path = os.path.join(base_dir, "data", "processed", "cleaned_events.csv")
    lookup_path = os.path.join(base_dir, "data", "processed", "historical_lookup.csv")
    
    if not os.path.exists(cleaned_path) or not os.path.exists(lookup_path):
        raise FileNotFoundError(
            "Processed data files not found. Please ensure src/data_cleaning.py and "
            "src/feature_engineering.py have been executed first."
        )
        
    cleaned_df = pd.read_csv(cleaned_path)
    cleaned_df["has_duration"] = cleaned_df["has_duration"].astype(bool)
    cleaned_df["requires_road_closure"] = cleaned_df["requires_road_closure"].astype(bool)
    
    lookup_df = pd.read_csv(lookup_path)
    
    return cleaned_df, lookup_df

# Load data safely
try:
    cleaned_df, lookup_df = load_data()
    data_loaded_successfully = True
except Exception as e:
    st.error(f"Error loading datasets: {e}")
    data_loaded_successfully = False

if data_loaded_successfully:
    # ─── SIDEBAR: Overall Dataset Statistics ────────────────────────────────────
    st.sidebar.title("🚦 BTP Congestion Planner")
    st.sidebar.markdown("### Dataset Overview")
    
    total_events = len(cleaned_df)
    st.sidebar.metric("Total Events Logged", f"{total_events:,}")
    
    # Planned vs Unplanned Split
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Event Type Split")
    planned_count = (cleaned_df["event_type"] == "planned").sum()
    unplanned_count = (cleaned_df["event_type"] == "unplanned").sum()
    planned_pct = (planned_count / total_events) * 100
    unplanned_pct = (unplanned_count / total_events) * 100
    
    col_s1, col_s2 = st.sidebar.columns(2)
    col_s1.metric("Planned", f"{planned_pct:.1f}%", f"{planned_count}")
    col_s2.metric("Unplanned", f"{unplanned_pct:.1f}%", f"{unplanned_count}")
    
    # Top 5 Corridors (excluding Non-corridor for clarity, or including it)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Top 5 Corridors")
    # We include all corridors to be faithful to the data, but sort by count
    top_corridors = cleaned_df["corridor"].value_counts().head(5)
    top_corridors_df = top_corridors.reset_index()
    top_corridors_df.columns = ["Corridor", "Count"]
    st.sidebar.bar_chart(top_corridors_df.set_index("Corridor"))


    # ─── MAIN PANEL: Form and Prediction Output ───────────────────────────────
    st.title("Congestion Prediction & Deployment Planner")
    st.markdown(
        "Generate traffic impact forecasts and manpower deployment plans "
        "for upcoming road events based on historical BTP logs."
    )
    
    # Input Form Container
    st.markdown("### Input Event Details")
    with st.form("event_details_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Cause Dropdown
            unique_causes = sorted(cleaned_df["event_cause"].dropna().unique())
            event_cause = st.selectbox(
                "Event Cause / Type",
                options=unique_causes,
                format_func=lambda x: x.replace("_", " ").title()
            )
            
            # Corridor Dropdown (optional "Unknown")
            raw_corridors = cleaned_df["corridor"].dropna().unique().tolist()
            if "Non-corridor" in raw_corridors:
                raw_corridors.remove("Non-corridor")
            unique_corridors = ["Unknown"] + sorted(raw_corridors)
            
            corridor_input = st.selectbox(
                "Corridor (Optional)",
                options=unique_corridors,
                index=0
            )
            # Map "Unknown" back to None so it gets mapped to Non-corridor downstream
            corridor = None if corridor_input == "Unknown" else corridor_input
            
        with col2:
            # Date & Time picker
            col_date, col_time = st.columns(2)
            with col_date:
                selected_date = st.date_input("Event Date", value=datetime.today())
            with col_time:
                selected_time = st.time_input("Event Start Time", value=datetime.now().time())
            
            planned_start_datetime = datetime.combine(selected_date, selected_time)
            
            # Road closure expected checkbox
            requires_road_closure = st.checkbox("Road Closure Expected / Required")
            
        submit_button = st.form_submit_button("Generate Prediction & Plan")

    # Output Rendering
    if submit_button:
        try:
            # Execute prediction pipeline inside try/except block to catch anomalous configurations
            impact = predict_impact(
                event_cause=event_cause,
                planned_start_datetime=planned_start_datetime,
                corridor=corridor,
                requires_road_closure=requires_road_closure,
                lookup_df=lookup_df,
                cleaned_df=cleaned_df
            )
            
            rec = generate_recommendation(
                impact=impact,
                event_cause=event_cause,
                corridor=corridor,
                junction=None,
                address=None,
                requires_road_closure=requires_road_closure
            )
            
            st.markdown("---")
            st.markdown("### Prediction & Recommendation Output")
            
            # 1. Severity Tier colored badge
            if impact.severity_tier == "Low":
                bg_color, text_color, border_color = "#E8F5E9", "#2E7D32", "#C8E6C9"
            elif impact.severity_tier == "Medium":
                bg_color, text_color, border_color = "#FFF3E0", "#EF6C00", "#FFE0B2"
            elif impact.severity_tier == "High":
                bg_color, text_color, border_color = "#FFEBEE", "#C62828", "#FFCDD2"
            else:
                bg_color, text_color, border_color = "#ECEFF1", "#37474F", "#CFD8DC"
                
            badge_html = f"""
            <div style="
                background-color: {bg_color}; 
                color: {text_color}; 
                border: 1px solid {border_color}; 
                padding: 10px 20px; 
                border-radius: 6px; 
                font-weight: bold; 
                font-size: 20px; 
                display: inline-block;
                text-align: center;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            ">
                {impact.severity_tier} Severity
            </div>
            """
            st.markdown(badge_html, unsafe_allow_html=True)
            
            # 2. Small caption under the badge showing severity_basis in plain words
            if impact.severity_basis == "duration":
                basis_caption = "Based on duration history"
            elif impact.severity_basis == "closure_priority":
                basis_caption = "Based on closure rate & priority (limited duration data)"
            else:
                basis_caption = "Insufficient historical data for duration or closure prediction"
                
            st.markdown(
                f"<div style='font-size: 13px; color: #6c757d; margin-top: 8px; margin-bottom: 24px; font-style: italic;'>"
                f"{basis_caption}</div>", 
                unsafe_allow_html=True
            )
            
            # 3. Expected duration and closure probability metrics
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                if impact.duration_reliability == "low":
                    dur_str = f"{impact.expected_duration_minutes:.0f} min" if impact.expected_duration_minutes is not None else "Unknown"
                    st.markdown(
                        f"""
                        <div style="
                            padding: 12px 18px; 
                            border: 1px solid #e9ecef; 
                            border-radius: 8px; 
                            background-color: #f8f9fa;
                            min-height: 85px;
                        ">
                            <span style="font-size: 14px; color: rgba(49, 51, 63, 0.6); display: block; margin-bottom: 4px;">Expected Duration</span>
                            <span style="font-size: 28px; color: #868e96; font-weight: 500;">{dur_str}</span>
                            <span style="font-size: 12px; color: #c62828; margin-left: 8px; font-weight: 600;">(low reliability)</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                else:
                    dur_val = f"{impact.expected_duration_minutes:.0f} min" if impact.expected_duration_minutes is not None else "Unknown"
                    st.metric(label="Expected Duration", value=dur_val)
                    
            with m_col2:
                prob_val = f"{impact.road_closure_probability * 100:.1f}%" if impact.road_closure_probability is not None else "Unknown"
                st.metric(label="Road Closure Probability", value=prob_val)
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 4. Recommended personnel count, barricade hint, diversion note
            col_p, col_b, col_d = st.columns(3)
            with col_p:
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 5px solid #007bff; min-height: 125px;">
                        <strong style="color: #495057; font-size: 13px; text-transform: uppercase;">Personnel Count</strong>
                        <h2 style="margin: 10px 0 0 0; color: #212529; font-weight: 700; font-size: 28px;">{rec.recommended_personnel_count}</h2>
                        <span style="font-size: 11px; color: #6c757d;">Officers recommended for deployment</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_b:
                barr_status = "Recommended" if rec.barricade_recommended else "Not Indicated"
                barr_color = "#dc3545" if rec.barricade_recommended else "#6c757d"
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 5px solid {barr_color}; min-height: 125px;">
                        <strong style="color: #495057; font-size: 13px; text-transform: uppercase;">Barricading</strong>
                        <h4 style="margin: 10px 0 0 0; color: #212529; font-weight: 600; font-size: 18px;">{barr_status}</h4>
                        <span style="font-size: 11px; color: #6c757d; display: block; margin-top: 5px; line-height: 1.2;">
                            Hint: {rec.barricade_location_hint}
                        </span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_d:
                st.markdown(
                    f"""
                    <div style="background-color: #f8f9fa; padding: 16px; border-radius: 8px; border-left: 5px solid #28a745; min-height: 125px;">
                        <strong style="color: #495057; font-size: 13px; text-transform: uppercase;">Diversion Guidance</strong>
                        <p style="margin: 8px 0 0 0; font-size: 11px; color: #333; line-height: 1.3;">
                            {rec.diversion_suggestion}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            st.markdown("<br>", unsafe_allow_html=True)
            
            # 5. Full rationale text from resource_engine in expandable section
            with st.expander("Why this recommendation?", expanded=False):
                st.write(rec.rationale)
                
            # 5.5 Draft Public Traffic Advisory
            advisory_text = generate_traffic_advisory(
                event_cause=event_cause,
                planned_start_datetime=planned_start_datetime,
                corridor=corridor,
                requires_road_closure=requires_road_closure,
                impact=impact,
                rec=rec,
                cleaned_df=cleaned_df
            )
            
            st.markdown("### Draft Public Traffic Advisory")
            st.text_area(
                "Copy and edit this advisory for public posting:",
                value=advisory_text,
                height=280,
                key="advisory_draft_area"
            )
            
            # File download name
            file_date = planned_start_datetime.strftime("%Y%m%d")
            file_cause = event_cause.replace("_", "")
            download_filename = f"traffic_advisory_{file_date}_{file_cause}.txt"
            
            st.download_button(
                label="Export as .txt",
                data=advisory_text,
                file_name=download_filename,
                mime="text/plain"
            )
            
            st.markdown("<br>", unsafe_allow_html=True)
                
            # 6. Historical evidence events table
            # Filter cleaned_df by event_cause and corridor (if not None/Unknown)
            evidence_df = cleaned_df[cleaned_df["event_cause"] == event_cause]
            if corridor:
                evidence_df = evidence_df[evidence_df["corridor"] == corridor]
                
            st.markdown("### Historical Evidence")
            st.markdown(
                f"Based on **{impact.evidence_count}** matching past events in database. "
                f"Match Level Used: `{impact.match_level.upper()}`"
            )
            
            if len(evidence_df) > 0:
                cols_to_show = [
                    "start_datetime_ist",
                    "event_cause",
                    "corridor",
                    "event_type",
                    "priority",
                    "requires_road_closure",
                    "duration_minutes",
                    "police_station",
                    "junction",
                    "address"
                ]
                # Filter to available columns in dataset
                available_cols = [c for c in cols_to_show if c in evidence_df.columns]
                
                disp_df = evidence_df[available_cols].copy()
                if "start_datetime_ist" in disp_df.columns:
                    disp_df["start_datetime_ist"] = pd.to_datetime(disp_df["start_datetime_ist"])
                    disp_df = disp_df.sort_values(by="start_datetime_ist", ascending=False)
                    disp_df["start_datetime_ist"] = disp_df["start_datetime_ist"].dt.strftime("%Y-%m-%d %H:%M")
                    
                st.dataframe(disp_df.head(20), use_container_width=True, hide_index=True)
            else:
                st.info("No matching historical records found for this combination in the cleaned dataset.")
                
        except Exception as e:
            st.warning(f"⚠️ Insufficient data or anomalous input configuration. Could not generate prediction. (Details: {e})")

    # ─── CONTINUOUS LEARNING / FEEDBACK LOOP ──────────────────────────────────
    st.markdown("---")
    st.markdown("### Log Actual Outcome (Continuous Learning)")
    st.markdown("Record actual duration and closure outcomes for past events to improve the historical baselines on the next data refresh.")
    
    with st.form("feedback_form"):
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            unique_causes = sorted(cleaned_df["event_cause"].dropna().unique())
            f_cause = st.selectbox("Actual Event Cause", options=unique_causes, format_func=lambda x: x.replace("_", " ").title())
            
            raw_corridors = cleaned_df["corridor"].dropna().unique().tolist()
            if "Non-corridor" in raw_corridors:
                raw_corridors.remove("Non-corridor")
            unique_corridors = ["Unknown"] + sorted(raw_corridors)
            f_corridor_input = st.selectbox("Actual Corridor", options=unique_corridors)
            f_corridor = None if f_corridor_input == "Unknown" else f_corridor_input
            
        with f_col2:
            f_duration = st.number_input("Actual Duration (minutes)", min_value=0, value=60)
            f_closure = st.checkbox("Road Closure Actually Occurred")
            
        submit_feedback = st.form_submit_button("Log Outcome")
        
    if submit_feedback:
        log_actual_outcome(
            event_cause=f_cause,
            corridor=f_corridor,
            actual_duration_minutes=f_duration,
            requires_road_closure=f_closure
        )
        st.success(f"✅ Recorded. Historical baseline for `{f_cause}` on `{f_corridor_input}` will improve on next data refresh.")
