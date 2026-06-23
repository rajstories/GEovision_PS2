import streamlit as st
import pandas as pd
from datetime import datetime, time
from time import sleep
import sys
import os

# Add src folder to python path to resolve sibling imports cleanly
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from impact_model import predict_impact
from resource_engine import generate_recommendation
from feedback_loop import (
    build_combined_events_with_feedback,
    log_actual_outcome,
    refresh_lookup_with_feedback,
)
from advisory_generator import generate_traffic_advisory
from action_brief import generate_action_brief_html
from command_queue import rank_and_allocate

# Folium is optional: the dashboard still runs without it (the map section
# degrades to a friendly notice). Both libs are in requirements.txt.
try:
    import folium
    from folium.plugins import HeatMap
    from streamlit_folium import st_folium
    _FOLIUM_OK = True
except Exception:
    _FOLIUM_OK = False

# Severity-proxy thresholds mirror SEVERITY_CONFIG in src/impact_model.py
# (used only to colour historical points on the map; the live prediction still
# uses the real impact_model logic).
_SEV_LOW_MAX = 50.0
_SEV_MED_MAX = 120.0
_SEV_COLORS = {"Low": "#2E7D32", "Medium": "#EF6C00", "High": "#C62828", "Unknown": "#78909C"}


def _severity_proxy(minutes) -> str:
    """Map a historical duration (minutes) to a Low/Medium/High/Unknown tier."""
    if minutes is None or pd.isna(minutes):
        return "Unknown"
    if minutes < _SEV_LOW_MAX:
        return "Low"
    if minutes < _SEV_MED_MAX:
        return "Medium"
    return "High"


def _format_minutes(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{float(value):.0f} min"


def _format_percent(value) -> str:
    if value is None or pd.isna(value):
        return "Unknown"
    return f"{float(value) * 100:.1f}%"


def _lookup_snapshot(impact) -> dict:
    return {
        "event_count": int(impact.evidence_count),
        "median_duration": impact.expected_duration_minutes,
        "closure_rate": impact.road_closure_probability,
        "match_level": impact.match_level,
    }


def _render_verified_refresh_comparison(before: dict, after: dict) -> None:
    before_duration = _format_minutes(before["median_duration"])
    after_duration = _format_minutes(after["median_duration"])
    before_closure = _format_percent(before["closure_rate"])
    after_closure = _format_percent(after["closure_rate"])

    st.markdown("#### Verified Outcome Refresh")
    st.markdown(
        f"""
        <div style="display:grid;grid-template-columns:1fr 44px 1fr;gap:12px;align-items:stretch;margin:12px 0 18px 0;">
            <div style="border:1px solid #e9ecef;border-radius:8px;padding:14px 16px;background:#f8f9fa;">
                <div style="font-size:12px;font-weight:700;color:#6c757d;text-transform:uppercase;margin-bottom:8px;">Before feedback</div>
                <div style="font-size:26px;font-weight:700;color:#212529;">{before['event_count']} matching events</div>
                <div style="font-size:14px;color:#495057;margin-top:6px;">{before_duration} median duration</div>
                <div style="font-size:14px;color:#495057;">{before_closure} closure rate</div>
                <div style="font-size:12px;color:#868e96;margin-top:8px;">Evidence: {before['match_level']}</div>
            </div>
            <div style="display:flex;align-items:center;justify-content:center;font-size:28px;font-weight:800;color:#007bff;">&rarr;</div>
            <div style="border:2px solid #007bff;border-radius:8px;padding:14px 16px;background:#eef6ff;">
                <div style="font-size:12px;font-weight:700;color:#0056b3;text-transform:uppercase;margin-bottom:8px;">After feedback</div>
                <div style="font-size:26px;font-weight:700;color:#0b3558;">{after['event_count']} matching events</div>
                <div style="font-size:14px;color:#0b3558;margin-top:6px;">{after_duration} median duration</div>
                <div style="font-size:14px;color:#0b3558;">{after_closure} closure rate</div>
                <div style="font-size:12px;color:#35617f;margin-top:8px;">Evidence: {after['match_level']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _short_confidence(confidence_label: str) -> str:
    label = str(confidence_label)
    if label.startswith("High"):
        return "High"
    if label.startswith("Moderate"):
        return "Medium"
    if label.startswith("Low"):
        return "Low"
    return "Insufficient"


def _render_compact_command_summary(impact, rec) -> None:
    duration = _format_minutes(impact.expected_duration_minutes)
    closure = _format_percent(impact.road_closure_probability)
    confidence = _short_confidence(impact.confidence_label)
    barricade_text = (
        f"Barricade {rec.barricade_location_hint}"
        if rec.barricade_recommended
        else "Barricade not indicated"
    )
    st.markdown(
        f"""
        <div style="border:2px solid #212529;border-radius:8px;padding:16px 18px;margin:8px 0 22px 0;background:#ffffff;">
            <div style="font-size:22px;font-weight:800;color:#212529;line-height:1.3;">
                {impact.severity_tier.upper()} RISK | {duration} expected duration | {closure} closure risk
            </div>
            <div style="font-size:15px;color:#343a40;margin-top:8px;">
                Deploy <strong>{rec.recommended_personnel_count} officers</strong> | {barricade_text}
            </div>
            <div style="font-size:13px;color:#6c757d;margin-top:8px;">
                Confidence: {confidence} | Evidence: {impact.evidence_count} similar historical events | Match: {impact.match_level}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _historical_evidence_for_prediction(
    cleaned_df: pd.DataFrame,
    event_cause: str,
    corridor: str | None,
    match_level: str,
) -> pd.DataFrame:
    evidence = cleaned_df[cleaned_df["event_cause"] == event_cause].copy()
    if match_level == "fine":
        effective_corridor = corridor if corridor else "Non-corridor"
        evidence = evidence[evidence["corridor"] == effective_corridor]
    if "start_datetime_ist" in evidence.columns:
        evidence["_sort_dt"] = pd.to_datetime(evidence["start_datetime_ist"], errors="coerce")
        evidence = evidence.sort_values("_sort_dt", ascending=False).drop(columns=["_sort_dt"])
    return evidence


def _incident_id(event_cause: str, planned_start_datetime: datetime) -> str:
    cause_code = "".join(ch for ch in event_cause.upper() if ch.isalnum())[:6] or "EVENT"
    return f"BTP-{planned_start_datetime:%Y%m%d%H%M}-{cause_code}"


def _build_command_queue_events(
    presets: list[dict],
    lookup_df: pd.DataFrame,
    cleaned_df: pd.DataFrame,
) -> list[dict]:
    queue_events = []
    for preset in presets:
        corridor = None if preset["corridor_input"] == "Unknown" else preset["corridor_input"]
        planned_start = datetime.combine(preset["date"], preset["time"])
        impact = predict_impact(
            event_cause=preset["event_cause"],
            planned_start_datetime=planned_start,
            corridor=corridor,
            requires_road_closure=preset["requires_road_closure"],
            lookup_df=lookup_df,
            cleaned_df=cleaned_df,
        )
        rec = generate_recommendation(
            impact=impact,
            event_cause=preset["event_cause"],
            corridor=corridor,
            requires_road_closure=preset["requires_road_closure"],
        )
        queue_events.append(
            {
                "event_id": _incident_id(preset["event_cause"], planned_start),
                "description": preset["label"],
                "severity_tier": impact.severity_tier,
                "road_closure_probability": impact.road_closure_probability,
                "evidence_count": impact.evidence_count,
                "confidence_label": impact.confidence_label,
                "match_level": impact.match_level,
                "officers_requested": rec.recommended_personnel_count,
                "barricades_requested": 1 if rec.barricade_recommended else 0,
            }
        )
    return queue_events


DEMO_PRESETS = [
    {
        "label": "Planned Public Event",
        "event_cause": "procession",
        "corridor_input": "Unknown",
        "date": datetime(2024, 4, 1).date(),
        "time": time(1, 0),
        "requires_road_closure": True,
    },
    {
        "label": "Unplanned Breakdown",
        "event_cause": "vehicle_breakdown",
        "corridor_input": "ORR East 1",
        "date": datetime(2024, 4, 1).date(),
        "time": time(9, 0),
        "requires_road_closure": False,
    },
    {
        "label": "Rare Event — Limited Evidence",
        "event_cause": "vip_movement",
        "corridor_input": "CBD 2",
        "date": datetime(2024, 4, 6).date(),
        "time": time(23, 0),
        "requires_road_closure": True,
    },
]


def build_incident_map(points_df: pd.DataFrame, marker_cap: int = 300):
    """
    Build a Folium map of historical incidents backing a prediction.

    Markers are coloured by a duration-based severity proxy; a heatmap layer
    shows incident density. Uses only the dataset's own latitude/longitude —
    no external geographic data is fetched.

    Returns (folium.Map | None, n_plotted).
    """
    lat = pd.to_numeric(points_df["latitude"], errors="coerce")
    lon = pd.to_numeric(points_df["longitude"], errors="coerce")
    valid = lat.between(12.7, 13.3) & lon.between(77.3, 77.9)
    pts = points_df.loc[valid].copy()
    pts["_lat"], pts["_lon"] = lat[valid], lon[valid]
    if len(pts) == 0:
        return None, 0

    center = [pts["_lat"].mean(), pts["_lon"].mean()]
    fmap = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Density heatmap over ALL valid points
    HeatMap(
        pts[["_lat", "_lon"]].values.tolist(),
        radius=16, blur=12, min_opacity=0.25, name="Incident density",
    ).add_to(fmap)

    # Individual markers (capped to the most recent N for responsiveness)
    if "start_datetime_ist" in pts.columns:
        pts = pts.sort_values("start_datetime_ist", ascending=False)
    markers = folium.FeatureGroup(name="Incidents (most recent)")
    for _, r in pts.head(marker_cap).iterrows():
        sev = _severity_proxy(r.get("duration_minutes"))
        color = _SEV_COLORS[sev]
        dur = r.get("duration_minutes")
        dur_str = f"{float(dur):.0f} min" if pd.notna(dur) else "unknown"
        popup_html = (
            f"<b>{str(r.get('event_cause','')).replace('_',' ').title()}</b><br>"
            f"Corridor: {r.get('corridor','—')}<br>"
            f"When: {r.get('start_datetime_ist','—')}<br>"
            f"Duration: {dur_str} (proxy: {sev})<br>"
            f"Road closure: {'Yes' if bool(r.get('requires_road_closure')) else 'No'}<br>"
            f"Station: {r.get('police_station','—')}"
        )
        folium.CircleMarker(
            location=[r["_lat"], r["_lon"]], radius=5, color=color, weight=1,
            fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(popup_html, max_width=260),
        ).add_to(markers)
    markers.add_to(fmap)
    folium.LayerControl(collapsed=True).add_to(fmap)
    return fmap, len(pts)

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
    base_cleaned_df, base_lookup_df = load_data()
    if "active_cleaned_df" not in st.session_state:
        st.session_state["active_cleaned_df"] = base_cleaned_df
    if "active_lookup_df" not in st.session_state:
        st.session_state["active_lookup_df"] = base_lookup_df
    cleaned_df = st.session_state["active_cleaned_df"]
    lookup_df = st.session_state["active_lookup_df"]
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

    unique_causes = sorted(cleaned_df["event_cause"].dropna().unique())
    raw_corridors = cleaned_df["corridor"].dropna().unique().tolist()
    if "Non-corridor" in raw_corridors:
        raw_corridors.remove("Non-corridor")
    unique_corridors = ["Unknown"] + sorted(raw_corridors)

    st.session_state.setdefault("input_event_cause", unique_causes[0])
    st.session_state.setdefault("input_corridor", "Unknown")
    st.session_state.setdefault("input_date", datetime.today().date())
    st.session_state.setdefault("input_time", datetime.now().time())
    st.session_state.setdefault("input_closure", False)

    preset_trigger_label = None
    preset_cols = st.columns(3)
    for col, preset in zip(preset_cols, DEMO_PRESETS):
        with col:
            if st.button(preset["label"], width="stretch"):
                st.session_state["input_event_cause"] = preset["event_cause"]
                st.session_state["input_corridor"] = preset["corridor_input"]
                st.session_state["input_date"] = preset["date"]
                st.session_state["input_time"] = preset["time"]
                st.session_state["input_closure"] = preset["requires_road_closure"]
                preset_trigger_label = preset["label"]

    with st.form("event_details_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Cause Dropdown
            event_cause = st.selectbox(
                "Event Cause / Type",
                options=unique_causes,
                format_func=lambda x: x.replace("_", " ").title(),
                key="input_event_cause",
            )
            
            # Corridor Dropdown (optional "Unknown")
            corridor_input = st.selectbox(
                "Corridor (Optional)",
                options=unique_corridors,
                key="input_corridor",
            )
            # Map "Unknown" back to None so it gets mapped to Non-corridor downstream
            corridor = None if corridor_input == "Unknown" else corridor_input
            
        with col2:
            # Date & Time picker
            col_date, col_time = st.columns(2)
            with col_date:
                selected_date = st.date_input("Event Date", key="input_date")
            with col_time:
                selected_time = st.time_input("Event Start Time", key="input_time")
            
            planned_start_datetime = datetime.combine(selected_date, selected_time)
            
            # Road closure expected checkbox
            requires_road_closure = st.checkbox(
                "Road Closure Expected / Required",
                key="input_closure",
            )
            
        submit_button = st.form_submit_button("Generate Prediction & Plan")

    # Output Rendering
    run_prediction = submit_button or preset_trigger_label is not None
    if run_prediction:
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
            evidence_df = _historical_evidence_for_prediction(
                cleaned_df=cleaned_df,
                event_cause=event_cause,
                corridor=corridor,
                match_level=impact.match_level,
            )
            
            st.markdown("---")
            st.markdown("### Prediction & Recommendation Output")
            if preset_trigger_label:
                st.caption(f"Preset scenario: {preset_trigger_label}")
            _render_compact_command_summary(impact, rec)
            
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

            brief_incident_id = _incident_id(event_cause, planned_start_datetime)
            action_brief_html = generate_action_brief_html(
                incident_id=brief_incident_id,
                generated_at=datetime.now(),
                event_cause=event_cause,
                planned_start_datetime=planned_start_datetime,
                corridor=corridor,
                requires_road_closure=requires_road_closure,
                impact=impact,
                rec=rec,
                advisory_text=advisory_text,
                evidence_df=evidence_df,
            )
            export_col1, export_col2 = st.columns(2)
            with export_col1:
                st.download_button(
                    label="Export Advisory .txt",
                    data=advisory_text,
                    file_name=download_filename,
                    mime="text/plain"
                )
            with export_col2:
                st.download_button(
                    label="Download Action Brief",
                    data=action_brief_html,
                    file_name=f"action_brief_{brief_incident_id}.html",
                    mime="text/html",
                )
            
            st.markdown("<br>", unsafe_allow_html=True)
                
            # 6. Historical evidence events table
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
                    
                st.dataframe(disp_df.head(20), width="stretch", hide_index=True)

                # ── Geospatial view of the backing evidence ───────────────────
                st.markdown("#### 🗺️ Incident Map")
                st.markdown(
                    "Where these matching incidents occurred — markers coloured by a "
                    "duration-based severity proxy, plus a density heatmap. This shows "
                    "*visually* how much historical evidence backs the prediction."
                )
                if _FOLIUM_OK:
                    fmap, n_plotted = build_incident_map(evidence_df)
                    if fmap is not None:
                        st.caption(
                            f"Plotting {n_plotted} geolocated historical incidents "
                            f"(markers capped to the 300 most recent; heatmap uses all)."
                        )
                        st_folium(fmap, use_container_width=True, height=460,
                                  returned_objects=[], key="evidence_map")
                    else:
                        st.info("No geolocated records available to map for this combination.")
                else:
                    st.info(
                        "Map requires `folium` and `streamlit-folium` "
                        "(`pip install folium streamlit-folium`)."
                    )
            else:
                st.info("No matching historical records found for this combination in the cleaned dataset.")

        except Exception as e:
            st.warning(f"⚠️ Insufficient data or anomalous input configuration. Could not generate prediction. (Details: {e})")

    # ─── BUDGET-AWARE COMMAND QUEUE ──────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Budget-Aware Command Queue")
    st.caption(
        "Deterministic rule-based allocator for the current demo preset incidents; "
        "every ranking decision is traceable to named weights."
    )

    q_col1, q_col2 = st.columns(2)
    with q_col1:
        total_officers = st.number_input(
            "Total Available Officers",
            min_value=0,
            value=12,
            step=1,
            key="queue_total_officers",
        )
    with q_col2:
        total_barricades = st.number_input(
            "Total Available Barricades",
            min_value=0,
            value=2,
            step=1,
            key="queue_total_barricades",
        )

    if st.button("Run Budget-Aware Command Queue", width="stretch"):
        try:
            queue_inputs = _build_command_queue_events(DEMO_PRESETS, lookup_df, cleaned_df)
            queue_result = rank_and_allocate(
                queue_inputs,
                total_officers=int(total_officers),
                total_barricades=int(total_barricades),
            )
            summary = queue_result.summary
            sm1, sm2, sm3, sm4 = st.columns(4)
            sm1.metric(
                "Officers",
                f"{summary.officers_allocated}/{summary.officers_requested}",
                f"{summary.officers_remaining} remaining",
            )
            sm2.metric(
                "Barricades",
                f"{summary.barricades_allocated}/{summary.barricades_requested}",
                f"{summary.barricades_remaining} remaining",
            )
            sm3.metric("Fully Covered", summary.events_covered)
            sm4.metric("At Risk", summary.events_at_risk)

            queue_rows = [
                {
                    "Rank": event.rank,
                    "Incident": event.description,
                    "Priority Score": event.priority_score,
                    "Officers Requested": event.officers_requested,
                    "Officers Allocated": event.officers_allocated,
                    "Barricades Requested": event.barricades_requested,
                    "Barricades Allocated": event.barricades_allocated,
                    "Covered": event.covered,
                    "Rationale": event.rationale,
                    "Risk Flag": event.risk_flag,
                }
                for event in queue_result.events
            ]
            st.table(pd.DataFrame(queue_rows))
        except Exception as e:
            st.warning(f"Budget-Aware Command Queue could not run. Details: {e}")

    # ─── VERIFIED OUTCOME REFRESH ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Verified Outcome Refresh")
    st.markdown(
        "Record officer-verified outcomes and rebuild the historical lookup in "
        "this session. The original cleaned ASTRAM dataset is not modified."
    )
    
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

            priority_values = sorted(cleaned_df["priority"].dropna().astype(str).unique().tolist())
            priority_options = [p for p in ["High", "Low", "Unknown"] if p in set(priority_values) or p == "Unknown"]
            for p in priority_values:
                if p not in priority_options:
                    priority_options.append(p)
            f_priority = st.selectbox("Priority", options=priority_options)
            
        with f_col2:
            f_date_col, f_time_col = st.columns(2)
            with f_date_col:
                f_date = st.date_input("Event Date/Time", value=datetime.today(), key="feedback_event_date")
            with f_time_col:
                f_time = st.time_input(" ", value=datetime.now().time(), key="feedback_event_time", label_visibility="collapsed")

            f_duration = st.number_input("Actual Duration (minutes)", min_value=0, value=60)
            f_closure = st.checkbox("Actual Road Closure Occurred")
            f_verified_by = st.text_input("Verified By (Officer/Operator ID)", value="demo-officer")
            f_status = st.selectbox(
                "Verification Status",
                options=["verified", "unverified", "disputed"],
                index=0,
            )
            
        submit_feedback = st.form_submit_button("Submit Verified Outcome Refresh")
        
    if submit_feedback:
        feedback_event_datetime = datetime.combine(f_date, f_time)
        before_impact = predict_impact(
            event_cause=f_cause,
            planned_start_datetime=feedback_event_datetime,
            corridor=f_corridor,
            requires_road_closure=f_closure,
            lookup_df=lookup_df,
            cleaned_df=cleaned_df,
        )
        before_snapshot = _lookup_snapshot(before_impact)

        appended = log_actual_outcome(
            event_cause=f_cause,
            corridor=f_corridor,
            actual_duration_minutes=f_duration,
            actual_requires_road_closure=f_closure,
            event_datetime=feedback_event_datetime,
            priority=f_priority,
            verified_by=f_verified_by,
            verification_status=f_status,
        )

        if f_status == "verified":
            refreshed_lookup = refresh_lookup_with_feedback()
            refreshed_cleaned = build_combined_events_with_feedback()
            st.session_state["active_lookup_df"] = refreshed_lookup
            st.session_state["active_cleaned_df"] = refreshed_cleaned
            lookup_df = refreshed_lookup
            cleaned_df = refreshed_cleaned

            after_impact = predict_impact(
                event_cause=f_cause,
                planned_start_datetime=feedback_event_datetime,
                corridor=f_corridor,
                requires_road_closure=f_closure,
                lookup_df=lookup_df,
                cleaned_df=cleaned_df,
            )
            after_snapshot = _lookup_snapshot(after_impact)

            if appended:
                st.success("Verified outcome recorded. Lookup refreshed for the next prediction in this session.")
            else:
                st.warning("Duplicate verified outcome detected; it was not counted again.")
            _render_verified_refresh_comparison(before_snapshot, after_snapshot)
        else:
            if appended:
                st.info(
                    f"Outcome logged as `{f_status}`. It is excluded from Verified Outcome Refresh until marked verified."
                )
            else:
                st.warning("Duplicate outcome detected; it was not logged again.")

    # ─── LIVE INCIDENT FEED (HISTORICAL REPLAY) ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🔴 Live Incident Feed (Replay)")
    st.markdown(
        "Replays real historical ASTRAM events in timestamp order at an accelerated "
        "rate, feeding each one through the **same** `predict_impact → resource_engine` "
        "pipeline live — a stand-in for a streaming incident feed. This demonstrates the "
        "*real-time* half of the problem statement using only the provided dataset."
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        replay_speed = st.slider("Simulated hours per real second", 1, 24, 6,
                                 help="Higher = faster replay. The simulated clock advances "
                                      "by each event's real time gap, divided by this factor.")
    with rc2:
        replay_count = st.slider("Number of recent events to replay", 5, 40, 15)

    if st.button("▶ Start Replay"):
        try:
            # Most-recent `replay_count` events, replayed oldest → newest
            feed_src = cleaned_df.copy()
            feed_src["_ist"] = pd.to_datetime(feed_src["start_datetime_ist"], errors="coerce")
            feed_src = (feed_src.dropna(subset=["_ist"])
                        .sort_values("_ist").tail(replay_count).reset_index(drop=True))

            clock_ph = st.empty()
            feed_ph = st.empty()
            progress = st.progress(0)
            feed_cards: list[str] = []
            prev_t = None

            for i, row in feed_src.iterrows():
                t = row["_ist"]
                corr = None if row["corridor"] == "Non-corridor" else row["corridor"]
                impact = predict_impact(
                    event_cause=row["event_cause"],
                    planned_start_datetime=t.to_pydatetime(),
                    corridor=corr,
                    requires_road_closure=bool(row["requires_road_closure"]),
                    lookup_df=lookup_df,
                    cleaned_df=cleaned_df,
                )
                rec = generate_recommendation(
                    impact=impact, event_cause=row["event_cause"], corridor=corr,
                    requires_road_closure=bool(row["requires_road_closure"]),
                )
                sev = impact.severity_tier
                color = _SEV_COLORS.get(sev, "#78909C")
                card = (
                    f"<div style='border-left:5px solid {color};background:#f8f9fa;"
                    f"padding:8px 14px;margin-bottom:6px;border-radius:6px;'>"
                    f"<span style='color:#6c757d;font-size:12px;'>{t:%Y-%m-%d %H:%M}</span>"
                    f"&nbsp;·&nbsp;<b>{str(row['event_cause']).replace('_',' ').title()}</b>"
                    f"&nbsp;on&nbsp;{row['corridor']}<br>"
                    f"<span style='color:{color};font-weight:600;'>{sev} severity</span>"
                    f"&nbsp;·&nbsp;{rec.recommended_personnel_count} officers"
                    f"&nbsp;·&nbsp;barricade: {'Yes' if rec.barricade_recommended else 'No'}"
                    f"&nbsp;·&nbsp;<span style='color:#868e96;font-size:12px;'>"
                    f"match: {impact.match_level}</span></div>"
                )
                feed_cards.insert(0, card)
                feed_cards = feed_cards[:8]  # rolling window

                clock_ph.markdown(
                    f"**🕐 Simulated time:** {t:%Y-%m-%d %H:%M} &nbsp;|&nbsp; "
                    f"event {i+1} of {len(feed_src)}"
                )
                feed_ph.markdown("".join(feed_cards), unsafe_allow_html=True)
                progress.progress(int((i + 1) / len(feed_src) * 100))

                # Advance the simulated clock; clamp real delay so it's watchable
                if prev_t is not None:
                    gap_h = max((t - prev_t).total_seconds() / 3600.0, 0.0)
                    sleep(min(max(gap_h / replay_speed, 0.05), 1.5))
                prev_t = t

            st.success("Replay complete — each event was scored live through the production pipeline.")
        except Exception as e:
            st.warning(f"⚠️ Replay could not complete. (Details: {e})")
