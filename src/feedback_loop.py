#!/usr/bin/env python3
"""
Feedback Loop Module

Demonstrates the concept of continuous learning.
In a production system, this module captures actual event outcomes (like duration and
whether a road was actually closed) and logs them. These logs are then periodically
merged into the main dataset before feature engineering is re-run, ensuring the
historical lookup baselines continuously improve and adapt to changing ground realities.
"""

import pandas as pd
import os
from datetime import datetime

FEEDBACK_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "feedback_log.csv")

def log_actual_outcome(event_cause: str, corridor: str, actual_duration_minutes: float, requires_road_closure: bool) -> None:
    """
    Logs an actual event outcome by appending it to a feedback log.
    
    Args:
        event_cause: The type of event.
        corridor: The road corridor where it happened.
        actual_duration_minutes: The real, observed duration.
        requires_road_closure: Whether a road closure actually occurred.
    """
    new_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "event_cause": event_cause,
        "corridor": corridor if corridor else "Unknown",
        "actual_duration_minutes": actual_duration_minutes,
        "requires_road_closure": requires_road_closure
    }
    
    df_new = pd.DataFrame([new_record])
    
    # Append to CSV, writing header only if file doesn't exist
    file_exists = os.path.exists(FEEDBACK_LOG_PATH)
    df_new.to_csv(FEEDBACK_LOG_PATH, mode='a', header=not file_exists, index=False)
