import os
import sys
import unittest
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from action_brief import extract_vms_message, generate_action_brief_html


@dataclass
class _Impact:
    expected_duration_minutes: float = 82.0
    road_closure_probability: float = 0.36
    severity_tier: str = "High"
    evidence_count: int = 14
    duration_count: int = 12
    match_level: str = "fine"
    confidence_label: str = "High confidence"


@dataclass
class _Recommendation:
    recommended_personnel_count: int = 8
    rationale: str = "High severity with closure risk requires active junction management."
    barricade_recommended: bool = True
    barricade_location_hint: str = "MG Road junction"
    diversion_suggestion: str = (
        "A precise diversion route requires a live road-network graph with "
        "current traffic volumes, which is not available in this system."
    )


class ActionBriefTests(unittest.TestCase):
    def test_extract_vms_message_reuses_advisory_text(self):
        advisory = "Public note\n\nVMS SIGNBOARD MESSAGE:\n[ DIVERSION: MG ROAD CLOSED ]"
        self.assertEqual(extract_vms_message(advisory), "DIVERSION: MG ROAD CLOSED")

    def test_action_brief_contains_required_sections(self):
        advisory = "Traffic Advisory\n\nVMS SIGNBOARD MESSAGE:\n[ DIVERSION: MG ROAD CLOSED ]"
        evidence_df = pd.DataFrame(
            [
                {
                    "start_datetime_ist": "2024-04-01 08:00",
                    "event_cause": "procession",
                    "corridor": "MG Road",
                    "priority": "High",
                    "requires_road_closure": True,
                    "duration_minutes": 82,
                    "police_station": "Ulsoor",
                    "junction": "MG Road",
                }
            ]
        )

        html = generate_action_brief_html(
            incident_id="BTP-202404010800-PRO",
            generated_at=datetime(2024, 4, 1, 9, 0),
            event_cause="procession",
            planned_start_datetime=datetime(2024, 4, 1, 8, 0),
            corridor="MG Road",
            requires_road_closure=True,
            impact=_Impact(),
            rec=_Recommendation(),
            advisory_text=advisory,
            evidence_df=evidence_df,
            latitude=12.9716,
            longitude=77.5946,
        )

        self.assertIn("Incident ID", html)
        self.assertIn("Predicted Impact", html)
        self.assertIn("Supporting Historical Evidence", html)
        self.assertIn("8 officers", html)
        self.assertIn("MG Road junction", html)
        self.assertIn("live road-network graph", html)
        self.assertIn("DIVERSION: MG ROAD CLOSED", html)
        self.assertIn("Officer Sign-Off", html)


if __name__ == "__main__":
    unittest.main()
