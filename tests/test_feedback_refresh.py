import os
import sys
import tempfile
import unittest

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from feedback_loop import (
    log_actual_outcome,
    read_feedback_log,
    refresh_lookup_with_feedback,
)


def _base_cleaned_events() -> pd.DataFrame:
    rows = []
    for idx, duration in enumerate([60, 70, 70, 70, 80], start=1):
        rows.append(
            {
                "id": idx,
                "event_type": "planned",
                "event_cause": "public_event",
                "corridor": "MG Road",
                "priority": "High",
                "requires_road_closure": False,
                "start_datetime": f"2024-04-0{idx} 08:00:00",
                "start_datetime_ist": f"2024-04-0{idx} 08:00:00",
                "duration_minutes": duration,
                "has_duration": True,
                "hour_of_day": 8,
                "day_of_week": 0,
                "is_weekend": False,
            }
        )
    return pd.DataFrame(rows)


def _fine_row(lookup: pd.DataFrame) -> pd.Series:
    hits = lookup[
        (lookup["match_level"] == "fine")
        & (lookup["event_cause"] == "public_event")
        & (lookup["corridor"] == "MG Road")
        & (lookup["is_weekend"] == False)
        & (lookup["hour_bucket"] == "morning")
    ]
    if hits.empty:
        raise AssertionError("Expected fine lookup row was not produced")
    return hits.iloc[0]


class FeedbackRefreshTests(unittest.TestCase):
    def _paths(self, tmpdir: str) -> tuple[str, str]:
        cleaned_path = os.path.join(tmpdir, "cleaned_events.csv")
        feedback_path = os.path.join(tmpdir, "feedback_log.csv")
        _base_cleaned_events().to_csv(cleaned_path, index=False)
        return cleaned_path, feedback_path

    def test_verified_feedback_updates_lookup_immediately(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path, feedback_path = self._paths(tmpdir)
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-05-01 10:00:00",
                        "event_datetime": "2024-05-01 08:30:00",
                        "event_cause": "public_event",
                        "corridor": "MG Road",
                        "actual_duration_minutes": 120,
                        "actual_requires_road_closure": True,
                        "priority": "High",
                        "verified_by": "officer-1",
                        "verification_status": "verified",
                    }
                ]
            ).to_csv(feedback_path, index=False)

            lookup = refresh_lookup_with_feedback(cleaned_path, feedback_path)
            row = _fine_row(lookup)

            self.assertEqual(int(row["event_count"]), 6)
            self.assertEqual(int(row["duration_count"]), 6)
            self.assertAlmostEqual(float(row["road_closure_rate"]), 1 / 6, places=4)

    def test_duplicate_verified_feedback_does_not_double_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path, feedback_path = self._paths(tmpdir)
            duplicate = {
                "timestamp": "2024-05-01 10:00:00",
                "event_datetime": "2024-05-01 08:30:00",
                "event_cause": "public_event",
                "corridor": "MG Road",
                "actual_duration_minutes": 120,
                "actual_requires_road_closure": True,
                "priority": "High",
                "verified_by": "officer-1",
                "verification_status": "verified",
            }
            pd.DataFrame([duplicate, {**duplicate, "timestamp": "2024-05-01 10:01:00"}]).to_csv(
                feedback_path, index=False
            )

            lookup = refresh_lookup_with_feedback(cleaned_path, feedback_path)
            row = _fine_row(lookup)

            self.assertEqual(int(row["event_count"]), 6)
            self.assertEqual(int(row["duration_count"]), 6)

    def test_invalid_duration_feedback_is_excluded_from_duration_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path, feedback_path = self._paths(tmpdir)
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-05-01 10:00:00",
                        "event_datetime": "2024-05-01 08:30:00",
                        "event_cause": "public_event",
                        "corridor": "MG Road",
                        "actual_duration_minutes": "not-a-number",
                        "actual_requires_road_closure": True,
                        "priority": "High",
                        "verified_by": "officer-1",
                        "verification_status": "verified",
                    }
                ]
            ).to_csv(feedback_path, index=False)

            lookup = refresh_lookup_with_feedback(cleaned_path, feedback_path)
            row = _fine_row(lookup)

            self.assertEqual(int(row["event_count"]), 6)
            self.assertEqual(int(row["duration_count"]), 5)
            self.assertEqual(float(row["median_duration_minutes"]), 70)

    def test_unverified_feedback_is_excluded_entirely(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cleaned_path, feedback_path = self._paths(tmpdir)
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-05-01 10:00:00",
                        "event_datetime": "2024-05-01 08:30:00",
                        "event_cause": "public_event",
                        "corridor": "MG Road",
                        "actual_duration_minutes": 120,
                        "actual_requires_road_closure": True,
                        "priority": "High",
                        "verified_by": "officer-1",
                        "verification_status": "unverified",
                    }
                ]
            ).to_csv(feedback_path, index=False)

            lookup = refresh_lookup_with_feedback(cleaned_path, feedback_path)
            row = _fine_row(lookup)

            self.assertEqual(int(row["event_count"]), 5)
            self.assertEqual(int(row["duration_count"]), 5)
            self.assertEqual(float(row["road_closure_rate"]), 0.0)

    def test_old_feedback_schema_is_readable_and_unverified_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_path = os.path.join(tmpdir, "feedback_log.csv")
            pd.DataFrame(
                [
                    {
                        "timestamp": "2024-05-01 10:00:00",
                        "event_cause": "public_event",
                        "corridor": "Unknown",
                        "actual_duration_minutes": 45,
                        "requires_road_closure": True,
                    }
                ]
            ).to_csv(feedback_path, index=False)

            feedback = read_feedback_log(feedback_path)

            self.assertEqual(feedback.loc[0, "corridor"], "Non-corridor")
            self.assertTrue(bool(feedback.loc[0, "actual_requires_road_closure"]))
            self.assertEqual(feedback.loc[0, "verification_status"], "unverified")

    def test_log_actual_outcome_skips_exact_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            feedback_path = os.path.join(tmpdir, "feedback_log.csv")
            first = log_actual_outcome(
                event_cause="public_event",
                corridor="MG Road",
                actual_duration_minutes=90,
                requires_road_closure=True,
                event_datetime=pd.Timestamp("2024-05-01 08:30:00").to_pydatetime(),
                priority="High",
                verified_by="officer-1",
                verification_status="verified",
                feedback_log_path=feedback_path,
            )
            second = log_actual_outcome(
                event_cause="public_event",
                corridor="MG Road",
                actual_duration_minutes=90,
                requires_road_closure=True,
                event_datetime=pd.Timestamp("2024-05-01 08:30:00").to_pydatetime(),
                priority="High",
                verified_by="officer-1",
                verification_status="verified",
                feedback_log_path=feedback_path,
            )

            self.assertTrue(first)
            self.assertFalse(second)
            self.assertEqual(len(pd.read_csv(feedback_path)), 1)


if __name__ == "__main__":
    unittest.main()
