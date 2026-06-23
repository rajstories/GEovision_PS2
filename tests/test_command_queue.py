import os
import sys
import unittest

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from command_queue import rank_and_allocate


def _events():
    return [
        {
            "event_id": "medium-well-evidenced",
            "description": "Medium incident with strong evidence",
            "severity_tier": "Medium",
            "road_closure_probability": 0.90,
            "evidence_count": 40,
            "confidence_label": "High confidence",
            "match_level": "fine",
            "officers_requested": 4,
            "barricades_requested": 1,
        },
        {
            "event_id": "high-rare",
            "description": "High incident with limited evidence",
            "severity_tier": "High",
            "road_closure_probability": 0.10,
            "evidence_count": 2,
            "confidence_label": "Low confidence — cause-only match",
            "match_level": "cause_only",
            "officers_requested": 6,
            "barricades_requested": 1,
        },
        {
            "event_id": "low-routine",
            "description": "Low routine event",
            "severity_tier": "Low",
            "road_closure_probability": 0.05,
            "evidence_count": 22,
            "confidence_label": "High confidence",
            "match_level": "fine",
            "officers_requested": 2,
            "barricades_requested": 0,
        },
    ]


class CommandQueueTests(unittest.TestCase):
    def test_budget_is_never_exceeded_across_multiple_scenarios(self):
        for officers, barricades in [(20, 4), (8, 1), (3, 0), (0, 2)]:
            with self.subTest(officers=officers, barricades=barricades):
                result = rank_and_allocate(_events(), officers, barricades)
                self.assertLessEqual(
                    sum(event.officers_allocated for event in result.events), officers
                )
                self.assertLessEqual(
                    sum(event.barricades_allocated for event in result.events), barricades
                )

    def test_allocation_order_respects_priority_score(self):
        result = rank_and_allocate(_events(), total_officers=20, total_barricades=4)
        scores = [event.priority_score for event in result.events]

        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(result.events[0].event_id, "high-rare")
        self.assertEqual(result.events[1].event_id, "medium-well-evidenced")

    def test_partial_allocation_when_budget_runs_out_mid_list(self):
        result = rank_and_allocate(_events(), total_officers=7, total_barricades=1)
        first, second, third = result.events

        self.assertEqual(first.event_id, "high-rare")
        self.assertEqual(first.officers_allocated, 6)
        self.assertEqual(first.barricades_allocated, 1)
        self.assertTrue(first.covered)
        self.assertEqual(second.officers_allocated, 1)
        self.assertEqual(second.barricades_allocated, 0)
        self.assertFalse(second.covered)
        self.assertIn("UNCOVERED RISK", second.risk_flag)
        self.assertEqual(third.officers_allocated, 0)

    def test_zero_evidence_event_is_ranked_without_crashing(self):
        events = _events() + [
            {
                "event_id": "new-cause",
                "description": "Unknown new cause",
                "severity_tier": "Unknown",
                "road_closure_probability": None,
                "evidence_count": 0,
                "confidence_label": "Insufficient data — defer to officer judgment",
                "match_level": "no_data",
                "officers_requested": 3,
                "barricades_requested": 1,
            }
        ]

        result = rank_and_allocate(events, total_officers=20, total_barricades=4)

        self.assertEqual(len(result.events), 4)
        self.assertIn("new-cause", {event.event_id for event in result.events})

    def test_summary_totals_reconcile(self):
        result = rank_and_allocate(_events(), total_officers=7, total_barricades=1)
        summary = result.summary

        self.assertEqual(summary.officers_allocated + summary.officers_remaining, 7)
        self.assertEqual(summary.barricades_allocated + summary.barricades_remaining, 1)
        self.assertEqual(
            summary.officers_allocated,
            sum(event.officers_allocated for event in result.events),
        )
        self.assertEqual(
            summary.barricades_allocated,
            sum(event.barricades_allocated for event in result.events),
        )
        self.assertEqual(
            summary.events_covered + summary.events_at_risk,
            len(result.events),
        )


if __name__ == "__main__":
    unittest.main()
