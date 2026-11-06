"""Unit tests for the pure dispatching heuristics."""

from types import SimpleNamespace
import unittest

import numpy as np

from src.scheduler.sequencing_rule import SequencingMethod


def make_job(due: float, remaining_pt, now: float = 0.0):
    """Create the minimal Job-like object required by the rule functions."""
    return SimpleNamespace(
        env=SimpleNamespace(now=now),
        due=float(due),
        remaining_pt=[float(x) for x in remaining_pt],
        remaining_machines=list(range(len(remaining_pt))),
    )


class TestSequencingRules(unittest.TestCase):
    def test_fifo_returns_front_of_queue(self):
        jobs = [make_job(10, [1]), make_job(10, [1])]
        self.assertEqual(SequencingMethod.FIFO(jobs), 0)

    def test_lifo_returns_last_concrete_index(self):
        jobs = [make_job(10, [1]) for _ in range(4)]
        # Must be a non-negative index (not Python's -1 alias) so after_operation
        # pops the job that was actually selected.
        self.assertEqual(SequencingMethod.LIFO(jobs), 3)
        self.assertGreaterEqual(SequencingMethod.LIFO(jobs), 0)

    def test_spt_selects_shortest_current_operation(self):
        jobs = [make_job(10, [5]), make_job(10, [2]), make_job(10, [9])]
        self.assertEqual(SequencingMethod.SPT(jobs), 1)

    def test_lwkr_selects_least_total_remaining_work(self):
        jobs = [make_job(10, [5]), make_job(10, [2, 1]), make_job(10, [9])]
        self.assertEqual(SequencingMethod.LWKR(jobs), 1)

    def test_cr_selects_smallest_time_to_due_ratio(self):
        # Ratios: 10/10 = 1.0; 20/10 = 2.0; 8/10 = 0.8.
        jobs = [make_job(10, [10]), make_job(20, [10]), make_job(8, [10])]
        self.assertEqual(SequencingMethod.CR(jobs), 2)

    def test_slack_uses_remaining_work_not_just_due_date(self):
        # EDD would pick job 0 (due 10); true slack picks job 1 because its
        # remaining work is much larger (slack 9 vs -9).
        jobs = [make_job(10, [1]), make_job(11, [20])]
        self.assertEqual(SequencingMethod.Slack(jobs), 1)
        self.assertEqual(int(np.argmin([j.due for j in jobs])), 0)

    def test_slack_reduces_to_edd_when_remaining_work_is_equal(self):
        jobs = [make_job(30, [2, 3]), make_job(10, [1, 4]), make_job(20, [5])]
        self.assertEqual(SequencingMethod.Slack(jobs), 1)

    def test_all_builtin_rules_return_in_range_index(self):
        jobs = [make_job(10, [3, 2]), make_job(20, [1, 1]), make_job(15, [4])]
        for rule_name in ("FIFO", "LIFO", "Slack", "CR", "SPT", "LWKR"):
            rule = getattr(SequencingMethod, rule_name)
            pos = rule(jobs)
            self.assertIsInstance(pos, int, rule_name)
            self.assertGreaterEqual(pos, 0, rule_name)
            self.assertLess(pos, len(jobs), rule_name)


if __name__ == "__main__":
    unittest.main()
