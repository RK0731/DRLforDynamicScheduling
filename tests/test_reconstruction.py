"""Offline shopfloor reconstruction tests against a real recorded episode."""

from __future__ import annotations

import unittest

from src.analysis.state_reconstruction import ShopfloorStateReconstructor
from tests.base import TempProjectTestCase


class TestStateReconstruction(TempProjectTestCase):
    def setUp(self):
        super().setUp()
        self.run_sim(sqc_method="FIFO", span=80, seed=11)
        self.reconstructor = ShopfloorStateReconstructor(self.db_path, run_id=1)

    def test_decision_replay_matches_authoritative_queue(self):
        ids = self.reconstructor.decision_ids()
        self.assertGreater(len(ids), 0)
        self.assertEqual(ids, sorted(ids))

        for decision_id in ids:
            state = self.reconstructor.snapshot_at_decision(decision_id)
            dec = self.reconstructor.decision_by_id[decision_id]
            machine = state["machines"][dec["m_idx"]]

            self.assertEqual(
                [entry["j_idx"] for entry in machine["queue"]],
                [int(j) for j in dec["queue"]],
            )
            self.assertEqual(machine["queue_len"], len(dec["queue"]))
            self.assertEqual(
                state["summary"]["num_waiting_jobs"],
                sum(m["queue_len"] for m in state["machines"]),
            )
            self.assertEqual(state["target_decision"]["decision_id"], decision_id)

    def test_iter_snapshots_visits_every_decision(self):
        ids = self.reconstructor.decision_ids()
        snapshots = list(self.reconstructor.iter_snapshots())
        self.assertEqual(len(snapshots), len(ids))
        self.assertEqual(
            [snap["target_decision"]["decision_id"] for snap in snapshots], ids
        )

    def test_arbitrary_time_snapshot_structure(self):
        for t in (0.0, 20.0, 50.0, 200.0):
            state = self.reconstructor.snapshot(t)
            self.assertEqual(state["run_id"], 1)
            self.assertEqual(state["t"], t)
            self.assertEqual(len(state["machines"]), self.reconstructor.m_no)
            self.assertIn("num_waiting_jobs", state["summary"])
            self.assertIn("num_processing_machines", state["summary"])
            for machine in state["machines"]:
                self.assertIn("queue", machine)
                self.assertIn("status", machine)

    def test_incomplete_operation_phase_semantics(self):
        op = {
            "completed": 0,
            "begin_t": 10.0,
            "end_t": 20.0,
            "arrival_t": 5.0,
            "decision_id": 5,
        }
        phase = self.reconstructor._phase
        self.assertEqual(phase(op, 30.0, "post"), "processing")
        self.assertEqual(phase(op, 10.0, "post"), "processing")
        self.assertEqual(phase(op, 10.0, "pre"), "waiting")
        self.assertEqual(phase(op, 10.0, "post", cutoff=5), "waiting")
        self.assertEqual(phase(op, 10.0, "post", cutoff=6), "processing")
        self.assertEqual(phase(op, 1.0, "post"), "future")

    def test_incomplete_operation_keeps_machine_busy(self):
        op = {
            "j_idx": 0,
            "op_seq": 0,
            "m_idx": 0,
            "begin_t": 10.0,
            "end_t": 20.0,
            "arrival_t": 5.0,
            "completed": 0,
            "decision_id": 5,
        }
        self.reconstructor.ops_by_machine[0] = [op]
        self.reconstructor._begin_times[0] = [10.0]

        self.assertIs(self.reconstructor._current_op(0, 30.0, "post"), op)
        self.assertIs(self.reconstructor._current_op(0, 10.0, "post"), op)
        self.assertIsNone(self.reconstructor._current_op(0, 10.0, "post", cutoff=5))


if __name__ == "__main__":
    unittest.main()
