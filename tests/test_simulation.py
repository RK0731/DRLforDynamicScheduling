"""End-to-end simulator invariants, failure handling, and multi-process checks."""

from __future__ import annotations

import json
import sqlite3
import unittest
from unittest import mock

from src.scheduler import scheduler as scheduler_module
from src.simulator.event import Recorder
from src.simulator.exc import SimulatorError
from src.simulator.simulator import SimulatorMultiThread
from tests.base import TempProjectTestCase


class TestSimulationIntegrity(TempProjectTestCase):
    def _assert_decision_invariants(self, conn):
        decisions = conn.execute(
            "SELECT * FROM decisions ORDER BY decision_id"
        ).fetchall()
        self.assertGreater(len(decisions), 0)
        for dec in decisions:
            queue = json.loads(dec["queue_json"])
            candidates = json.loads(dec["candidates_json"])
            self.assertEqual(dec["queue_len"], len(queue))
            self.assertLess(dec["chosen_pos"], len(queue))
            self.assertGreaterEqual(dec["chosen_pos"], 0)
            self.assertEqual(queue[dec["chosen_pos"]], dec["j_idx"])
            self.assertEqual(set(candidates.keys()), {str(j) for j in queue})
            self.assertIsInstance(dec["sim_t"], float)

    def test_fifo_run_integrity(self):
        self.run_sim(sqc_method="FIFO", span=60, seed=42)
        conn = self.connect()

        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual(run["status"], "completed")
        self.assertIsNone(run["error"])
        self.assertGreater(run["total_jobs"], 0)
        self.assertEqual(run["total_jobs"], run["completed_jobs"])
        self.assertGreater(run["completed_jobs"], 0)
        self.assertIsNotNone(run["mean_tardiness"])

        jobs = conn.execute("SELECT * FROM jobs").fetchall()
        self.assertEqual(len(jobs), run["total_jobs"])
        operations = conn.execute("SELECT * FROM operations").fetchall()
        m_no = int(run["m_no"])
        self.assertEqual(len(operations), m_no * len(jobs))

        for job in jobs:
            trajectory = json.loads(job["trajectory_json"])
            self.assertEqual(len(trajectory), m_no)
            self.assertEqual(len(set(trajectory)), m_no)
            ops = conn.execute(
                "SELECT * FROM operations WHERE j_idx = ? ORDER BY op_seq",
                (job["j_idx"],),
            ).fetchall()
            self.assertEqual(len(ops), m_no)
            self.assertEqual([op["op_seq"] for op in ops], list(range(m_no)))
            for op in ops:
                self.assertEqual(op["completed"], 1)
                self.assertGreaterEqual(op["begin_t"] + 0.01, op["arrival_t"])
                self.assertGreaterEqual(op["end_t"] + 1e-9, op["begin_t"])
                self.assertGreater(op["expected_pt"], 0)
                self.assertGreater(op["actual_pt"], 0)
                self.assertGreaterEqual(op["wait_t"], 0)

        self._assert_decision_invariants(conn)
        self.assertGreater(conn.execute("SELECT COUNT(*) FROM sim_events").fetchone()[0], 0)
        self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_ortools_run_integrity_and_schedule_alignment(self):
        # Disable breakdowns here so a deterministic PT schedule can be compared exactly.
        self.run_sim(
            sqc_method="ORTools",
            span=60,
            seed=42,
            machine_breakdown=False,
        )
        conn = self.connect()

        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual(run["status"], "completed")
        self.assertGreater(run["sqc_cnt_opt"], 0)

        cycles = conn.execute("SELECT solver_status FROM schedule_cycles").fetchall()
        self.assertGreater(len(cycles), 0)
        statuses = {row["solver_status"] for row in cycles}
        self.assertTrue(
            all("PASSIVE_FALLBACK" not in s for s in statuses),
            f"unexpected solver fallback: {statuses}",
        )
        self.assertTrue(any(s in ("OPTIMAL", "FEASIBLE") for s in statuses))

        # Every scheduled operation should start at its promised time under deterministic
        # processing times and no breakdowns.
        scheduled = conn.execute(
            """
            SELECT begin_t, scheduled_begin_t
            FROM operations
            WHERE decision_type = 'Scheduled' AND scheduled_begin_t IS NOT NULL
            """
        ).fetchall()
        self.assertGreater(len(scheduled), 0)
        for op in scheduled:
            self.assertAlmostEqual(op["begin_t"], op["scheduled_begin_t"], delta=0.11)

        self._assert_decision_invariants(conn)
        self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])

    def test_infeasible_solver_falls_back_to_passive_schedule(self):
        class InfeasibleSolver:
            @classmethod
            def solve_scheduling_problem(cls, *args, **kwargs):
                return None, None, "INFEASIBLE"

        old = scheduler_module.OPTIMIZATION_SOLVERS["ORTools"]
        scheduler_module.OPTIMIZATION_SOLVERS["ORTools"] = InfeasibleSolver
        self.addCleanup(
            lambda: scheduler_module.OPTIMIZATION_SOLVERS.__setitem__("ORTools", old)
        )

        self.run_sim(sqc_method="ORTools", span=40, seed=7)
        conn = self.connect()

        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual(run["status"], "completed")
        statuses = {
            row["solver_status"]
            for row in conn.execute("SELECT solver_status FROM schedule_cycles")
        }
        self.assertTrue(any("PASSIVE_FALLBACK" in s for s in statuses), statuses)
        self._assert_decision_invariants(conn)

    def test_solver_exception_marks_run_failed_and_reraises(self):
        class ExplodingSolver:
            @classmethod
            def solve_scheduling_problem(cls, *args, **kwargs):
                raise RuntimeError("synthetic solver failure")

        old = scheduler_module.OPTIMIZATION_SOLVERS["ORTools"]
        scheduler_module.OPTIMIZATION_SOLVERS["ORTools"] = ExplodingSolver
        self.addCleanup(
            lambda: scheduler_module.OPTIMIZATION_SOLVERS.__setitem__("ORTools", old)
        )

        with self.assertRaises(RuntimeError):
            self.run_sim(sqc_method="ORTools", span=40, seed=7)

        conn = self.connect()
        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual(run["status"], "failed")
        self.assertIn("synthetic solver failure", run["error"])
        self.assertIsNone(run["total_jobs"])

    def test_non_interactive_stochastic_run_requires_force(self):
        with mock.patch("sys.stdin.isatty", return_value=False):
            with self.assertRaises(SimulatorError):
                self.run_sim(
                    sqc_method="ORTools",
                    span=30,
                    seed=3,
                    machine_breakdown=False,
                    processing_time_variability=True,
                    force=False,
                )
        self.assertFalse(self.db_path.exists())

        # Explicit opt-in must proceed and finish without ever reading stdin, even when
        # a TTY is attached (so `-y` is deterministic in terminals too).
        with mock.patch("sys.stdin.isatty", return_value=True), mock.patch(
            "builtins.input",
            side_effect=AssertionError("force=True must not call input()"),
        ):
            self.run_sim(
                sqc_method="ORTools",
                span=30,
                seed=3,
                machine_breakdown=False,
                processing_time_variability=True,
                force=True,
            )
        conn = self.connect()
        self.assertEqual(
            conn.execute("SELECT status FROM runs").fetchone()["status"], "completed"
        )

    def test_same_seed_is_reproducible(self):
        db_a = self.tmp_path / "seed_a.db"
        db_b = self.tmp_path / "seed_b.db"
        self.run_sim(sqc_method="FIFO", span=40, seed=123, db_path=str(db_a))
        self.run_sim(sqc_method="FIFO", span=40, seed=123, db_path=str(db_b))

        snapshots = []
        for path in (db_a, db_b):
            conn = sqlite3.connect(path)
            conn.row_factory = sqlite3.Row
            run = conn.execute(
                """
                SELECT total_jobs, completed_jobs, actual_span,
                       mean_tardiness, max_tardiness, mean_flowtime, max_flowtime
                FROM runs
                """
            ).fetchone()
            ops = conn.execute(
                """
                SELECT j_idx, op_seq, m_idx, begin_t, end_t, actual_pt
                FROM operations ORDER BY j_idx, op_seq
                """
            ).fetchall()
            conn.close()
            snapshots.append((dict(run), [tuple(row) for row in ops]))

        self.assertEqual(snapshots[0], snapshots[1])

    def test_streaming_flush_preserves_all_events(self):
        db_default = self.tmp_path / "events_default.db"
        db_streamed = self.tmp_path / "events_streamed.db"
        self.run_sim(sqc_method="FIFO", span=20, seed=5, db_path=str(db_default))
        with mock.patch.object(Recorder, "FLUSH_EVENTS_THRESHOLD", 5):
            self.run_sim(sqc_method="FIFO", span=20, seed=5, db_path=str(db_streamed))

        counts = []
        for path in (db_default, db_streamed):
            with sqlite3.connect(path) as conn:
                counts.append(conn.execute("SELECT COUNT(*) FROM sim_events").fetchone()[0])
        self.assertEqual(counts[0], counts[1])
        self.assertGreater(counts[0], 5)

    def test_multi_process_runs_are_isolated_and_complete(self):
        SimulatorMultiThread(
            **self.sim_kwargs(
                sqc_method="FIFO",
                span=30,
                seed=100,
                multi_thread=True,
                thread_no=2,
            )
        )
        conn = self.connect()
        runs = conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        self.assertEqual(len(runs), 2)
        self.assertTrue(all(r["status"] == "completed" for r in runs))
        self.assertEqual(len({r["seed"] for r in runs}), 2)


if __name__ == "__main__":
    unittest.main()
