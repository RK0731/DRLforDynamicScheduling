"""HiGHS solver tests (subprocess-isolated).

Standalone `highspy` (HiGHS 1.x) is binary-incompatible with OR-Tools, which bundles
its own older HiGHS. Loading both native libraries into one process segfaults on macOS
(a symbol clash in `libhighs` / `libortools`), and the rest of the test suite imports
`ortools.sat.python.cp_model`. A `-sqc HiGHS` CLI run never imports OR-Tools (both
backends are lazily imported), so it is unaffected — but these tests must run the HiGHS
solver/simulation in a fresh subprocess and assert on its JSON / SQLite output.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import unittest

from tests.base import PROJECT_ROOT, TempProjectTestCase

# Solves a two-job, single-machine instance and prints the result as one JSON line.
_SOLVE_SNIPPET = textwrap.dedent(
    """
    import json, logging, sys
    from types import SimpleNamespace
    import numpy as np
    from src.scheduler.scheduler import HiGHSOptimizer

    def make_job(pt, due):
        return SimpleNamespace(
            available_T=0.0, remaining_pt=[float(pt)],
            pt_by_m_idx=np.array([float(pt)]), due=float(due),
        )

    pt0, pt1, due0, due1 = map(float, sys.argv[1:5])
    env = SimpleNamespace(now=0.0)
    machine = SimpleNamespace(m_idx=0, release_T=0.0)
    jobs = {0: make_job(pt0, due0), 1: make_job(pt1, due1)}
    begin_t, over, status = HiGHSOptimizer.solve_scheduling_problem(
        logging.getLogger("highs"), env, [machine],
        {(0, 1): [0]}, {j: [0] for j in jobs}, jobs,
    )
    print(json.dumps({
        "status": status,
        "over": over,
        "begin_t": {"0,0": begin_t[(0, 0)], "1,0": begin_t[(1, 0)]} if begin_t else None,
    }))
    """
)

# Runs a full HiGHS simulation into the SQLite database passed as argv[1].
_SIM_SNIPPET = textwrap.dedent(
    """
    import sys
    from src.simulator.simulator import Simulator
    Simulator.run(
        m_no=5, span=60, E_utliz=0.6, seed=42,
        pt_range=[1, 10], due_tightness=2.0,
        processing_time_variability=False, pt_cv=0.1,
        machine_breakdown=False, MTBF=50.0, MTTR=10.0,
        random_MTBF=True, random_MTTR=False,
        stream=False, draw_gantt=0, save_gantt=False,
        db_path=sys.argv[1], sqc_method="HiGHS", force=True,
    )
    """
)


def _run(snippet: str, *args: str) -> str:
    """Run `snippet` in a fresh Python process (isolated from OR-Tools) and return stdout."""
    proc = subprocess.run(
        [sys.executable, "-c", snippet, *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        env=dict(os.environ),
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"HiGHS subprocess failed (exit {proc.returncode}):\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return proc.stdout


class TestHiGHSSolver(TempProjectTestCase):
    def _solve(self, pt0, pt1, due0, due1):
        out = _run(_SOLVE_SNIPPET, pt0, pt1, due0, due1)
        return json.loads(out.strip().splitlines()[-1])

    def test_tardiness_is_optimized_before_makespan(self):
        # Both jobs due at t=0. Short-job-first gives cumulative tardiness 1 + 11 = 12;
        # long-job-first gives 10 + 11 = 21. Stage 1 must put job 0 first.
        result = self._solve("1", "10", "0", "0")

        self.assertIsNotNone(result["begin_t"])
        self.assertIn(result["status"], ("Optimal", "TimeLimit"))
        self.assertEqual(result["begin_t"]["0,0"], 0.0)
        self.assertEqual(result["begin_t"]["1,0"], 1.0)
        self.assertIsNone(result["over"])

    def test_continuous_time_preserves_decimal_precision(self):
        # HiGHS uses continuous variables: no SCALE transform, so sub-unit times are
        # solved exactly (1.5) rather than truncated.
        result = self._solve("1.5", "2.0", "100", "100")

        self.assertIsNotNone(result["begin_t"])
        self.assertEqual(result["begin_t"]["0,0"], 0.0)
        self.assertEqual(result["begin_t"]["1,0"], 1.5)

    def test_highs_run_integrity_and_schedule_alignment(self):
        _run(_SIM_SNIPPET, str(self.db_path))
        conn = self.connect()

        run = conn.execute("SELECT * FROM runs").fetchone()
        self.assertEqual(run["status"], "completed")
        self.assertEqual(run["sqc_method"], "HiGHS")
        self.assertGreater(run["sqc_cnt_opt"], 0)

        cycles = conn.execute("SELECT solver_status FROM schedule_cycles").fetchall()
        self.assertGreater(len(cycles), 0)
        statuses = {row["solver_status"] for row in cycles}
        self.assertTrue(
            all("PASSIVE_FALLBACK" not in s for s in statuses),
            f"unexpected solver fallback: {statuses}",
        )
        self.assertTrue(any(s == "Optimal" for s in statuses))

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

        self.assertEqual(conn.execute("PRAGMA foreign_key_check").fetchall(), [])


if __name__ == "__main__":
    unittest.main()
