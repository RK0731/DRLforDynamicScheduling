"""Unit tests for solver dispatch, lexicographic CP-SAT behavior, and status guards."""

import logging
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from ortools.sat.python import cp_model

from src.scheduler.scheduler import (
    GurobiOptimizer,
    ORTools,
    OPTIMIZATION_SOLVERS,
    is_optimization_scheduler,
    resolve_optimization_solver,
)


def make_job(pt: float, due: float, now: float = 0.0):
    """Minimal Job-like object consumed by `ORTools.solve_scheduling_problem`."""
    return SimpleNamespace(
        available_T=now,
        remaining_pt=[float(pt)],
        pt_by_m_idx=np.array([float(pt)]),
        due=float(due),
    )


class TestSchedulerRegistry(unittest.TestCase):
    def test_registry_contains_real_backends(self):
        self.assertIs(OPTIMIZATION_SOLVERS["ORTools"], ORTools)
        self.assertIs(OPTIMIZATION_SOLVERS["GurobiOptimizer"], GurobiOptimizer)

    def test_resolve_optimization_solver(self):
        self.assertIs(resolve_optimization_solver("ORTools"), ORTools)
        self.assertIs(resolve_optimization_solver("GurobiOptimizer"), GurobiOptimizer)
        self.assertIs(resolve_optimization_solver(ORTools), ORTools)
        self.assertIsNone(resolve_optimization_solver("FIFO"))
        self.assertIsNone(resolve_optimization_solver(lambda jobs: 0))

    def test_is_optimization_scheduler(self):
        self.assertTrue(is_optimization_scheduler("ORTools"))
        self.assertTrue(is_optimization_scheduler("GurobiOptimizer"))
        self.assertFalse(is_optimization_scheduler("FIFO"))
        self.assertFalse(is_optimization_scheduler("CR"))


class TestORToolsSolver(unittest.TestCase):
    def setUp(self):
        self.logger = logging.getLogger("test.ortools")
        self.env = SimpleNamespace(now=0.0)
        self.machine = SimpleNamespace(m_idx=0, release_T=0.0)

    def _solve(self, jobs):
        return ORTools.solve_scheduling_problem(
            self.logger,
            self.env,
            [self.machine],
            {(0, 1): [0]},
            {j_idx: [0] for j_idx in jobs},
            jobs,
        )

    def test_tardiness_is_optimized_before_makespan(self):
        # Both jobs are due at t=0. Scheduling the short job first yields cumulative
        # tardiness 1 + 11 = 12; the long job first yields 10 + 11 = 21.
        # Stage 1 must therefore put job 0 on the machine first.
        jobs = {0: make_job(pt=1, due=0), 1: make_job(pt=10, due=0)}
        begin_t, over_extended, status = self._solve(jobs)

        self.assertIsNotNone(begin_t)
        self.assertIn(status, ("OPTIMAL", "FEASIBLE"))
        self.assertEqual(begin_t[(0, 0)], 0.0)
        self.assertEqual(begin_t[(1, 0)], 1.0)
        self.assertIsNone(over_extended)

    def test_solution_uses_decimal_time_precision(self):
        jobs = {0: make_job(pt=1.5, due=100), 1: make_job(pt=2.0, due=100)}
        begin_t, _, status = self._solve(jobs)

        self.assertIn(status, ("OPTIMAL", "FEASIBLE"))
        for value in begin_t.values():
            # SCALE=10 means solved starts must be on the 0.1 grid, never truncated.
            self.assertAlmostEqual(value * 10, round(value * 10), places=9)

    def test_infeasible_status_returns_no_solution(self):
        jobs = {0: make_job(pt=1, due=0)}
        with mock.patch.object(
            cp_model.CpSolver, "Solve", return_value=cp_model.INFEASIBLE
        ):
            begin_t, over_extended, status = self._solve(jobs)

        self.assertIsNone(begin_t)
        self.assertIsNone(over_extended)
        self.assertEqual(status, "INFEASIBLE")

    def test_unknown_status_returns_no_solution(self):
        jobs = {0: make_job(pt=1, due=0)}
        with mock.patch.object(
            cp_model.CpSolver, "Solve", return_value=cp_model.UNKNOWN
        ):
            begin_t, _, status = self._solve(jobs)

        self.assertIsNone(begin_t)
        self.assertEqual(status, "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
