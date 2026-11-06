"""Shared test infrastructure: temporary databases, isolated logs, and sim helpers."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict

from src import utilities
from src.simulator.simulator import Simulator

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TempProjectTestCase(unittest.TestCase):
    """Base class that keeps every test away from the project's real `logs/` directory.

    `src.utilities.LOG_ROOT_DIR` / `DEFAULT_DB_PATH` are monkeypatched for in-process runs,
    and `SIM_LOG_ROOT` is exported so spawn-based worker processes write their logs into the
    same temporary root (the environment is inherited by children).
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory(prefix="drl-djssp-tests-")
        self.tmp_path = Path(self._tmp.name)
        self.log_root = self.tmp_path / "logs"
        self.db_path = self.tmp_path / "simulation.db"

        self._old_log_root = utilities.LOG_ROOT_DIR
        self._old_default_db = utilities.DEFAULT_DB_PATH
        self._old_log_env = os.environ.get("SIM_LOG_ROOT")

        utilities.LOG_ROOT_DIR = self.log_root
        utilities.DEFAULT_DB_PATH = self.log_root / "simulation.db"
        os.environ["SIM_LOG_ROOT"] = str(self.log_root)

    def tearDown(self) -> None:
        utilities.LOG_ROOT_DIR = self._old_log_root
        utilities.DEFAULT_DB_PATH = self._old_default_db
        if self._old_log_env is None:
            os.environ.pop("SIM_LOG_ROOT", None)
        else:
            os.environ["SIM_LOG_ROOT"] = self._old_log_env
        self._tmp.cleanup()

    # ------------------------------------------------------------------ helpers
    def sim_kwargs(self, **overrides: Any) -> Dict[str, Any]:
        """Return a complete, fast, non-interactive configuration for `Shopfloor`."""
        kwargs: Dict[str, Any] = dict(
            m_no=5,
            span=60,
            E_utliz=0.6,
            seed=42,
            pt_range=[1, 10],
            due_tightness=2.0,
            processing_time_variability=False,
            pt_cv=0.1,
            machine_breakdown=True,
            MTBF=50.0,
            MTTR=10.0,
            random_MTBF=True,
            random_MTTR=False,
            stream=False,
            draw_gantt=0,
            save_gantt=False,
            db_path=str(self.db_path),
            sqc_method="FIFO",
            force=True,
            multi_thread=False,
            thread_no=1,
        )
        kwargs.update(overrides)
        return kwargs

    def run_sim(self, **overrides: Any) -> None:
        """Run one in-process episode with isolated logs and database."""
        Simulator.run(**self.sim_kwargs(**overrides))

    def connect(self) -> sqlite3.Connection:
        """Open the test database with row access by column name."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        return conn
