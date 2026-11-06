"""SQLite schema, migration, and run-lifecycle tests."""

from __future__ import annotations

from contextlib import closing
import sqlite3
import unittest

from src.utilities import SimulationDatabase
from tests.base import TempProjectTestCase


class TestSimulationDatabase(TempProjectTestCase):
    def _run_record(self, created_at: str = "2026-01-01 00:00:00") -> dict:
        return dict(
            created_at=created_at,
            seed=1,
            sqc_method="FIFO",
            m_no=5,
            span=100.0,
            e_utliz=0.6,
            due_tightness=2.0,
            pt_min=1.0,
            pt_max=10.0,
            pt_cv=0.0,
            machine_bkd=0,
            mtbf=None,
            random_mtbf=None,
            mttr=None,
            random_mttr=None,
        )

    def _finalize_record(self) -> dict:
        return dict(
            actual_span=90.0,
            actual_utliz=0.5,
            total_jobs=4,
            completed_jobs=4,
            sqc_cnt_opt=0,
            sqc_cnt_si=0,
            sqc_cnt_reactive=3,
            sqc_cnt_passive=1,
            wall_time_s=0.5,
            opt_time_s=0.0,
            mean_tardiness=1.0,
            max_tardiness=2.0,
            mean_flowtime=10.0,
            max_flowtime=20.0,
        )

    def test_schema_contains_new_columns(self):
        SimulationDatabase(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as conn:
            run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            op_cols = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
            dec_cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}

        self.assertTrue({"status", "error"} <= run_cols)
        self.assertIn("completed", op_cols)
        self.assertTrue(
            {"queue_json", "candidates_json", "state_json", "action", "reward"} <= dec_cols
        )
        self.assertNotIn("rule_picks_json", dec_cols)

    def test_migration_restores_columns_and_drops_legacy(self):
        SimulationDatabase(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("ALTER TABLE runs DROP COLUMN status")
            conn.execute("ALTER TABLE runs DROP COLUMN error")
            conn.execute("ALTER TABLE operations DROP COLUMN completed")
            conn.execute("ALTER TABLE decisions DROP COLUMN candidates_json")
            conn.execute("ALTER TABLE decisions ADD COLUMN rule_picks_json TEXT")
            conn.commit()

        SimulationDatabase(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as conn:
            run_cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)")}
            op_cols = {row[1] for row in conn.execute("PRAGMA table_info(operations)")}
            dec_cols = {row[1] for row in conn.execute("PRAGMA table_info(decisions)")}

        self.assertTrue({"status", "error"} <= run_cols)
        self.assertIn("completed", op_cols)
        self.assertIn("candidates_json", dec_cols)
        self.assertNotIn("rule_picks_json", dec_cols)

    def test_run_lifecycle_running_completed_failed(self):
        db = SimulationDatabase(self.db_path)
        run_id = db.create_run(self._run_record())
        with closing(sqlite3.connect(self.db_path)) as conn:
            self.assertEqual(
                conn.execute("SELECT status FROM runs WHERE run_id=?", (run_id,)).fetchone()[0],
                "running",
            )

        db.finalize_run(run_id, self._finalize_record())
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status, error, total_jobs, mean_tardiness FROM runs WHERE run_id=?",
                (run_id,),
            ).fetchone()
        self.assertEqual(row[0], "completed")
        self.assertIsNone(row[1])
        self.assertEqual(row[2], 4)
        self.assertEqual(row[3], 1.0)

        failed_id = db.create_run(self._run_record("2026-01-02 00:00:00"))
        db.fail_run(failed_id, "boom")
        with closing(sqlite3.connect(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status, error FROM runs WHERE run_id=?", (failed_id,)
            ).fetchone()
        self.assertEqual(row[0], "failed")
        self.assertEqual(row[1], "boom")

    def test_migration_backfills_completed_status_for_old_rows(self):
        SimulationDatabase(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as conn:
            run_id = conn.execute(
                """
                INSERT INTO runs (
                    created_at, seed, sqc_method, m_no, span, e_utliz,
                    due_tightness, pt_min, pt_max, pt_cv, machine_bkd, actual_span
                ) VALUES ('old', 1, 'FIFO', 5, 100, 0.6, 2.0, 1, 10, 0, 0, 80)
                """
            ).lastrowid
            conn.commit()

        SimulationDatabase(self.db_path)
        with closing(sqlite3.connect(self.db_path)) as conn:
            status = conn.execute(
                "SELECT status FROM runs WHERE run_id=?", (run_id,)
            ).fetchone()[0]
        self.assertEqual(status, "completed")


if __name__ == "__main__":
    unittest.main()
