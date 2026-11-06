"""CLI parser defaults, validation, and entry-point smoke tests."""

from __future__ import annotations

import contextlib
import io
import os
import sqlite3
import subprocess
import sys
import unittest

from main import build_parser, validate_args
from tests.base import PROJECT_ROOT, TempProjectTestCase


class TestCliParser(unittest.TestCase):
    def _parse(self, *argv):
        parser = build_parser()
        return parser, parser.parse_args(list(argv))

    def test_defaults(self):
        _, args = self._parse()
        self.assertEqual(args.sqc_method, "ORTools")
        self.assertTrue(args.random_MTBF)
        self.assertFalse(args.multi_thread)
        self.assertEqual(args.thread_no, 1)
        self.assertFalse(args.force)
        self.assertFalse(args.processing_time_variability)

    def test_optional_flags(self):
        _, args = self._parse(
            "--no-random_MTBF", "-mt", "-tn", "3", "-y", "-pt_v", "-sqc", "FIFO"
        )
        self.assertFalse(args.random_MTBF)
        self.assertTrue(args.multi_thread)
        self.assertEqual(args.thread_no, 3)
        self.assertTrue(args.force)
        self.assertTrue(args.processing_time_variability)
        self.assertEqual(args.sqc_method, "FIFO")

        _, args = self._parse("-rnd_mtbf")
        self.assertTrue(args.random_MTBF)

    def test_invalid_numeric_ranges_are_rejected(self):
        cases = [
            ("-m_no", "0"),
            ("-span", "-1"),
            ("-pt_r", "10", "1"),
            ("-utl", "0"),
            ("-dt", "1.0"),
            ("-mtbf", "0"),
            ("-tn", "0"),
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                parser, args = self._parse(*argv)
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        validate_args(parser, args)


class TestCliEntryPoint(TempProjectTestCase):
    def _env(self):
        env = os.environ.copy()
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            str(PROJECT_ROOT) + (os.pathsep + existing if existing else "")
        )
        return env

    def test_main_runs_a_small_fifo_episode(self):
        db_path = self.tmp_path / "cli.db"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "-sqc",
                "FIFO",
                "-seed",
                "1",
                "-span",
                "20",
                "-draw",
                "0",
                "-save_gantt",
                "-ns",
                "-db",
                str(db_path),
            ],
            cwd=self.tmp_path,
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with contextlib.closing(sqlite3.connect(db_path)) as conn:
            row = conn.execute("SELECT status, total_jobs FROM runs").fetchone()
        self.assertEqual(row[0], "completed")
        self.assertGreater(row[1], 0)

    def test_non_interactive_stochastic_run_exits_nonzero(self):
        db_path = self.tmp_path / "blocked.db"
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "main.py"),
                "-sqc",
                "ORTools",
                "-seed",
                "1",
                "-span",
                "20",
                "-draw",
                "0",
                "-save_gantt",
                "-ns",
                "-pt_v",
                "-db",
                str(db_path),
            ],
            cwd=self.tmp_path,
            env=self._env(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Non-interactive", result.stderr + result.stdout)
        self.assertFalse(db_path.exists())

    def test_invalid_cli_arguments_exit_with_usage_error(self):
        result = subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "main.py"), "-m_no", "0"],
            cwd=self.tmp_path,
            env=self._env(),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("-m_no must be >= 1", result.stderr)


if __name__ == "__main__":
    unittest.main()
