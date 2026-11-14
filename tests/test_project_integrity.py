"""Project-wide integrity checks: syntax, importability, and required artifacts."""

from __future__ import annotations

import compileall
import importlib
import tomllib
import unittest

from src.scheduler.scheduler import OPTIMIZATION_SOLVERS
from src.scheduler.sequencing_rule import SequencingMethod
from tests.base import PROJECT_ROOT


class TestProjectIntegrity(unittest.TestCase):
    def test_all_python_sources_compile(self):
        ok = compileall.compile_dir(PROJECT_ROOT / "src", quiet=1)
        ok = compileall.compile_file(PROJECT_ROOT / "main.py", quiet=1) and ok
        ok = compileall.compile_file(PROJECT_ROOT / "test.py", quiet=1) and ok
        self.assertTrue(ok)

    def test_core_modules_import(self):
        modules = [
            "main",
            "src.utilities",
            "src.simulator.simulator",
            "src.simulator.event",
            "src.simulator.machine",
            "src.simulator.job",
            "src.simulator.exc",
            "src.scheduler.scheduler",
            "src.scheduler.sequencing_rule",
            "src.analysis.state_reconstruction",
            "src.DRL.brain_sequencing",
            "src.DRL.reward_function",
        ]
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_scheduler_registry_and_rules_are_complete(self):
        self.assertEqual(
            set(OPTIMIZATION_SOLVERS), {"GurobiOptimizer", "ORTools", "HiGHS"}
        )
        for rule in ("FIFO", "LIFO", "Slack", "CR", "SPT", "LWKR"):
            self.assertTrue(callable(getattr(SequencingMethod, rule)))

    def test_required_project_artifacts_exist(self):
        for name in (
            "README.md",
            "AGENTS.md",
            "WORKNOTE.md",
            "pyproject.toml",
            "uv.lock",
            ".python-version",
            "requirements.txt",
            "main.py",
            "test.py",
            "Makefile",
        ):
            self.assertTrue((PROJECT_ROOT / name).exists(), name)

    def test_python_version_is_pinned_to_313(self):
        py_version = (PROJECT_ROOT / ".python-version").read_text().strip()
        self.assertTrue(py_version.startswith("3.13"), py_version)

        with (PROJECT_ROOT / "pyproject.toml").open("rb") as f:
            project = tomllib.load(f)["project"]
        self.assertIn("3.13", project["requires-python"])
        self.assertIn("torch", " ".join(project["dependencies"]))


if __name__ == "__main__":
    unittest.main()
