#!/usr/bin/env python3
"""Convenience runner for the verification suite.

Equivalent to:

    python -m unittest discover -s tests -t . -v

Run this from the repository root.
"""

import unittest


def main() -> int:
    suite = unittest.defaultTestLoader.discover("tests", top_level_dir=".")
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
