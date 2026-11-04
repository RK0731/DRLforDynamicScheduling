#!/usr/bin/env python3
"""
Command-line entry point for offline shopfloor state reconstruction.

Usage::

    python -m src.analysis --run_id 1 --time 77.0
    python -m src.analysis --run_id 1 --decision_id 42
    python -m src.analysis                      # latest run, first decision
"""

import argparse
import json

from ..utilities import DEFAULT_DB_PATH
from .state_reconstruction import ShopfloorStateReconstructor


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct the complete shopfloor state from a recorded simulation run."
    )
    parser.add_argument("-db", "--db_path", default=str(DEFAULT_DB_PATH), help="SQLite database path")
    parser.add_argument("-r", "--run_id", type=int, default=None, help="Run id (default: latest)")
    parser.add_argument("-t", "--time", type=float, default=None, help="Simulation time to reconstruct")
    parser.add_argument("-d", "--decision_id", type=int, default=None, help="Reconstruct a recorded decision")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation")
    args = parser.parse_args()

    recon = ShopfloorStateReconstructor(db_path=args.db_path, run_id=args.run_id)
    if args.decision_id is not None:
        state = recon.snapshot_at_decision(args.decision_id)
    else:
        t = args.time if args.time is not None else recon.decisions[0]["sim_t"]
        state = recon.snapshot(t)
    print(json.dumps(state, indent=args.indent, default=str))


if __name__ == "__main__":
    main()
