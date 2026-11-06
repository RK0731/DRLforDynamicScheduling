#!/usr/bin/env python3
"""
Author: Liu Renke, dr.liu.renke@gmail.com
Main entry-point script to configure and start the Dynamic Job Shop Scheduling simulation.
"""

import argparse
import inspect

from src.scheduler.sequencing_rule import SequencingMethod
from src.scheduler.scheduler import OPTIMIZATION_SOLVERS
from src.simulator.simulator import Simulator, SimulatorMultiThread


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Discrete-Event Simulation for Dynamic Job Shop Scheduling Problem (DJSSP)'
    )

    # 1. System specification parameters
    parser.add_argument('-m_no', default=5, action='store', type=int,
                        help='Number of machines (workstations) in the shopfloor system')
    parser.add_argument('-seed', default=0, action='store', type=int,
                        help='Random seed (0 means generate a random seed automatically)')
    parser.add_argument('-span', default=100, action='store', type=int,
                        help='Nominal length (horizon) of the simulation for generating job arrivals')
    parser.add_argument('-utl', '--E_utliz', default=0.6, action='store', type=float,
                        help='Expected system utilization rate (determines job arrival frequency)')

    # 2. Job and processing time parameters
    parser.add_argument('-dt', '--due_tightness', default=2.0, action='store', type=float,
                        help='Due time tightness factor (upper bound of uniform multiplier on total processing time)')
    parser.add_argument('-pt_r', '--pt_range', default=[1, 10], nargs=2, type=int,
                        help='Range of nominal processing time [min_pt, max_pt]')
    parser.add_argument('-pt_v', '--processing_time_variability', default=False, action='store_true',
                        help='Flag to activate non-deterministic (stochastic) processing time')
    parser.add_argument('-pt_cv', default=0.1, action='store', type=float,
                        help='Coefficient of variation (CV) of processing time when variability is enabled')

    # 3. Machine breakdown parameters
    parser.add_argument('-mbkd', '--machine_breakdown', default=True, action='store_false',
                        help='Flag to DISABLE machine breakdown events (enabled by default)')
    parser.add_argument('-mtbf', '--MTBF', default=50.0, action='store', type=float,
                        help='Mean Time Between Failures (MTBF)')
    parser.add_argument('-rnd_mtbf', '--random_MTBF', dest='random_MTBF', default=True,
                        action='store_true',
                        help='Sample time between failures from an exponential distribution (default)')
    parser.add_argument('--no-random_MTBF', dest='random_MTBF', action='store_false',
                        help='Use a deterministic MTBF instead of sampling from an exponential distribution')
    parser.add_argument('-mttr', '--MTTR', default=10.0, action='store', type=float,
                        help='Mean Time To Repair (MTTR)')
    parser.add_argument('-rnd_mttr', '--random_MTTR', default=False, action='store_true',
                        help='Sample repair duration from a uniform distribution [0.5*MTTR, 1.5*MTTR]')

    # 4. Logging, database, plotting, and execution parameters
    parser.add_argument('-db', '--db_path', default=None, action='store', type=str,
                        help='Path to the persistent SQLite database file (defaults to logs/simulation.db)')
    parser.add_argument('-draw', '--draw_gantt', default=5, action='store', type=int,
                        help='Seconds to display Gantt chart pop-up (>0 shows plot; disabled automatically if span >= 250)')
    parser.add_argument('-save_gantt', default=True, action='store_false',
                        help='Flag to DISABLE saving the Gantt chart figure to the log directory (saved by default)')
    parser.add_argument('-ns', '--no_stream', default=False, action='store_true',
                        help='Flag to disable stream logger (suppress console output)')
    parser.add_argument('-y', '--force', default=False, action='store_true',
                        help='Proceed without interactive confirmation (required for stochastic '
                             'occupation times with an optimization scheduler when stdin is not a TTY)')
    parser.add_argument('-mt', '--multi_thread', default=False, action='store_true',
                        help='Run multiple independent episodes in parallel processes')
    parser.add_argument('-tn', '--thread_no', default=1, action='store', type=int,
                        help='Number of parallel worker processes when --multi_thread is enabled')

    # 5. Select a sequencing heuristic rule or centralized mathematical optimization scheduler
    # Heuristic rules are callables on `SequencingMethod`; optimization backends are registered
    # by name in `OPTIMIZATION_SOLVERS` (see `src/scheduler/scheduler.py`).
    heuristic_methods = dict(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
    parser.add_argument('-sqc', '--sqc_method', default='ORTools',
                        choices=list(heuristic_methods.keys()) + list(OPTIMIZATION_SOLVERS.keys()),
                        help='Sequencing rule or centralized scheduler to use (default: ORTools)')
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail fast on invalid numeric ranges instead of producing a confusing runtime error."""
    if args.m_no < 1:
        parser.error('-m_no must be >= 1')
    if args.span < 0:
        parser.error('-span must be >= 0')
    if args.pt_range[0] <= 0 or args.pt_range[0] > args.pt_range[1]:
        parser.error('-pt_r requires 0 < min_pt <= max_pt')
    if not (0.0 < args.E_utliz <= 1.0):
        parser.error('-utl/--E_utliz must be in (0, 1]')
    if args.due_tightness < 1.2:
        parser.error('-dt/--due_tightness must be >= 1.2 (due dates are sampled from U(1.2, dt))')
    if args.processing_time_variability and args.pt_cv < 0:
        parser.error('-pt_cv must be >= 0')
    if args.machine_breakdown and (args.MTBF <= 0 or args.MTTR <= 0):
        parser.error('-mtbf and -mttr must be > 0 when machine breakdowns are enabled')
    if args.thread_no < 1:
        parser.error('-tn/--thread_no must be >= 1')


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)

    heuristic_methods = dict(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
    # A heuristic name resolves to its callable; an optimization backend stays a name
    # string that `CentralScheduler` looks up in `OPTIMIZATION_SOLVERS`.
    sqc_method = heuristic_methods.get(args.sqc_method, args.sqc_method)

    kwargs = dict(
        m_no=args.m_no,
        span=args.span,
        E_utliz=args.E_utliz,
        seed=args.seed,
        pt_range=args.pt_range,
        due_tightness=args.due_tightness,
        processing_time_variability=args.processing_time_variability,
        pt_cv=args.pt_cv,
        machine_breakdown=args.machine_breakdown,
        MTBF=args.MTBF,
        MTTR=args.MTTR,
        random_MTBF=args.random_MTBF,
        random_MTTR=args.random_MTTR,
        stream=not args.no_stream,
        draw_gantt=args.draw_gantt,
        save_gantt=args.save_gantt,
        db_path=args.db_path,
        sqc_method=sqc_method,
        force=args.force,
        multi_thread=args.multi_thread,
        thread_no=args.thread_no,
    )

    if args.multi_thread:
        SimulatorMultiThread(**kwargs)
    else:
        Simulator.run(**kwargs)


if __name__ == '__main__':
    main()
