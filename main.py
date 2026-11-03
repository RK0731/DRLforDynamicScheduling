#!/usr/bin/env python3
"""
Author: Liu Renke, dr.liu.renke@gmail.com
Main entry-point script to configure and start the Dynamic Job Shop Scheduling simulation.
"""

import argparse
import inspect

from src.scheduler.sequencing_rule import SequencingMethod
from src.simulator.simulator import Simulator, SimulatorMultiThread

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
                    help='Flag to disable machine breakdown events (enabled by default)')
parser.add_argument('-mtbf', '--MTBF', default=50.0, action='store', type=float,
                    help='Mean Time Between Failures (MTBF)')
parser.add_argument('-rnd_mtbf', '--random_MTBF', default=True, action='store_true',
                    help='Sample time between failures from an exponential distribution')
parser.add_argument('-mttr', '--MTTR', default=10.0, action='store', type=float,
                    help='Mean Time To Repair (MTTR)')
parser.add_argument('-rnd_mttr', '--random_MTTR', default=False, action='store_true',
                    help='Sample repair duration from a uniform distribution [0.5*MTTR, 1.5*MTTR]')

# 4. Logging, database, and plotting parameters
parser.add_argument('-db', '--db_path', default=None, action='store', type=str,
                    help='Path to the persistent SQLite database file (defaults to logs/simulation.db)')
parser.add_argument('-draw', '--draw_gantt', default=5, action='store', type=int,
                    help='Seconds to display Gantt chart pop-up (>0 shows plot; disabled automatically if span >= 250)')
parser.add_argument('-save_gantt', default=True, action='store_false',
                    help='Flag to disable saving the Gantt chart figure to the log directory (saved by default)')
parser.add_argument('-ns', '--no_stream', default=False, action='store_true',
                    help='Flag to disable stream logger (suppress console output)')

# 5. Select a sequencing heuristic rule or centralized mathematical optimization scheduler
methods = dict(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
parser.add_argument('-sqc', '--sqc_method', default='GurobiOptimizer', choices=list(methods.keys()),
                    help='Sequencing rule or centralized scheduler to use')

args = parser.parse_args()

if __name__ == '__main__':
    Simulator.run(
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
        sqc_method=methods[args.sqc_method],
    )
