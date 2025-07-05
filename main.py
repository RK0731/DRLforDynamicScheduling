#!/usr/bin/env python3

import argparse
import inspect
from pathlib import Path
from typing import List

from src.scheduler.sequencing_rule import SequencingMethod
from src.simulator.simulator import Simulator, SimulatorMultiThread

parser = argparse.ArgumentParser(description='demonstration')

# system specification
parser.add_argument('-m_no', default=5, action='store', type=int, help='Number of Machines in system')
parser.add_argument('-seed', default=0, action='store', type=int, help='Random seed')
parser.add_argument('-span', default=100, action='store', type=int, help='Length of simulation')
parser.add_argument('-utl', '--E_utliz', default=0.6, action='store', type=float, help='Expected system utilization rate')

# job and processing time settings
parser.add_argument('-dt', '--due_tightness', default=2, action='store', type=float, help='Due time tightness')
parser.add_argument('-pt_r','--pt_range', default=[1,10], action='store', type=List[int], help='Range of processing time')
parser.add_argument('-pt_v', '--processing_time_variability', default=False, action='store_true', help='Flag to activate non-deterministic processing time (boolean)')
parser.add_argument('-pt_cv', default=0.1, action='store', type=float, help='Coefficiency of variance of processing time')

# machine breakdown settings
parser.add_argument('-mbkd', '--machine_breakdown', default=True, action='store_false', help='Simulate machine breakdown events? (boolean)')
parser.add_argument('-mtbf', '--MTBF', default=50, help='Mean time between failure')
parser.add_argument('-rnd_mtbf', '--random_MTBF', default=True, action='store_true', help='Use random MTBF')
parser.add_argument('-mttr', '--MTTR', default=10, help='Mean time to repair')
parser.add_argument('-rnd_mttr', '--random_MTTR', default=False, action='store_true', help='Use random MTTR')

# logging and plotting settings
parser.add_argument('-draw', '--draw_gantt', default=5, action='store', type=int, help='Any value greater than 0 would plot the gantt chart, strictly no-show for >200 simulation')
parser.add_argument('-save_gantt', default=True, action='store_false', help='Save the gantt chart figure to log?')
parser.add_argument('-ns', '--no_stream', default=False, action='store_false', help='Flag to disable stream logger (print to console)')

# select a scheudling rule or centralized scheduler
methods = dict(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
parser.add_argument('-sqc', '--sqc_method', default='GurobiOptimizer', help='Sequencing rule or scheduler')

# threading
parser.add_argument('-multi_thread', default=False , action='store_true', help='Use this flag to create multiple threads/environments')
parser.add_argument('-thread_no', default= 4, type=int, help='Number of threads')


args = parser.parse_args()


if __name__ == '__main__':
    Simulator.run(
        m_no = args.m_no, span = args.span, E_utliz = args.E_utliz, seed = args.seed, 
        pt_range = args.pt_range, due_tightness = args.due_tightness, 
        processing_time_variability = args.processing_time_variability, pt_cv = args.pt_cv,
        machine_breakdown = args.machine_breakdown, MTBF = args.MTBF, MTTR = args.MTTR, 
        random_MTBF = args.random_MTBF, random_MTTR = args.random_MTTR,
        stream = not args.no_stream, draw_gantt = args.draw_gantt, save_gantt = args.save_gantt,
        sqc_method = methods[args.sqc_method]
        )