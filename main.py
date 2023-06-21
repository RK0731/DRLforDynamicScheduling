#!/usr/bin/env python3

import argparse
from pathlib import Path
from simulator.sequencing_rule import *
from simulator.shopfloor import *

parser = argparse.ArgumentParser(description='demonstration')

# system specification
parser.add_argument('-m_no', default=5, action='store', type=int, help='Number of Machines in system')
parser.add_argument('-span', default=100, action='store', type=int, help='Length of simulation')
parser.add_argument('-utl', default=0.75, action='store', type=float, help='Expected system utilization rate')
parser.add_argument('-seed', default=0, action='store', type=int, help='Random seed')

# job and processing time settings
parser.add_argument('-ptr','--pt_range', default=[1,10], action='store', type=list, help='Range of processing time')
parser.add_argument('-dt', '--due_tightness', default=2, action='store', type=float, help='Due time tightness')
parser.add_argument('-ptv', '--pt_variability', default=False, action='store_true', help='Flag to use non-deterministic processing time? (boolean)')
parser.add_argument('-pt_cv', default=0.1, action='store', type=float, help='Coefficiency of variance of processing time')

# machine breakdown settings
parser.add_argument('-mbkd', '--machine_breakdown', default=True, action='store_false', help='Simulate machine breakdown? (boolean)')
parser.add_argument('-mtbf', default=100, help='Mean time between failure')
parser.add_argument('-rnd_mtbf', default=False, action='store_true', help='Use random MTBF (if flagged)')
parser.add_argument('-mttr', default=10, help='Mean time to repair')
parser.add_argument('-rnd_mttr', default=False, action='store_true', help='Use random MTTR (if flagged)')

# plotting settings
parser.add_argument('-draw', default=5, action='store', type=int, help='Any value greater than 0 would plot the gantt chart, always no-show when simulation span is longer than 200')
parser.add_argument('-save_gantt', default=True, action='store_false', help='Save the gantt chart?')

# scheduler
parser.add_argument('-sqc', default= SQC_rule.FIFO, type=lambda x:eval("SQC_rule."+str(x)), help='Sequencing rule or scheduler')

args = parser.parse_args()


if __name__ == '__main__':
    spf = Shopfloor(m_no = args.m_no, span = args.span, E_utliz = args.utl, seed = args.seed,
                    pt_range = args.pt_range, due_tightness = args.due_tightness, processing_time_variability = args.pt_variability, pt_cv = args.pt_variability,
                    machine_breakdown = args.machine_breakdown, MTBF = args.mtbf, MTTR = args.mttr, random_MTBF = args.rnd_mtbf, random_MTTR = args.rnd_mttr,
                    draw_gantt = args.draw, save_gantt = args.save_gantt,
                    sqc_rule = args.sqc
                    )
    spf.run_simulation()