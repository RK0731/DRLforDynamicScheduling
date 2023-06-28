#!/usr/bin/env python3

import argparse
from pathlib import Path
from simulator.sequencing_rule import *
from simulator.simulator import *

parser = argparse.ArgumentParser(description='demonstration')

# system specification
parser.add_argument('-m_no', default=5, action='store', type=int, help='Number of Machines in system')
parser.add_argument('-span', default=50, action='store', type=int, help='Length of simulation')
parser.add_argument('-utl', '--E_utliz', default=0.75, action='store', type=float, help='Expected system utilization rate')
parser.add_argument('-seed', default=0, action='store', type=int, help='Random seed')

# job and processing time settings
parser.add_argument('-pt_r','--pt_range', default=[1,10], action='store', type=list, help='Range of processing time')
parser.add_argument('-dt', '--due_tightness', default=2, action='store', type=float, help='Due time tightness')
parser.add_argument('-pt_v', '--processing_time_variability', default=False, action='store_true', help='Flag to use non-deterministic processing time? (boolean)')
parser.add_argument('-pt_cv', default=0.1, action='store', type=float, help='Coefficiency of variance of processing time')

# machine breakdown settings
parser.add_argument('-mbkd', '--machine_breakdown', default=True, action='store_false', help='Simulate machine breakdown? (boolean)')
parser.add_argument('-mtbf', '--MTBF', default=50, help='Mean time between failure')
parser.add_argument('-rnd_mtbf', '--random_MTBF', default=False, action='store_true', help='Use random MTBF')
parser.add_argument('-mttr', '--MTTR', default=10, help='Mean time to repair')
parser.add_argument('-rnd_mttr', '--random_MTTR', default=False, action='store_true', help='Use random MTTR')

# plotting settings
parser.add_argument('-draw', '--draw_gantt', default=5, action='store', type=int, help='Any value greater than 0 would plot the gantt chart, always no-show when simulation span is longer than 200')
parser.add_argument('-save_gantt', default=True, action='store_false', help='Save the gantt chart?')

# scheduler
parser.add_argument('-sqc', '--sqc_rule', default= SQC_rule.FIFO, type=lambda x:eval("SQC_rule."+str(x)), help='Sequencing rule or scheduler')

# threading
parser.add_argument('-multi_thread', default= False, action='store_true', help='Use this flag to create multiple threads/environments')
parser.add_argument('-thread_no', default= 4, type=int, help='Number of threads')


args = parser.parse_args()


if __name__ == '__main__':
    MainSimulator(m_no = args.m_no, span = args.span, E_utliz = args.E_utliz,
        pt_range = args.pt_range, due_tightness = args.due_tightness, processing_time_variability = args.processing_time_variability, pt_cv = args.pt_cv,
        machine_breakdown = args.machine_breakdown, MTBF = args.MTBF, MTTR = args.MTTR, random_MTBF = args.random_MTBF, random_MTTR = args.random_MTTR,
        draw_gantt = args.draw_gantt, save_gantt = args.save_gantt, multi_thread = args.multi_thread, thread_no = args.thread_no,
        sqc_rule = args.sqc_rule
        )