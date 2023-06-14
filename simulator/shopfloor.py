#!/usr/bin/env python3

import simpy
import pandas as pd
import logging.config
import time
import json
from pathlib import Path
import shutil
import argparse
from job import *
from machine import *
from sequencing_rule import *
from event_narrator import *
from gantt_chart import *


class Shopfloor:
    def __init__(self, **kwargs):
        # STEP 1. important features shared by all machine and job instances
        self.env = simpy.Environment()
        self.kwargs = kwargs
        with open(Path.cwd() / "config" / "logging_config.json") as f:
            log_config = json.load(f)
            logging.config.dictConfig(log_config)
            self.logger = logging.getLogger("sim_logger")
        self.recorder = Recorder(**kwargs) # recorder object shared by all other objects
        # STEP 2. create machines
        self.m_list = []
        self.logger.info("Simulation run at: {}".format(time.strftime("%Y-%m-%d, %H:%M:%S")))
        self.logger.debug("Creating {} machines on shopfloor ".format(kwargs['m_no']))
        for i in range(kwargs['m_no']):
            self.m_list.append(Machine(env = self.env, logger = self.logger, recorder = self.recorder, m_idx=i, **kwargs))
        # STEP 3: create the event narrator of dynamic events
        self.logger.debug("Initializing event narrator, machine breakdown: {}, processing time variability: {}".format(kwargs['machine_breakdown'], kwargs['processing_time_variability']))
        self.narrator = Narrator(env = self.env, logger = self.logger, recorder = self.recorder, m_list = self.m_list, **kwargs)

    
    def run_simulation(self):
        self.env.run(until=self.kwargs['span']+1000)
        self.narrator.post_simulation()
        # if the simulation completed without error and formal mode is activated, copy paste the log file to storage
        if "formal" in self.kwargs and self.kwargs['formal']:
            ct = ''.join([str(x) for x in time.strftime("%Y,%m,%d,%H,%M,%S").split(',')])
            shutil.copy(Path.cwd() / "log" / "sim.log", Path.cwd() / "log" / "past" / "{}_sim.log".format(ct))
        if "draw_gantt" in self.kwargs and self.kwargs['draw_gantt'] > 0:
            painter = Draw(self.recorder, **self.kwargs)




if __name__ == '__main__':
    spf = Shopfloor(m_no = 5, span = 80, pt_range = [1,10], due_tightness = 1.4, E_utliz = 0.9,
                    machine_breakdown = False, MTBF = 100, MTTR = 10, random_bkd = True,
                    processing_time_variability = False, pt_cv = 0.1,
                    draw_gantt = 5, save_gantt = True, seed = 10908,
                    sqc_rule = SQC_rule.opt_scheduler
                    )
    spf.run_simulation()