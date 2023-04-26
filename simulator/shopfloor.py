import simpy
import pandas as pd
import logging.config
import time
import json
from pathlib import Path
import shutil
import os
import argparse

from job import *
from machine import *
from event_narrator import *

class Shopfloor:
    def __init__(self, **kwargs):
        ''' STEP 1. important features shared by all machine and job instances'''
        self.env = simpy.Environment()
        with open(Path(__file__).parent / "config" / "logging_config.json") as f:
            log_config = json.load(f)
            logging.config.dictConfig(log_config)
            self.sim_logger = logging.getLogger("sim_logger")
        self.recorder = Recorder()
        ''' STEP 2. create machines and event narrator'''
        # machines
        self.m_list = []
        self.sim_logger.debug("Creating {} machines in shopfloor ".format(kwargs['m_no']))
        for i in range(kwargs['m_no']):
            self.m_list.append(Machine(self.env, self.sim_logger, self.recorder, m_idx=i, **kwargs))
        for m in self.m_list:
            m.initialization(machine_list = self.m_list)
        # dynamic events narrator
        self.sim_logger.debug("Initializing event narrator, machine breakdown: {}, processing time variability: {}".format(kwargs['machine_breakdown'], kwargs['processing_time_variability']))
        self.narrator = Narrator(self.env, self.sim_logger, self.recorder, machine_list=self.m_list, **kwargs)


    def run_simulation(self):
        self.env.run()
        # if the simulation completed without error, paste the log file to storage
        ct = ''.join([str(x) for x in time.strftime("%Y,%m,%d,%H,%M,%S").split(',')])
        shutil.copy(Path.cwd()/"log"/"sim.log", Path.cwd()/"log"/"past"/"{}_sim.log".format(ct))


if __name__ == '__main__':
    spf = Shopfloor(m_no=4, span=100, pt_range=[1,10], pt_variance=0, due_tightness=2, E_utliz=0.8,
                    machine_breakdown=False, processing_time_variability=False)
    spf.run_simulation()