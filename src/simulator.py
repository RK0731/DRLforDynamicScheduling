#!/usr/bin/env python3

import threading
import multiprocessing as mp
import simpy
import pandas as pd
import logging.config
import time
import json
from pathlib import Path
import shutil
import traceback

from .job import *
from .machine import *
from .sequencing_rule import SequencingMethod
from .event import *
from .exc import *
from .utilities import create_logger, draw_gantt_chart


class Simulator:
    @classmethod
    def run(cls, **kwargs) -> None:
        # create the shopfloor instance
        spf = Shopfloor(**kwargs)
        # run the simulation
        spf.run_simulation()


class SimulatorMultiThread:
    def __init__(self, **kwargs) -> None:
        if kwargs['multi_thread'] == False:
            spf = Shopfloor(**kwargs)
            spf.run_simulation()
        else:
            # disable plotting
            kwargs['draw_gantt'] = False
            # create multiple threads
            spf_dict = {}
            sim_thread_dict = {}
            spfPool = mp.Pool(kwargs['thread_no'])
            for idx in range(1, kwargs['thread_no'] + 1):
                spf_dict[idx] = Shopfloor(thread_idx = idx, **kwargs)
                sim_thread_dict[idx] = mp.Process(target=spf_dict[idx].run_simulation)
                sim_thread_dict[idx].start()



class Shopfloor:
    def __init__(self, **kwargs):
        # STEP 1. important features shared by all machine and job instances
        self.env = simpy.Environment()
        self.kwargs = kwargs
        # initialize the logger
        self.logger = create_logger()
        # create the recorder object that shared by all other objects
        self.recorder = Recorder(**kwargs) 
        # STEP 2. create machines
        self.m_list = []
        self.logger.debug("Creating {} machines on shopfloor ".format(kwargs['m_no']))
        for i in range(kwargs['m_no']):
            self.m_list.append(Machine(env = self.env, logger = self.logger, recorder = self.recorder, m_idx = i, **kwargs))
        # STEP 3: create the event narrator of dynamic events
        self.logger.debug("Initializing event narrator, machine breakdown: {}, processing time variability: {}".format(kwargs['machine_breakdown'], kwargs['processing_time_variability']))
        self.narrator = Narrator(env = self.env, logger = self.logger, recorder = self.recorder, m_list = self.m_list, **kwargs)

    
    def run_simulation(self):
        try:
            self.verify_simulation_setting()
            _start_T = time.time()
            self.logger.info("Simulation starts at: {}".format(time.strftime("%Y-%m-%d, %H:%M:%S")))
            self.env.run(until=self.kwargs['span']+1000)
            self.logger.info("Program elapsed after {}s".format(round(time.time()-_start_T,5)))
            self.narrator.post_simulation()
            # if the simulation completed without error and "keep" mode is activated, copy the log file to storage
            if "keep" in self.kwargs and self.kwargs['keep']:
                ct = ''.join([str(x) for x in time.strftime("%Y,%m,%d,%H,%M,%S").split(',')])
                shutil.copy(Path.cwd() / "log" / "sim.log", Path.cwd() / "log" / "past" / "{}_sim.log".format(ct))
            # whether to plot the gantt chart
            if "draw_gantt" in self.kwargs and self.kwargs['draw_gantt'] > 0:
                draw_gantt_chart(self.logger, self.recorder, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Simulation failed due to following exception:\n{str(traceback.format_exc())}")

    
    def verify_simulation_setting(self):
        # check for clash between randomness and use of optimization
        occ_variability = self.kwargs['random_MTTR'] or self.kwargs['processing_time_variability']
        if occ_variability and (self.kwargs['sqc_method'] == SequencingMethod.opt_scheduler):
            Input = input("WARNING: Machine occupation time variance observed when using optimization algorithm-based scheduler! Processing time variance: {}, Random MTTR: {}.\nDo you still want to proceed? [Y/N]: ".format(
                self.kwargs['processing_time_variability'], self.kwargs['random_MTTR']))
            if Input != "Y":
                exit()