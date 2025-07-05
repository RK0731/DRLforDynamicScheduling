#!/usr/bin/env python3

# standard imports
import logging.config
import multiprocessing as mp
import pandas as pd
from pathlib import Path
import simpy
import time
import traceback

# Project modules
from .event import *
from .exc import *
from .job import *
from .machine import *
from ..scheduler.sequencing_rule import SequencingMethod
from ..utilities import create_logger, setup_logger, draw_gantt_chart


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
        self.logger = setup_logger(stream=kwargs['stream'])
        # create the recorder object that shared by all other objects
        self.recorder = Recorder(**kwargs) 
        # STEP 2. create machines
        self.m_list = []
        self.logger.debug(f"Creating {kwargs['m_no']} machines on shopfloor ")
        for i in range(kwargs['m_no']):
            self.m_list.append(Machine(env = self.env, logger = self.logger, recorder = self.recorder, m_idx = i, **kwargs))
        # STEP 3. create the event narrator of dynamic events
        self.logger.debug(f"Initializing event narrator, machine breakdown: {kwargs['machine_breakdown']}, processing time variability: {kwargs['processing_time_variability']}")
        self.narrator = Narrator(env = self.env, logger = self.logger, recorder = self.recorder, m_list = self.m_list, **kwargs)

    
    def run_simulation(self):
        try:
            self.verify_simulation_setting()
            _start_T = time.time()
            self.logger.info("Simulation starts at: {}".format(time.strftime("%Y-%m-%d, %H:%M:%S")))
            self.env.run(until=self.kwargs['span']+1000)
            self.logger.info("Simulation elapsed after {}s".format(round(time.time()-_start_T,5)))
            self.narrator.post_simulation()
            # whether to plot the gantt chart
            if "draw_gantt" in self.kwargs and self.kwargs['draw_gantt'] > 0:
                draw_gantt_chart(self.logger, self.recorder, **self.kwargs)
        except Exception as e:
            self.logger.error(f"Simulation failed due to following exception:\n{str(traceback.format_exc())}")

    
    def verify_simulation_setting(self):
        # check for clash between randomness and use of optimization
        occ_variability = self.kwargs['random_MTTR'] or self.kwargs['processing_time_variability']
        if occ_variability and (self.kwargs['sqc_method'] == SequencingMethod.opt_scheduler):
            Input = input("WARNING: Machine occupation time variance enabled when using optimization algorithm-based scheduler! Processing time variance: {}, Random MTTR: {}.\nDo you still want to proceed? [Y/N]: ".format(
                self.kwargs['processing_time_variability'], self.kwargs['random_MTTR']))
            if Input != "Y":
                exit()