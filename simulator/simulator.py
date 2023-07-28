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
from .job import *
from .machine import *
from .sequencing_rule import *
from .event_narrator import *
from .gantt_chart import *


class MainSimulator:
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
            for idx in range(1, kwargs['thread_no'] + 1):
                spf_dict[idx] = Shopfloor(thread_idx = idx, **kwargs)
                sim_thread_dict[idx] = threading.Thread(target=spf_dict[idx].run_simulation)
                sim_thread_dict[idx].start()


class MainSimulatorProcessing:
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
        # set logger
        if "thread_idx" in kwargs:
            self.logger = logging.getLogger("logger_"+str(kwargs['thread_idx']))
            formatter = logging.Formatter("[T{}][%(module)+13s: %(lineno)-3d] %(levelname)-5s => %(message)s".format(kwargs['thread_idx']))
            _fpath = Path("./log/multi_thread")
            if not _fpath.exists(): _fpath.mkdir() 
            filehandler = logging.FileHandler(_fpath / "sim_log_{}.log".format(kwargs['thread_idx']), 'w')
            filehandler.setFormatter(formatter)
            self.logger.addHandler(filehandler)
            self.logger.setLevel(logging.DEBUG)
        else:
            with open(Path.cwd() / "config" / "logger_config.json") as f:
                log_config = json.load(f)
                logging.config.dictConfig(log_config)
                self.logger = logging.getLogger("sim_logger")
        self.recorder = Recorder(**kwargs) # recorder object shared by all other objects
        # STEP 2. create machines
        self.m_list = []
        self.logger.debug("Creating {} machines on shopfloor ".format(kwargs['m_no']))
        for i in range(kwargs['m_no']):
            self.m_list.append(Machine(env = self.env, logger = self.logger, recorder = self.recorder, m_idx=i, **kwargs))
        # STEP 3: create the event narrator of dynamic events
        self.logger.debug("Initializing event narrator, machine breakdown: {}, processing time variability: {}".format(kwargs['machine_breakdown'], kwargs['processing_time_variability']))
        self.narrator = Narrator(env = self.env, logger = self.logger, recorder = self.recorder, m_list = self.m_list, **kwargs)

    
    def run_simulation(self):
        self.check_settings()
        _start_T = time.time()
        self.logger.info("Simulation starts at: {}".format(time.strftime("%Y-%m-%d, %H:%M:%S")))
        self.env.run(until=self.kwargs['span']+1000)
        self.logger.info("Program elapsed after {}s".format(round(time.time()-_start_T,5)))
        self.narrator.post_simulation()
        # if the simulation completed without error and "keep" mode is activated, copy the log file to storage
        if "keep" in self.kwargs and self.kwargs['keep']:
            ct = ''.join([str(x) for x in time.strftime("%Y,%m,%d,%H,%M,%S").split(',')])
            shutil.copy(Path.cwd() / "log" / "sim.log", Path.cwd() / "log" / "past" / "{}_sim.log".format(ct))
        if "draw_gantt" in self.kwargs and self.kwargs['draw_gantt'] > 0:
            painter = Draw(self.recorder, **self.kwargs)
        return

    
    def check_settings(self):
        # check for clash betwwen randomness and use of optimization
        occ_variability = self.kwargs['random_MTTR'] or self.kwargs['processing_time_variability']
        if occ_variability and self.kwargs['sqc_rule'] == SQC_rule.opt_scheduler:
            Input = input("WARNING: Machine occupation time variance observed when using optimization algorithm-based scheduler! Processing time variance: {}, Random MTTR: {}.\nDo you still want to proceed? [Y/N]: ".format(
                self.kwargs['processing_time_variability'], self.kwargs['random_MTTR']))
            if Input != "Y":
                exit()