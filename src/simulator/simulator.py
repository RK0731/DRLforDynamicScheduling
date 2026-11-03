#!/usr/bin/env python3
"""
High-level simulation runner and Shopfloor container class.
Coordinates SimPy discrete-event environment, machines, event narrator, recorder, and visualization.
"""

# standard imports
import multiprocessing as mp
import time
import traceback

import simpy

# Project modules
from .event import Narrator, Recorder
from .machine import Machine
from ..scheduler.sequencing_rule import SequencingMethod
from ..utilities import setup_logger, draw_gantt_chart


class Simulator:
    """Single-process entry point to instantiate a Shopfloor and run one simulation episode."""

    @classmethod
    def run(cls, **kwargs) -> None:
        # create the shopfloor instance
        spf = Shopfloor(**kwargs)
        # run the simulation
        spf.run_simulation()


class SimulatorMultiThread:
    """Multi-process runner to execute multiple Shopfloor simulation instances in parallel."""

    def __init__(self, **kwargs) -> None:
        if not kwargs.get('multi_thread', False):
            spf = Shopfloor(**kwargs)
            spf.run_simulation()
        else:
            # disable interactive plotting in parallel execution mode
            kwargs['draw_gantt'] = 0
            spf_dict = {}
            sim_thread_dict = {}
            for idx in range(1, kwargs['thread_no'] + 1):
                spf_dict[idx] = Shopfloor(thread_idx=idx, **kwargs)
                sim_thread_dict[idx] = mp.Process(target=spf_dict[idx].run_simulation)
                sim_thread_dict[idx].start()
            for proc in sim_thread_dict.values():
                proc.join()


class Shopfloor:
    """
    Encapsulates a complete job shop simulation environment, including:
    1. SimPy discrete-event Environment, Logger, and shared Recorder;
    2. Workstations (Machine instances);
    3. Dynamic event generator and centralized scheduler coordinator (Narrator).
    """

    def __init__(self, **kwargs):
        # STEP 1. Initialize core objects shared across all machine and job instances
        self.env = simpy.Environment()
        self.kwargs = kwargs
        # initialize the logger
        self.logger = setup_logger(stream=kwargs.get('stream', True))
        # create the recorder object shared by all simulation entities
        self.recorder = Recorder(**kwargs)

        # STEP 2. Create machine (workstation) instances on the shopfloor
        self.m_list = []
        self.logger.debug(f"Creating {kwargs['m_no']} machines on shopfloor")
        for i in range(kwargs['m_no']):
            self.m_list.append(
                Machine(env=self.env, logger=self.logger, recorder=self.recorder, m_idx=i, **kwargs)
            )

        # STEP 3. Create the event narrator to drive dynamic job arrivals, breakdowns, and scheduling
        self.logger.debug(
            f"Initializing event narrator, machine breakdown: {kwargs['machine_breakdown']}, "
            f"processing time variability: {kwargs['processing_time_variability']}"
        )
        self.narrator = Narrator(
            env=self.env, logger=self.logger, recorder=self.recorder, m_list=self.m_list, **kwargs
        )

    def run_simulation(self) -> None:
        """Execute the SimPy environment, log post-simulation statistics, and generate the Gantt chart."""
        try:
            self.verify_simulation_setting()
            start_t = time.time()
            self.logger.info("Simulation starts at: {}".format(time.strftime("%Y-%m-%d, %H:%M:%S")))
            # Run beyond nominal span (+1000 buffer) so all jobs created within `span` can finish processing
            self.env.run(until=self.kwargs['span'] + 1000)
            self.logger.info("Simulation elapsed after {}s".format(round(time.time() - start_t, 5)))
            self.narrator.post_simulation()
            # Plot and/or save the Gantt chart if enabled
            if self.kwargs.get('draw_gantt', 0) > 0 or self.kwargs.get('save_gantt', False):
                draw_gantt_chart(self.logger, self.recorder, **self.kwargs)
        except Exception:
            self.logger.error(f"Simulation failed due to following exception:\n{traceback.format_exc()}")

    def verify_simulation_setting(self) -> None:
        """
        Check for potential incompatibility between stochastic occupation times (random MTTR or
        variable processing times) and deterministic mathematical-optimization schedulers.
        """
        occ_variability = self.kwargs.get('random_MTTR', False) or self.kwargs.get('processing_time_variability', False)
        is_opt_scheduler = self.kwargs.get('sqc_method') in (
            SequencingMethod.GurobiOptimizer,
            SequencingMethod.ORTools,
        )
        if occ_variability and is_opt_scheduler:
            user_input = input(
                "WARNING: Machine occupation time variance enabled when using optimization algorithm-based scheduler! "
                "Processing time variance: {}, Random MTTR: {}.\n"
                "Do you still want to proceed? [Y/N]: ".format(
                    self.kwargs['processing_time_variability'], self.kwargs['random_MTTR']
                )
            )
            if user_input.strip().upper() != "Y":
                exit()
