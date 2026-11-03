"""
Module containing classes for initializing simulation entities, driving dynamic/stochastic
events (job arrivals, machine breakdowns, processing time variability), and recording
simulation metrics (`Narrator` and `Recorder`).
"""

# standard imports
from logging import Logger
import time
from typing import Dict, List

import numpy as np
from simpy import Environment
from tabulate import tabulate

# project modules
from .job import Job
from .machine import Machine
from ..scheduler.sequencing_rule import SequencingMethod
from ..scheduler.scheduler import CentralScheduler


class Narrator:
    """
    Event narrator that initializes shopfloor entities (machines, centralized scheduler)
    and spawns SimPy processes for dynamic events such as Poisson job arrivals and machine breakdowns.
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kwargs = kwargs

        # Declare core component types
        self.logger: Logger
        self.env: Environment
        self.m_list: List[Machine]
        self.recorder: Recorder
        self.logger.debug("Event narrator created")

        # Specify or generate random seed for reproducible stochastic events
        if ('seed' in kwargs) and (kwargs['seed'] != 0):
            self.logger.debug("Random seed is specified, seed: {}".format(kwargs['seed']))
        else:
            self.seed = int(np.random.randint(0, int(1e10)))
            self.logger.warning(
                "Random seed is not specified, use seed: {}, do this only in test or training!".format(self.seed)
            )
        self.rng = np.random.default_rng(seed=self.seed)

        """
        1.1 Core components: machines and dynamic Poisson job arrivals
        """
        self.m_no = len(self.m_list)  # number of machines (equals number of operations per job)
        exp_pt = np.average(self.pt_range)  # expected processing time of an individual operation
        self.j_idx = 0  # running job index, starting from 0

        # Model new job arrivals as a Poisson process by drawing inter-arrival intervals
        # from an Exponential distribution with mean beta = 1 / lambda = E[PT] / E[utilization]
        beta = exp_pt / self.E_utliz
        self.logger.debug(
            "The expected utilization rate (excluding machine down time) is: [{}%]".format(self.E_utliz * 100)
        )
        self.logger.debug(
            "Converted expected interval between job arrival is: [{}] (m_no: [{}], pt_range: {}, exp_pt: [{}])".format(
                beta, self.m_no, self.pt_range, exp_pt
            )
        )

        # Total number of jobs arriving within the nominal simulation span
        self.total_no = int(np.round(self.span / beta))
        # Pre-sample inter-arrival intervals from the Exponential distribution (rounded to integer time steps)
        self.arrival_interval = self.rng.exponential(beta, self.total_no).round()
        # Register the job creation SimPy process
        self.env.process(self.process_job_creation())

        """
        1.2 Machine initialization: link machines and configure sequencing strategy / central scheduler
        """
        self.opt_mode = False
        self.central_scheduler = None
        if 'sqc_method' in kwargs:
            if kwargs['sqc_method'] == 'complete_schedule':  # placeholder for following a pre-computed static schedule
                job_sequencing_func = SequencingMethod.FIFO
            elif kwargs['sqc_method'] in (SequencingMethod.GurobiOptimizer, SequencingMethod.ORTools):
                # Mathematical optimization mode: create a CentralScheduler to dynamically compute global schedules
                self.central_scheduler = CentralScheduler(**self.kwargs)
                self.opt_mode = True
                job_sequencing_func = self.central_scheduler.draw_from_schedule
                self.logger.info(
                    f"Optimization mode is ON, A [centralized {self.sqc_method.__name__} scheduler] is created, "
                    f"all machines use a central schedule"
                )
            else:
                # Reactive heuristic sequencing rule (e.g., FIFO, LIFO, Slack, CR) or DRL agent
                job_sequencing_func = kwargs['sqc_method']
                self.logger.info(f"Machine use [{job_sequencing_func.__name__}] sequencing rule")
        else:
            # Default fallback sequencing rule is FIFO
            self.logger.info("* All machines use default [FIFO] rule for job sequencing")
            job_sequencing_func = SequencingMethod.FIFO

        """
        2. Optional event I: machine breakdown
        """
        if self.machine_breakdown:
            for m in self.m_list:
                self.env.process(self.process_machine_breakdown(m, self.random_MTBF, self.random_MTTR))
            self.logger.info(
                "Machine breakdown mode is ON, MTBF: [{}], MTTR: [{}]".format(self.MTBF, self.MTTR)
            )
        else:
            self.logger.debug("Machine breakdown is disabled.")

        # Initialize all Machine instances with the complete machine list, sequencing function, and scheduler reference
        for m in self.m_list:
            m.initialization(
                machine_list=self.m_list,
                sqc_method=job_sequencing_func,
                central_scheduler=self.central_scheduler,
            )

        """
        3. Optional event II: processing time variability
        """
        if kwargs.get('processing_time_variability', False) and kwargs.get('pt_cv', 0) > 0:
            self.pt_cv = kwargs['pt_cv']
            self.logger.debug(
                "Variable processing time mode is ON, coefficient of variance: [{}]".format(kwargs['pt_cv'])
            )
        else:
            self.pt_cv = 0

    def process_job_creation(self):
        """SimPy process that continuously generates new jobs according to pre-sampled arrival intervals."""
        self.program_start_T = time.time()
        # Each job visits all machines in the shopfloor once in a randomly shuffled order
        trajectory_seed = np.arange(self.m_no)

        while self.j_idx < self.total_no:
            # Wait for the inter-arrival interval before creating the next job
            job_arrival_interval = self.arrival_interval[self.j_idx]
            yield self.env.timeout(job_arrival_interval)

            # Generate a random routing trajectory by shuffling machine indices
            self.rng.shuffle(trajectory_seed)
            # Sample nominal (expected) integer processing times for each machine in `pt_range`
            ptl = self.rng.integers(low=self.pt_range[0], high=self.pt_range[1] + 1, size=[self.m_no])

            # Instantiate the new Job
            job_instance = Job(
                env=self.env,
                logger=self.logger,
                recorder=self.recorder,
                rng=self.rng,
                j_idx=self.j_idx,
                trajectory=trajectory_seed.copy(),
                pt_by_m_idx=ptl.copy(),
                pt_range=self.pt_range,
                pt_cv=self.pt_cv,
                due_tightness=self.due_tightness,
            )
            # Register this active job in the shared recorder
            self.recorder.in_system_jobs[self.j_idx] = job_instance

            # Trigger the central scheduler to rebuild the schedule if optimization mode is active
            if self.opt_mode:
                if not self.central_scheduler.build_schedule_event.triggered:
                    self.central_scheduler.build_schedule_event.succeed()
                    self.logger.debug("New job arrived, call central scheduler to build schedule\n" + "-" * 88)

            # Route the newly created job to the first machine in its trajectory
            first_m = int(trajectory_seed[0])
            self.m_list[first_m].job_arrival(job_instance)

            # Increment the running job counter
            self.j_idx += 1
            # Yield a zero-time step to allow arrival events and scheduling to be processed
            yield self.env.timeout(0)

    def process_machine_breakdown(self, m_instance: Machine, random_MTBF: bool, random_MTTR: bool):
        """
        SimPy process that periodically disables a machine to simulate breakdowns and repairs.
        Breakdowns are non-preemptive: if a machine is currently processing an operation when a failure
        occurs, the breakdown commences immediately after the current operation completes.
        """
        while self.env.now < self.span:
            # Draw the time interval until the next failure (MTBF) and the repair duration (MTTR)
            if random_MTBF:
                mtbf_interval = float(np.around(self.rng.exponential(self.MTBF), decimals=1))
            else:
                mtbf_interval = float(self.MTBF)

            if random_MTTR:
                bkd_t = float(np.around(self.rng.uniform(self.MTTR * 0.5, self.MTTR * 1.5), decimals=1))
            else:
                bkd_t = float(self.MTTR)

            # Wait until the next failure event is due
            yield self.env.timeout(mtbf_interval)

            # Non-preemptive breakdown: if the machine is currently busy, wait until its current operation finishes
            while m_instance.status == "processing" or m_instance.hidden_release_T > self.env.now:
                wait_t = max(m_instance.hidden_release_T - self.env.now, 0.0)
                yield self.env.timeout(wait_t)

            actual_begin = round(self.env.now, 1)
            actual_end = round(actual_begin + bkd_t, 1)
            expected_end = round(actual_begin + self.MTTR, 1)

            # Switch off the machine and update its expected/actual release times
            m_instance.working_event = self.env.event()
            m_instance.release_T = self.env.now + self.MTTR
            m_instance.hidden_release_T = self.env.now + bkd_t
            m_instance.status = "down"
            self.logger.info(
                f"{self.env.now} > BKD start: Machine {m_instance.m_idx} will be down for {bkd_t}, till {expected_end}"
                + (". Invoke central scheduler to rebuild the schedule" if self.opt_mode else "")
            )

            # Trigger central scheduler to rebuild the schedule around the newly unavailable machine
            if self.opt_mode:
                if not self.central_scheduler.build_schedule_event.triggered:
                    self.central_scheduler.build_schedule_event.succeed()

            # Wait for the actual repair duration to elapse
            yield self.env.timeout(bkd_t)

            # Record breakdown interval and restore the machine to operational status
            self.recorder.m_bkd_dict[m_instance.m_idx].append([actual_begin, actual_end])
            m_instance.release_T = self.env.now
            m_instance.hidden_release_T = self.env.now
            if m_instance.status == "down":
                m_instance.status = "idle"
            m_instance.working_event.succeed()

    def post_simulation(self) -> None:
        """
        Verify simulation completion, check for schedule-execution mismatches (under deterministic settings),
        and print summary tables of configuration and performance metrics (tardiness and flowtime).
        """
        # 1. Verify that all created jobs have completed all their operations
        if len(self.recorder.j_operation_dict) != self.j_idx:
            msg = "Simulation FAILED, not all jobs have successfully completed their operations"
            self.logger.error(msg)

        # 2. Save any over-extended optimization problem instances (>1s solve time) to JSON
        if self.opt_mode and self.central_scheduler.ext_prob_log:
            self.central_scheduler.post_simulation()

        # 3. Compare scheduled operation start times vs. actual execution start times.
        # Only meaningful when processing time variability and repair time variability are both disabled.
        if self.opt_mode and self.pt_cv == 0 and not (self.machine_breakdown and self.random_MTTR):
            mismatch = {}
            for j_idx, ops in self.central_scheduler.j_op_by_schedule.items():
                if j_idx not in self.recorder.j_operation_dict:
                    continue
                compare = zip(ops, self.recorder.j_operation_dict[j_idx][-len(ops):])
                for expected, actual in compare:  # (m_idx, opBeginT) vs [m_idx, begin_T, actual_pt, wait_T]
                    if round(expected[1], 1) != round(actual[1], 1):
                        mismatch.setdefault(j_idx, []).append(
                            "M{}, ET:{}, AT:{}".format(expected[0], expected[1], actual[1])
                        )
            if mismatch:
                self.logger.warning(
                    "Schedule and execution MISMATCH:\n{}\n".format(
                        tabulate(
                            [
                                ["J.idx", "Mismatch"],
                                *[[j_idx, description] for j_idx, description in mismatch.items()],
                            ],
                            headers="firstrow",
                            tablefmt="psql",
                        )
                    )
                )

        # 4. Format simulation configuration summary table
        header = ["Category", "Number", "Details"]
        if self.machine_breakdown:
            m_config = ["Machine", self.m_no, f"Machine Breakdown: {self.machine_breakdown}"]
            if self.random_MTBF:
                m_config[-1] += f"\nMTBF: {self.MTBF}, random: {self.random_MTBF}"
            else:
                m_config[-1] += f"\nMTBF: {self.MTBF}, deterministic"
            if self.random_MTTR:
                m_config[-1] += f"\nMTTR: {self.MTTR}, random: {self.random_MTTR}"
            else:
                m_config[-1] += f"\nMTTR: {self.MTTR}, deterministic"
        else:
            m_config = ["Machine", self.m_no, "Machine Breakdown: False"]

        if self.processing_time_variability and self.pt_cv > 0:
            j_config = [
                "Job",
                self.j_idx,
                f"pt range: {self.pt_range}\npt cv: {self.pt_cv}\ndue tightness: {self.due_tightness}",
            ]
        else:
            j_config = [
                "Job",
                self.j_idx,
                f"pt range: {self.pt_range}, deterministic\ndue tightness: {self.due_tightness}",
            ]

        sqc_name = getattr(self.sqc_method, '__name__', str(self.sqc_method))
        sqc_config = ['Sqc', "N.A.", sqc_name]
        if self.opt_mode:
            sqc_config[-1] += "\nTotal: {}, Strategic idleness: {}".format(
                self.recorder.sqc_cnt_opt, self.recorder.sqc_cnt_SI
            )
        else:
            sqc_config[-1] += "\nReactive: {}, Passive: {}".format(
                self.recorder.sqc_cnt_reactive, self.recorder.sqc_cnt_passive
            )

        last_comp_t = max(self.recorder.last_job_comp_T, 1e-6)
        avg_cum_m_run_t = sum(m.cumulative_runtime for m in self.m_list) / self.m_no / last_comp_t
        sim_config = [
            "Sim",
            "N.A.",
            "Span: {}, Actual: {}\nUtilization: Expected: {}%, Actual: {}%\nRandom seed: {} / {}".format(
                self.span,
                self.recorder.last_job_comp_T,
                self.E_utliz * 100,
                round(avg_cum_m_run_t * 100, 2),
                self.seed,
                self.rng,
            ),
        ]
        wall_time = max(time.time() - self.program_start_T, 1e-6)
        if self.opt_mode:
            opt_tt = self.recorder.opt_time_expense
            sim_config[-1] += "\nWall Time: {}s, Opt.: {}s, {}%".format(
                round(wall_time, 2), round(opt_tt, 2), round(100 * (opt_tt / wall_time), 1)
            )
        else:
            sim_config[-1] += "\nWall Time: {}s".format(round(wall_time, 2))

        self.logger.info(
            'Problem/Simulation Configurations:\n{}\n'.format(
                tabulate([header, m_config, j_config, sqc_config, sim_config], headers="firstrow", tablefmt="grid")
            )
        )

        # 5. Format and log performance metrics (Tardiness and Flowtime)
        if self.recorder.j_tardiness_dict and self.j_idx > 0:
            cum_tard = sum(self.recorder.j_tardiness_dict.values())
            max_tard = max(self.recorder.j_tardiness_dict.values())
            cum_flow = sum(self.recorder.j_flowtime_dict.values())
            max_flow = max(self.recorder.j_flowtime_dict.values())
            self.logger.info(
                'Performance:\n{}\n'.format(
                    tabulate(
                        [
                            ["Category", "value"],
                            ["Tardiness", "max: {}, mean: {}".format(round(max_tard, 2), round(cum_tard / self.j_idx, 2))],
                            ["Flowtime", "max: {}, mean: {}".format(round(max_flow, 2), round(cum_flow / self.j_idx, 2))],
                        ],
                        headers="firstrow",
                        tablefmt="grid",
                    )
                )
            )

    def build_sqc_experience_repository(self, m_list: List[Machine]) -> None:
        """Initialize per-machine experience dictionaries for DRL sequencing training."""
        self.incomplete_experience = {}
        self.rep_memo = {}
        for m in m_list:
            self.incomplete_experience[m.m_idx] = {}  # stores partial transitions [s_0, a_0] at decision point
            self.rep_memo[m.m_idx] = []  # stores completed transitions [s_0, a_0, s_1, r_0]

    def complete_experience(self, m_idx: int, decision_point: float, r_t: float) -> None:
        """Complete a partial experience tuple by appending reward `r_t` and moving it to `rep_memo`."""
        self.incomplete_experience[m_idx][decision_point] += [r_t]
        complete_exp = self.incomplete_experience[m_idx].pop(decision_point)
        self.rep_memo[m_idx].append(complete_exp)
        self.reward_record[m_idx][0].append(self.env.now)
        self.reward_record[m_idx][1].append(r_t)


class Recorder:
    """Shared data container that tracks active jobs, completed operations, breakdowns, and KPIs."""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # Cumulative wall-clock time spent in mathematical optimization solvers
        self.opt_time_expense: float = 0.0
        self.last_job_comp_T: float = 0.0
        # Counters for sequencing decision types
        self.sqc_cnt_opt = self.sqc_cnt_SI = self.sqc_cnt_reactive = self.sqc_cnt_passive = 0
        # Active and completed job records
        self.in_system_jobs: Dict[int, Job] = {}
        self.j_operation_dict = {}
        self.j_tardiness_dict = {}
        self.j_flowtime_dict = {}
        self.m_bkd_dict = {idx: [] for idx in range(kwargs['m_no'])}
        self.m_cum_runtime_dict = {}
        self.pt_mean_dict = {}
        self.pt_std_dict = {}
        self.expected_tardiness_dict = {}
        # Aggregate performance metric
        self.cumulative_tardiness: float = 0.0

