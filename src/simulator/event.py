"""
Module containing classes for initializing simulation entities, driving dynamic/stochastic
events (job arrivals, machine breakdowns, processing time variability), and recording
simulation metrics (`Narrator` and `Recorder`).
"""

# standard imports
import json
from logging import Logger
import time
from typing import Dict, List, Optional

import numpy as np
from simpy import Environment
from tabulate import tabulate

# project modules
from .job import Job
from .machine import Machine
from ..scheduler.sequencing_rule import SequencingMethod
from ..scheduler.scheduler import CentralScheduler, is_optimization_scheduler
from ..utilities import SimulationDatabase


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
            elif is_optimization_scheduler(kwargs['sqc_method']):
                # Mathematical optimization mode: create a CentralScheduler to dynamically compute global schedules
                self.central_scheduler = CentralScheduler(**self.kwargs)
                self.opt_mode = True
                job_sequencing_func = self.central_scheduler.draw_from_schedule
                solver_name = getattr(self.sqc_method, '__name__', str(self.sqc_method))
                self.logger.info(
                    f"Optimization mode is ON, A [centralized {solver_name} scheduler] is created, "
                    f"all machines use a central schedule"
                )
            else:
                # Reactive heuristic sequencing rule (e.g., FIFO, LIFO, Slack, CR) or DRL agent.
                # Accept either a callable or a heuristic name string for API convenience.
                job_sequencing_func = kwargs['sqc_method']
                if isinstance(job_sequencing_func, str):
                    job_sequencing_func = getattr(SequencingMethod, job_sequencing_func)
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
                    self.central_scheduler.trigger_reschedule("JOB_ARRIVAL")
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
            bkd_start_msg = (
                f"{self.env.now} > BKD start: Machine {m_instance.m_idx} will be down for {bkd_t}, till {expected_end}"
                + (". Invoke central scheduler to rebuild the schedule" if self.opt_mode else "")
            )
            self.logger.info(bkd_start_msg)
            self.recorder.record_event(
                self.env.now, "INFO", "BKD_START", m_idx=m_instance.m_idx, message=bkd_start_msg
            )
            self.recorder.record_breakdown(
                m_idx=m_instance.m_idx,
                begin_t=actual_begin,
                end_t=actual_end,
                expected_end_t=expected_end,
                actual_duration=bkd_t,
            )

            # Trigger central scheduler to rebuild the schedule around the newly unavailable machine
            if self.opt_mode:
                if not self.central_scheduler.build_schedule_event.triggered:
                    self.central_scheduler.trigger_reschedule("MACHINE_BREAKDOWN")

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
        print summary tables of configuration and performance metrics, and persist the episode to SQLite.
        """
        # 1. Verify that all created jobs have completed all their operations
        if len(self.recorder.j_operation_dict) != self.j_idx:
            msg = "Simulation FAILED, not all jobs have successfully completed their operations"
            self.logger.error(msg)
            self.recorder.record_event(self.env.now, "ERROR", "SIM_FAILED", message=msg)

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
                mismatch_msg = "Schedule and execution MISMATCH:\n{}\n".format(
                    tabulate(
                        [
                            ["J.idx", "Mismatch"],
                            *[[j_idx, description] for j_idx, description in mismatch.items()],
                        ],
                        headers="firstrow",
                        tablefmt="psql",
                    )
                )
                self.logger.warning(mismatch_msg)
                self.recorder.record_event(self.env.now, "WARNING", "SCHEDULE_MISMATCH", message=mismatch_msg)

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
        actual_utliz = round(avg_cum_m_run_t, 4)
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
        mean_tard = max_tard = mean_flow = max_flow = None
        # Divide by the number of *completed* jobs, not by all created jobs: a run that is
        # cut off mid-operation would otherwise understate the mean KPIs.
        completed_tard = len(self.recorder.j_tardiness_dict)
        completed_flow = len(self.recorder.j_flowtime_dict)
        if completed_tard and self.j_idx > 0:
            cum_tard = sum(self.recorder.j_tardiness_dict.values())
            max_tard = round(float(max(self.recorder.j_tardiness_dict.values())), 2)
            mean_tard = round(float(cum_tard / completed_tard), 2)
            if completed_flow:
                cum_flow = sum(self.recorder.j_flowtime_dict.values())
                max_flow = round(float(max(self.recorder.j_flowtime_dict.values())), 2)
                mean_flow = round(float(cum_flow / completed_flow), 2)
            self.logger.info(
                'Performance:\n{}\n'.format(
                    tabulate(
                        [
                            ["Category", "value"],
                            ["Tardiness", "max: {}, mean: {}".format(max_tard, mean_tard)],
                            ["Flowtime", "max: {}, mean: {}".format(max_flow, mean_flow)],
                        ],
                        headers="firstrow",
                        tablefmt="grid",
                    )
                )
            )

        # 6. Finalize the episode: flush remaining buffered records and update the `runs` row KPIs
        kpi_record = {
            "actual_span": round(float(self.recorder.last_job_comp_T), 2),
            "actual_utliz": float(actual_utliz),
            "total_jobs": int(self.j_idx),
            "completed_jobs": int(len(self.recorder.j_operation_dict)),
            "sqc_cnt_opt": int(self.recorder.sqc_cnt_opt),
            "sqc_cnt_si": int(self.recorder.sqc_cnt_SI),
            "sqc_cnt_reactive": int(self.recorder.sqc_cnt_reactive),
            "sqc_cnt_passive": int(self.recorder.sqc_cnt_passive),
            "wall_time_s": round(float(wall_time), 5),
            "opt_time_s": round(float(self.recorder.opt_time_expense), 5),
            "mean_tardiness": mean_tard,
            "max_tardiness": max_tard,
            "mean_flowtime": mean_flow,
            "max_flowtime": max_flow,
        }
        run_id = self.recorder.finalize_episode(kpi_record)
        self.logger.info(f"Episode records persisted to SQLite database (run_id={run_id})")

    def build_sqc_experience_repository(self, m_list: List[Machine]) -> None:
        """Initialize per-machine experience dictionaries for DRL sequencing training."""
        # Single canonical buffer name shared with `Brain` and `Machine.complete_experience`:
        # `incomplete_rep_memo[m_idx][decision_t] = [s_t, a_t]` until the operation completes.
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        # Reward bookkeeping used by `complete_experience` / training diagnostics.
        self.sqc_reward_record = []
        self.reward_record = {m.m_idx: [[], []] for m in m_list}
        for m in m_list:
            self.incomplete_rep_memo[m.m_idx] = {}  # partial transitions [s_0, a_0] at decision point
            self.rep_memo[m.m_idx] = []  # completed transitions [s_0, a_0, s_1, r_0]
        # Backwards-compatible alias for older call sites.
        self.incomplete_experience = self.incomplete_rep_memo

    def complete_experience(self, m_idx: int, decision_point: float, r_t: float) -> None:
        """Complete a partial experience tuple by appending reward `r_t` and moving it to `rep_memo`."""
        self.incomplete_rep_memo[m_idx][decision_point] += [r_t]
        complete_exp = self.incomplete_rep_memo[m_idx].pop(decision_point)
        self.rep_memo[m_idx].append(complete_exp)
        self.reward_record[m_idx][0].append(self.env.now)
        self.reward_record[m_idx][1].append(r_t)


class Recorder:
    """
    Shared in-memory data container that tracks active jobs, completed operations, breakdowns,
    rescheduling cycles, chronological events, and KPIs during a simulation episode.

    Persistence strategy (bounded memory):
      - `sim_events` (the dominant table) is streamed to SQLite in batches once its in-memory
        buffer exceeds `FLUSH_EVENTS_THRESHOLD`, keeping RAM usage bounded for long simulation spans.
      - `jobs`, `operations`, `breakdowns`, and `schedule_cycles` are buffered in memory for the
        whole episode and flushed once at the end (they are much smaller per row than `sim_events`).
    """

    # Maximum number of buffered `sim_events` rows before an incremental flush to SQLite is triggered
    FLUSH_EVENTS_THRESHOLD = 10_000

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
        self.all_jobs_dict: Dict[int, Job] = {}
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

        # Structured buffers for SQLite batch persistence
        self.operations_list: list = []
        self.decisions_list: list = []
        self.breakdowns_list: list = []
        self.schedule_cycles_list: list = []
        self.sim_events_list: list = []

        # Database persistence state (initialized by `start_run` before the simulation begins)
        self.db: Optional[SimulationDatabase] = None
        self.run_id: Optional[int] = None

    def start_run(self, run_record: dict, db_path=None) -> int:
        """
        Instantiate the persistent `SimulationDatabase`, create the `runs` row upfront, and return
        the assigned `run_id`. Must be called once before the SimPy environment begins executing.
        """
        self.db = SimulationDatabase(
            db_path=db_path,
            init_schema=not getattr(self, "skip_db_init", False),
        )
        self.run_id = int(self.db.create_run(run_record))
        return self.run_id

    def record_event(
        self,
        sim_t: float,
        level: str,
        event_type: str,
        m_idx: Optional[int] = None,
        j_idx: Optional[int] = None,
        message: str = "",
    ) -> None:
        """
        Append a structured simulation event to the in-memory buffer for the `sim_events` table.
        When the buffer exceeds `FLUSH_EVENTS_THRESHOLD`, flush it incrementally to SQLite.
        """
        self.sim_events_list.append(
            (
                round(float(sim_t), 2),
                str(level),
                str(event_type),
                int(m_idx) if m_idx is not None else None,
                int(j_idx) if j_idx is not None else None,
                str(message),
            )
        )
        if len(self.sim_events_list) >= self.FLUSH_EVENTS_THRESHOLD:
            self._flush_events()

    def _flush_events(self) -> None:
        """Stream the buffered `sim_events` rows to SQLite and clear the in-memory buffer."""
        if not self.sim_events_list:
            return
        if self.db is None or self.run_id is None:
            # No run created yet (e.g., early event before `start_run`); defer to final flush
            return
        self.db.append_events(self.run_id, self.sim_events_list)
        self.sim_events_list.clear()

    def record_decision(
        self,
        sim_t: float,
        m_idx: int,
        j_idx: int,
        op_seq: int,
        decision_type: str,
        queue_job_indices: list,
        chosen_pos: int,
        candidates: dict,
    ) -> None:
        """
        Append an exact decision-replay snapshot to the in-memory buffer for the `decisions` table.

        Parameters
        ----------
        queue_job_indices : ordered `j_idx` list exactly as seen by the machine at decision time
                            (this is the authoritative queue snapshot for replay).
        chosen_pos        : index of the chosen job inside `queue_job_indices`.
        candidates        : mapping `{str(j_idx): {feature: value}}` for every queued job.
                            Queue order is authoritative in `queue_job_indices`.
        """
        self.decisions_list.append(
            (
                round(float(sim_t), 2),
                int(m_idx),
                int(j_idx),
                int(op_seq),
                str(decision_type),
                int(len(queue_job_indices)),
                int(chosen_pos),
                json.dumps([int(x) for x in queue_job_indices]),
                json.dumps(candidates),
                None,  # state_json      (reserved for DRL s_t)
                None,  # action          (reserved for DRL a_t)
                None,  # reward          (reserved for DRL r_t)
                None,  # next_state_json (reserved for DRL s_{t+1})
                None,  # episode_done    (reserved for DRL terminal flag)
            )
        )

    def record_breakdown(
        self,
        m_idx: int,
        begin_t: float,
        end_t: float,
        expected_end_t: float,
        actual_duration: float,
    ) -> None:
        """Append a machine breakdown record to the in-memory buffer for the `breakdowns` table."""
        self.breakdowns_list.append(
            [
                int(m_idx),
                round(float(begin_t), 2),
                round(float(end_t), 2),
                round(float(expected_end_t), 2),
                round(float(actual_duration), 2),
                0.0,  # prod_delay updated if a job waits on this breakdown in `Machine.process_breakdown`
            ]
        )

    def update_last_breakdown_delay(self, m_idx: int, prod_delay: float) -> None:
        """Update `prod_delay` on the most recent breakdown record for machine `m_idx`."""
        for bkd in reversed(self.breakdowns_list):
            if bkd[0] == int(m_idx):
                bkd[5] = round(float(prod_delay), 2)
                break

    def finalize_episode(self, kpi_record: dict) -> int:
        """
        Flush any remaining buffered data to SQLite at the end of the simulation:
          1. stream remaining `sim_events` (incremental flush),
          2. batch-insert `jobs`, `operations`, `decisions`, `breakdowns`, and `schedule_cycles`,
          3. update the `runs` row with final summary KPIs.
        Returns the episode's `run_id`.
        """
        if self.db is None or self.run_id is None:
            raise RuntimeError("Recorder.start_run() must be called before finalize_episode()")

        # 1. Flush any remaining buffered sim_events
        self._flush_events()

        # 2. Build and flush the smaller child tables
        jobs_records = []
        for j_idx in sorted(self.all_jobs_dict.keys()):
            job = self.all_jobs_dict[j_idx]
            flowtime = self.j_flowtime_dict.get(j_idx)
            tardiness = self.j_tardiness_dict.get(j_idx)
            jobs_records.append(
                (
                    int(job.j_idx),
                    round(float(job.creation_T), 2),
                    round(float(job.due), 2),
                    job.completion_T,
                    str(job.status),
                    round(float(tardiness), 2) if tardiness is not None else None,
                    round(float(flowtime), 2) if flowtime is not None else None,
                    json.dumps(job.initial_trajectory),
                    json.dumps(job.initial_exp_pt),
                    json.dumps(job.initial_actual_pt),
                )
            )
        # A job still processing when the episode ends has only an *expected* end time for
        # its last operation. Flag that row as incomplete so offline reconstruction does
        # not incorrectly treat the operation (and job) as finished.
        incomplete_last_ops = {
            int(job.j_idx): len(job.operation_record) - 1
            for job in self.all_jobs_dict.values()
            if job.completion_T is None
            and job.status == "processing"
            and job.operation_record
        }
        operations_records = []
        for row in self.operations_list:
            if incomplete_last_ops.get(int(row[0])) == int(row[1]):
                row = (*row[:-1], 0)
            operations_records.append(row)

        self.db.append_children(
            run_id=self.run_id,
            jobs_records=jobs_records,
            operations_records=operations_records,
            breakdowns_records=[tuple(b) for b in self.breakdowns_list],
            schedule_cycles_records=self.schedule_cycles_list,
            decisions_records=self.decisions_list,
        )

        # 3. Finalize the `runs` row with summary KPIs
        self.db.finalize_run(self.run_id, kpi_record)
        return int(self.run_id)

    def fail_run(self, error: str) -> None:
        """Mark this episode's `runs` row as failed (called from `Shopfloor.run_simulation`)."""
        if self.db is not None and self.run_id is not None:
            self.db.fail_run(self.run_id, error)

