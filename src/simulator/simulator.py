#!/usr/bin/env python3
"""
High-level simulation runner and Shopfloor container class.
Coordinates SimPy discrete-event environment, machines, event narrator, recorder, and visualization.
"""

# standard imports
import multiprocessing as mp
import sys
import time
import traceback

import simpy

# Project modules
from .event import Narrator, Recorder
from .exc import SimulatorError
from .machine import Machine
from ..scheduler.scheduler import is_optimization_scheduler
from ..utilities import (
    SimulationDatabase,
    _prune_log_dirs,
    draw_gantt_chart,
    setup_logger,
)


def _run_shopfloor(kwargs: dict) -> None:
    """Module-level worker used by `SimulatorMultiThread`.

    The child process must construct the `Shopfloor` itself: a `Shopfloor` contains a
    SimPy environment with live process generators and is therefore not picklable on
    platforms that use the ``spawn`` start method (macOS, Windows).
    """
    Shopfloor(**kwargs).run_simulation()


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
            Shopfloor(**kwargs).run_simulation()
            return

        # Disable interactive plotting in parallel execution mode.
        kwargs['draw_gantt'] = 0
        thread_no = int(kwargs.get('thread_no', 1))
        base_seed = int(kwargs.get('seed', 0) or 0)

        # Create/migrate the database once in the parent. Letting every worker run
        # `PRAGMA journal_mode = WAL` concurrently can fail with `database is locked`.
        SimulationDatabase(db_path=kwargs.get('db_path'))
        # Only the parent prunes the shared log root.
        _prune_log_dirs(3)

        processes = []
        for idx in range(1, thread_no + 1):
            thread_kwargs = dict(kwargs)
            thread_kwargs['thread_idx'] = idx
            # Workers use the schema created by the parent; only the parent may prune logs.
            thread_kwargs['skip_db_init'] = True
            thread_kwargs['prune_logs'] = False
            # Derive a distinct seed per worker. Reusing the same non-zero seed would
            # run identical episodes and write duplicate runs to the database.
            if base_seed:
                thread_kwargs['seed'] = base_seed + idx
            proc = mp.Process(target=_run_shopfloor, args=(thread_kwargs,))
            proc.start()
            processes.append(proc)

        failed = []
        for idx, proc in enumerate(processes, start=1):
            proc.join()
            if proc.exitcode != 0:
                failed.append((idx, proc.exitcode))
        if failed:
            raise RuntimeError(
                "Simulation worker(s) failed: "
                + ", ".join(f"thread {idx} (exit {code})" for idx, code in failed)
            )


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
        # initialize the logger (children of a multi-process run must not prune shared log dirs)
        self.logger = setup_logger(
            stream=kwargs.get('stream', True),
            prune=kwargs.get('prune_logs', True),
        )
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
            f"Initializing event narrator, machine breakdown: {kwargs.get('machine_breakdown', True)}, "
            f"processing time variability: {kwargs.get('processing_time_variability', False)}"
        )
        self.narrator = Narrator(
            env=self.env, logger=self.logger, recorder=self.recorder, m_list=self.m_list, **kwargs
        )

    def run_simulation(self) -> None:
        """Execute the SimPy environment, log post-simulation statistics, and generate the Gantt chart."""
        run_id = None
        try:
            self.verify_simulation_setting()
            # Create the persistent `runs` row upfront so child records (e.g., streamed `sim_events`)
            # can reference a valid foreign key during the simulation
            run_id = self.recorder.start_run(
                self._build_run_config(), db_path=self.kwargs.get('db_path')
            )
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
            error = traceback.format_exc()
            self.logger.error(f"Simulation failed due to following exception:\n{error}")
            # Mark the already-created `runs` row as failed instead of leaving an orphaned
            # row with NULL KPIs. Then re-raise so callers/CI see a non-zero exit status.
            if self.recorder.run_id is not None:
                try:
                    self.recorder.fail_run(error)
                except Exception:
                    self.logger.error(
                        "Failed to mark run %s as failed:\n%s",
                        self.recorder.run_id,
                        traceback.format_exc(),
                    )
            raise

    def _build_run_config(self) -> dict:
        """
        Build the configuration-only record used to create the `runs` row upfront.
        Final summary KPIs are written separately by `Recorder.finalize_episode`.
        """
        kwargs = self.kwargs
        sqc_method = kwargs.get('sqc_method', 'FIFO')
        sqc_name = getattr(sqc_method, '__name__', str(sqc_method))
        machine_bkd = bool(kwargs.get('machine_breakdown', True))
        pt_range = kwargs.get('pt_range', [1, 10])
        return {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "seed": int(self.narrator.seed),
            "sqc_method": str(sqc_name),
            "m_no": int(kwargs.get('m_no', len(self.m_list))),
            "span": float(kwargs.get('span', 100)),
            "e_utliz": float(kwargs.get('E_utliz', 0.6)),
            "due_tightness": float(kwargs.get('due_tightness', 2.0)),
            "pt_min": float(pt_range[0]),
            "pt_max": float(pt_range[1]),
            "pt_cv": float(self.narrator.pt_cv),
            "machine_bkd": int(machine_bkd),
            "mtbf": float(kwargs.get('MTBF', 50.0)) if machine_bkd else None,
            "random_mtbf": int(bool(kwargs.get('random_MTBF', True))) if machine_bkd else None,
            "mttr": float(kwargs.get('MTTR', 10.0)) if machine_bkd else None,
            "random_mttr": int(bool(kwargs.get('random_MTTR', False))) if machine_bkd else None,
        }

    def verify_simulation_setting(self) -> None:
        """
        Check for potential incompatibility between stochastic occupation times (random MTTR or
        variable processing times) and deterministic mathematical-optimization schedulers.
        """
        occ_variability = self.kwargs.get('random_MTTR', False) or self.kwargs.get('processing_time_variability', False)
        is_opt_scheduler = is_optimization_scheduler(self.kwargs.get('sqc_method'))
        if occ_variability and is_opt_scheduler:
            ptv = bool(self.kwargs.get('processing_time_variability', False))
            rmttr = bool(self.kwargs.get('random_MTTR', False))
            warning = (
                "WARNING: Machine occupation time variance enabled when using optimization "
                "algorithm-based scheduler! Processing time variance: {}, Random MTTR: {}.\n".format(
                    ptv, rmttr
                )
            )
            # `force=True` (CLI `-y/--force`) is an explicit, unconditional opt-in: it must
            # never prompt, even when a TTY happens to be attached. This keeps automated
            # runs and tests deterministic.
            if self.kwargs.get('force', False):
                self.logger.warning(warning + "Proceeding because force=True.")
                return

            # Without `force`, `input()` would raise EOFError in CI / piped runs.
            if not sys.stdin.isatty():
                raise SimulatorError(
                    warning
                    + "Non-interactive stdin detected. Re-run with force=True "
                    "(CLI: -y/--force) to proceed."
                )
            user_input = input(warning + "Do you still want to proceed? [Y/N]: ")
            if user_input.strip().upper() != "Y":
                raise SimulatorError("Simulation aborted by user.")
