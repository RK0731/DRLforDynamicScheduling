#!/usr/bin/env python3
"""
Author: Liu Renke
Utility module providing logging configuration and Gantt chart visualization for the simulator.
"""

from datetime import datetime as dt
import copy
from contextlib import closing
import json
import logging
from logging.config import dictConfig
import os
from pathlib import Path
import shutil
import sqlite3
import time
from typing import Optional

import numpy as np

# Root directory where all timestamped simulation log folders and the persistent SQLite database are stored.
# `SIM_LOG_ROOT` allows tests and batch jobs to isolate logs without touching the project's `logs/` folder.
LOG_ROOT_DIR = Path(os.environ.get("SIM_LOG_ROOT", str(Path.cwd() / 'logs')))
DEFAULT_DB_PATH = LOG_ROOT_DIR / 'simulation.db'
LOG_DIR = LOG_ROOT_DIR / dt.now().strftime("%Y%m%d-%H%M%S%f")

# Base logging dictionary configuration:
# - 'sim_log_file' records detailed DEBUG-level messages with timestamps, module names, and line numbers.
# - 'console' streams concise INFO-level messages to stdout.
# - 'root' is kept at WARNING so third-party libraries (e.g., matplotlib font_manager) do not flood sim.log.
LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": '%(asctime)s [%(module)s: %(lineno)-3d] %(levelname)-8s >>> %(message)s',
            "datefmt": "%H:%M:%S"
        },
        "brief": {
            "format": '%(levelname)-7s >>> %(message)s',
            "datefmt": "%H:%M:%S"
        },
    },
    "handlers": {
        "sim_log_file": {
            "class": "logging.FileHandler",
            "formatter": "verbose",
            "filename": str(LOG_DIR / "sim.log"),
            "mode": "a",
        },
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "brief",
            "level": "INFO",
        }
    },
    "root": {
        "handlers": ["sim_log_file"],
        "level": "WARNING",
    },
    "loggers": {
        "sim_logger": {
            "handlers": ["sim_log_file", "console"],
            "level": "DEBUG",
            "propagate": False
        }
    }
}


def _prune_log_dirs(keep: int = 3) -> None:
    """Delete older timestamped log directories, keeping the `keep` most recent.

    Non-directory files (such as `simulation.db`) are never touched. The operation is
    best-effort: concurrent multi-process runs may delete a folder between `iterdir()`
    and `stat()`, which must not crash the simulation.
    """
    if keep <= 0:
        return
    try:
        folders = [f for f in LOG_ROOT_DIR.iterdir() if f.is_dir()]
        folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        for folder_path in folders[keep:]:
            shutil.rmtree(folder_path, ignore_errors=True)
    except FileNotFoundError:
        # Another worker pruned a directory between listing and stat.
        pass


def setup_logger(stream: bool = True, keep: int = 3, prune: bool = True) -> logging.Logger:
    """
    Create a timestamped log directory, prune old log directories beyond `keep` (default 3),
    and initialize the 'sim_logger' logger instance.
    Note: Non-directory files in `LOG_ROOT_DIR` (such as `simulation.db`) are never deleted.

    `prune=False` is used by child processes of `SimulatorMultiThread`: only the parent
    may prune the shared log root, otherwise workers can delete each other's active logs.
    """
    # Generate a fresh timestamped folder for this simulation run
    current_log_dir = LOG_ROOT_DIR / dt.now().strftime("%Y%m%d-%H%M%S%f")
    current_log_dir.mkdir(parents=True, exist_ok=True)

    if prune:
        _prune_log_dirs(keep)

    # Deep-copy LOG_CONFIG so disabling stream handler does not permanently mutate the template dictionary
    config = copy.deepcopy(LOG_CONFIG)
    config['handlers']['sim_log_file']['filename'] = str(current_log_dir / "sim.log")
    if not stream and "console" in config['loggers']['sim_logger']['handlers']:
        config['loggers']['sim_logger']['handlers'].remove("console")

    dictConfig(config)
    return logging.getLogger("sim_logger")


def create_logger(log_dir: Path = Path('./log'), stream: bool = True, keep: int = 3) -> logging.Logger:
    """
    Alternative logger factory that loads configuration from `config/logger_config.json` if present,
    or falls back to `setup_logger`.
    """
    config_file = Path.cwd() / "config" / "logger_config.json"
    if not config_file.exists():
        return setup_logger(stream=stream, keep=keep)

    logger = logging.getLogger(__name__)
    current_t = dt.now().strftime("%Y%m%d-%H%M%S%f")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / current_t

    # Avoid folder name clash if multiple processes start within the same microsecond
    cnt = 1
    while log_path.exists():
        cnt += 1
        log_path = log_dir / f"{current_t}({cnt})"
    log_path.mkdir(parents=True, exist_ok=True)

    # Clear old logger handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
        hdlr.close()

    with open(config_file) as f:
        log_config = json.load(f)
        log_config['handlers']['root_file']['filename'] = str(log_path / 'simulation.log')
        if not stream and len(log_config['loggers']['sim_logger']['handlers']) > 1:
            log_config['loggers']['sim_logger']['handlers'].pop(1)
        dictConfig(log_config)
        logger = logging.getLogger("sim_logger")

    logger.info(f"Log path set to [{log_path}]")

    # Remove obsolete log directories exceeding `keep`
    folders = [f for f in log_dir.iterdir() if f.is_dir()]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for folder_path in folders[keep:]:
        shutil.rmtree(folder_path, ignore_errors=True)

    return logger


class SimulationDatabase:
    """
    Persistent SQLite database manager for storing structured simulation episodes.
    Maintains 7 relational tables:
      1. `runs`            - Episode configuration parameters and aggregate KPIs
      2. `jobs`            - Per-job arrival, due date, trajectory, processing times, and tardiness/flowtime
      3. `operations`      - Per-operation execution details, waiting time, and scheduled vs. actual start time
      4. `decisions`       - Exact per-decision replay snapshot: queue order, candidate features, chosen position,
                             baseline rule picks, and reserved DRL transition columns
      5. `breakdowns`      - Machine breakdown intervals, expected repair times, and production delays
      6. `schedule_cycles` - Centralized optimization rescheduling snapshots, solver status, and solve time
      7. `sim_events`      - Chronological discrete-event trace across machines, jobs, and scheduler
    All data for an episode is buffered in `Recorder` during `env.run()`. The dominant `sim_events`
    table is streamed to disk in batches once its in-memory buffer exceeds a threshold, while the
    smaller tables (`jobs`, `operations`, `breakdowns`, `schedule_cycles`) are flushed in a single
    transaction at the end of the episode via `append_children`. The `runs` row is created upfront
    (`create_run`) so streamed child rows can reference a valid foreign key.
    """

    SCHEMA_SQL = """
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS runs (
        run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at       TEXT    NOT NULL,
        status           TEXT    NOT NULL DEFAULT 'running',
        error            TEXT,
        seed             INTEGER NOT NULL,
        sqc_method       TEXT    NOT NULL,
        m_no             INTEGER NOT NULL,
        span             REAL    NOT NULL,
        actual_span      REAL,
        e_utliz          REAL    NOT NULL,
        actual_utliz     REAL,
        due_tightness    REAL    NOT NULL,
        pt_min           REAL    NOT NULL,
        pt_max           REAL    NOT NULL,
        pt_cv            REAL    NOT NULL,
        machine_bkd      INTEGER NOT NULL,
        mtbf             REAL,
        random_mtbf      INTEGER,
        mttr             REAL,
        random_mttr      INTEGER,
        total_jobs       INTEGER,
        completed_jobs   INTEGER,
        sqc_cnt_opt      INTEGER DEFAULT 0,
        sqc_cnt_si       INTEGER DEFAULT 0,
        sqc_cnt_reactive INTEGER DEFAULT 0,
        sqc_cnt_passive  INTEGER DEFAULT 0,
        wall_time_s      REAL,
        opt_time_s       REAL    DEFAULT 0.0,
        mean_tardiness   REAL,
        max_tardiness    REAL,
        mean_flowtime    REAL,
        max_flowtime     REAL
    );

    CREATE TABLE IF NOT EXISTS jobs (
        run_id           INTEGER NOT NULL,
        j_idx            INTEGER NOT NULL,
        creation_t       REAL    NOT NULL,
        due_t            REAL    NOT NULL,
        completion_t     REAL,
        status           TEXT    NOT NULL,
        tardiness        REAL,
        flowtime         REAL,
        trajectory_json  TEXT    NOT NULL,
        exp_pt_json      TEXT    NOT NULL,
        actual_pt_json   TEXT    NOT NULL,
        PRIMARY KEY (run_id, j_idx),
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS operations (
        run_id            INTEGER NOT NULL,
        j_idx             INTEGER NOT NULL,
        op_seq            INTEGER NOT NULL,
        m_idx             INTEGER NOT NULL,
        decision_type     TEXT    NOT NULL,
        queue_len_at_dec  INTEGER,
        arrival_t         REAL    NOT NULL,
        begin_t           REAL    NOT NULL,
        end_t             REAL    NOT NULL,
        wait_t            REAL    NOT NULL,
        expected_pt       REAL    NOT NULL,
        actual_pt         REAL    NOT NULL,
        scheduled_begin_t REAL,
        completed         INTEGER NOT NULL DEFAULT 1,
        PRIMARY KEY (run_id, j_idx, op_seq),
        FOREIGN KEY (run_id, j_idx) REFERENCES jobs(run_id, j_idx) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS decisions (
        decision_id      INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL,
        sim_t            REAL    NOT NULL,
        m_idx            INTEGER NOT NULL,
        j_idx            INTEGER NOT NULL,
        op_seq           INTEGER NOT NULL,
        decision_type    TEXT    NOT NULL,
        queue_len        INTEGER NOT NULL,
        chosen_pos       INTEGER NOT NULL,
        queue_json       TEXT    NOT NULL,
        candidates_json  TEXT    NOT NULL,
        state_json       TEXT,
        action           INTEGER,
        reward           REAL,
        next_state_json  TEXT,
        episode_done     INTEGER,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS breakdowns (
        bkd_id           INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL,
        m_idx            INTEGER NOT NULL,
        begin_t          REAL    NOT NULL,
        end_t            REAL    NOT NULL,
        expected_end_t   REAL    NOT NULL,
        actual_duration  REAL    NOT NULL,
        prod_delay       REAL    DEFAULT 0.0,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS schedule_cycles (
        cycle_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL,
        sim_t            REAL    NOT NULL,
        trigger_type     TEXT,
        num_jobs         INTEGER NOT NULL,
        num_ops          INTEGER NOT NULL,
        solver_status    TEXT    NOT NULL,
        solve_time_s     REAL    NOT NULL,
        m_schedule_json  TEXT    NOT NULL,
        j_schedule_json  TEXT    NOT NULL,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS sim_events (
        event_id         INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id           INTEGER NOT NULL,
        sim_t            REAL    NOT NULL,
        level            TEXT    NOT NULL,
        event_type       TEXT    NOT NULL,
        m_idx            INTEGER,
        j_idx            INTEGER,
        message          TEXT,
        FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
    );

    CREATE INDEX IF NOT EXISTS idx_runs_sqc_method       ON runs(sqc_method);
    CREATE INDEX IF NOT EXISTS idx_operations_run_m      ON operations(run_id, m_idx);
    CREATE INDEX IF NOT EXISTS idx_decisions_run_time    ON decisions(run_id, sim_t);
    CREATE INDEX IF NOT EXISTS idx_decisions_run_m       ON decisions(run_id, m_idx);
    CREATE INDEX IF NOT EXISTS idx_breakdowns_run_m      ON breakdowns(run_id, m_idx);
    CREATE INDEX IF NOT EXISTS idx_schedule_cycles_run   ON schedule_cycles(run_id, sim_t);
    CREATE INDEX IF NOT EXISTS idx_sim_events_run_time   ON sim_events(run_id, sim_t);
    CREATE INDEX IF NOT EXISTS idx_sim_events_run_type   ON sim_events(run_id, event_type);
    """

    def __init__(self, db_path: Optional[Path] = None, init_schema: bool = True):
        self.db_path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # A multi-process parent initializes the schema once; workers skip it to avoid
        # racing on `PRAGMA journal_mode = WAL`, which requires an exclusive lock.
        if init_schema:
            self._init_schema()

    def _init_schema(self) -> None:
        """Create the 7 tables and indexes if they do not exist, and enable WAL mode for concurrency."""
        last_error = None
        for attempt in range(10):
            try:
                with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
                    # `PRAGMA journal_mode = WAL` can return SQLITE_BUSY immediately even
                    # with a busy timeout, so retry with a short backoff when workers race.
                    conn.execute("PRAGMA busy_timeout = 30000;")
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.executescript(self.SCHEMA_SQL)
                    # Lightweight migration for databases created by earlier schema versions
                    self._migrate_schema(conn)
                    # `closing()` only closes the connection; commit DDL/migration changes explicitly.
                    conn.commit()
                return
            except sqlite3.OperationalError as exc:
                last_error = exc
                if "locked" not in str(exc).lower() or attempt == 9:
                    raise
                time.sleep(0.2 * (attempt + 1))
        raise last_error

    # Columns introduced after the original schema. `CREATE TABLE IF NOT EXISTS` never
    # alters an existing table, so old databases must be upgraded explicitly.
    ADDED_COLUMNS = {
        "runs": {
            "status": "TEXT NOT NULL DEFAULT 'running'",
            "error": "TEXT",
        },
        "operations": {
            "completed": "INTEGER NOT NULL DEFAULT 1",
        },
        "decisions": {
            "candidates_json": "TEXT",
            "state_json": "TEXT",
            "action": "INTEGER",
            "reward": "REAL",
            "next_state_json": "TEXT",
            "episode_done": "INTEGER",
        },
    }

    @classmethod
    def _migrate_schema(cls, conn: sqlite3.Connection) -> None:
        """Converge an existing database to the current schema (drop legacy + add new columns)."""
        # 1. Drop obsolete columns from pre-existing tables.
        legacy_columns = {"decisions": ["rule_picks_json"]}
        for table, columns in legacy_columns.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column in columns:
                if column in existing:
                    conn.execute(f"ALTER TABLE {table} DROP COLUMN {column}")

        # 2. Add columns introduced by newer schema versions.
        for table, columns in cls.ADDED_COLUMNS.items():
            existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            for column, declaration in columns.items():
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

        # 3. Backfill `runs.status` for episodes finalized before the column existed.
        # `actual_span` is only written by `finalize_run`, so it is a reliable marker.
        conn.execute(
            "UPDATE runs SET status = 'completed' "
            "WHERE status = 'running' AND actual_span IS NOT NULL"
        )

    def create_run(self, run_record: dict) -> int:
        """
        Insert a `runs` row with the episode's configuration upfront and return its `run_id`.
        This enables streaming child rows (e.g., `sim_events`) against a valid foreign key
        before the episode's summary KPIs are known at the end of the simulation.
        """
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO runs (
                    created_at, seed, sqc_method, m_no, span,
                    e_utliz, due_tightness, pt_min, pt_max, pt_cv,
                    machine_bkd, mtbf, random_mtbf, mttr, random_mttr
                ) VALUES (
                    :created_at, :seed, :sqc_method, :m_no, :span,
                    :e_utliz, :due_tightness, :pt_min, :pt_max, :pt_cv,
                    :machine_bkd, :mtbf, :random_mtbf, :mttr, :random_mttr
                )
                """,
                run_record,
            )
            run_id = int(cursor.lastrowid)
            conn.commit()
            return run_id

    def append_events(self, run_id: int, events: list) -> None:
        """Batch-insert chronological simulation events into `sim_events` in a single transaction."""
        if not events:
            return
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO sim_events (
                    run_id, sim_t, level, event_type, m_idx, j_idx, message
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [(run_id, *row) for row in events],
            )
            conn.commit()

    def append_children(
        self,
        run_id: int,
        jobs_records: list,
        operations_records: list,
        breakdowns_records: list,
        schedule_cycles_records: list,
        decisions_records: list = None,
    ) -> None:
        """
        Batch-insert the smaller per-episode child tables (`jobs`, `operations`, `decisions`,
        `breakdowns`, `schedule_cycles`) in a single transaction at the end of the simulation.
        """
        decisions_records = decisions_records or []
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            cursor = conn.cursor()

            # 1. Batch-insert jobs into `jobs`
            if jobs_records:
                cursor.executemany(
                    """
                    INSERT INTO jobs (
                        run_id, j_idx, creation_t, due_t, completion_t, status,
                        tardiness, flowtime, trajectory_json, exp_pt_json, actual_pt_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(run_id, *row) for row in jobs_records],
                )

            # 2. Batch-insert operations into `operations`
            if operations_records:
                cursor.executemany(
                    """
                    INSERT INTO operations (
                        run_id, j_idx, op_seq, m_idx, decision_type, queue_len_at_dec,
                        arrival_t, begin_t, end_t, wait_t, expected_pt, actual_pt,
                        scheduled_begin_t, completed
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(run_id, *row) for row in operations_records],
                )

            # 3. Batch-insert exact decision-replay snapshots into `decisions`
            if decisions_records:
                cursor.executemany(
                    """
                    INSERT INTO decisions (
                        run_id, sim_t, m_idx, j_idx, op_seq, decision_type, queue_len,
                        chosen_pos, queue_json, candidates_json,
                        state_json, action, reward, next_state_json, episode_done
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(run_id, *row) for row in decisions_records],
                )

            # 4. Batch-insert machine breakdowns into `breakdowns`
            if breakdowns_records:
                cursor.executemany(
                    """
                    INSERT INTO breakdowns (
                        run_id, m_idx, begin_t, end_t, expected_end_t, actual_duration, prod_delay
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(run_id, *row) for row in breakdowns_records],
                )

            # 5. Batch-insert centralized rescheduling snapshots into `schedule_cycles`
            if schedule_cycles_records:
                cursor.executemany(
                    """
                    INSERT INTO schedule_cycles (
                        run_id, sim_t, trigger_type, num_jobs, num_ops,
                        solver_status, solve_time_s, m_schedule_json, j_schedule_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    [(run_id, *row) for row in schedule_cycles_records],
                )

            conn.commit()

    def finalize_run(self, run_id: int, kpi_record: dict) -> None:
        """Update the `runs` row with final summary KPIs computed at the end of the episode."""
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                """
                UPDATE runs SET
                    status           = 'completed',
                    error            = NULL,
                    actual_span      = :actual_span,
                    actual_utliz     = :actual_utliz,
                    total_jobs       = :total_jobs,
                    completed_jobs   = :completed_jobs,
                    sqc_cnt_opt      = :sqc_cnt_opt,
                    sqc_cnt_si       = :sqc_cnt_si,
                    sqc_cnt_reactive = :sqc_cnt_reactive,
                    sqc_cnt_passive  = :sqc_cnt_passive,
                    wall_time_s      = :wall_time_s,
                    opt_time_s       = :opt_time_s,
                    mean_tardiness   = :mean_tardiness,
                    max_tardiness    = :max_tardiness,
                    mean_flowtime    = :mean_flowtime,
                    max_flowtime     = :max_flowtime
                WHERE run_id = :run_id
                """,
                {**kpi_record, "run_id": run_id},
            )
            conn.commit()

    def fail_run(self, run_id: int, error: str) -> None:
        """Mark an already-created run as failed and persist the traceback.

        Without this, an exception after ``create_run`` would leave an orphaned
        ``runs`` row with NULL KPIs and no child records, indistinguishable from a
        run that is still in progress.
        """
        with closing(sqlite3.connect(self.db_path, timeout=30.0)) as conn:
            conn.execute("PRAGMA foreign_keys = ON;")
            conn.execute(
                "UPDATE runs SET status = 'failed', error = ? WHERE run_id = ?",
                (error, run_id),
            )
            conn.commit()


def draw_gantt_chart(logger: logging.Logger, recorder, **kwargs) -> None:
    """
    Plot and/or save a Gantt chart of all jobs' operations and machine breakdowns.
    Plotting is skipped when simulation span >= 250 to conserve memory.
    """
    if kwargs.get('span', 0) >= 250:
        logger.info("Gantt chart skipped for span >= 250 to conserve memory.")
        return

    # Lazy import: matplotlib is only needed when a chart is actually drawn. This keeps
    # headless/benchmark runs and the offline analysis CLI importable without a GUI stack.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib is not installed; skipping the Gantt chart.")
        return

    op_data = recorder.j_operation_dict
    bkd_data = recorder.m_bkd_dict
    if not op_data:
        return

    # Initialize figure sized proportionally to the number of machines
    fig = plt.figure(figsize=(15, recorder.m_no + 1))
    col_list = [
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan'
    ]
    gantt_chart = fig.add_subplot(1, 1, 1)
    yticks_pos = np.arange(recorder.m_no)  # vertical position of machine tick labels
    last_output = 0.0

    # PART A: Plot jobs' operation history on each machine
    for j_idx, op_history in op_data.items():
        col = col_list[j_idx % len(col_list)]
        for m_idx, begin, pt, wait in op_history:
            gantt_chart.broken_barh(
                [(begin, pt)], (m_idx - 0.25, 0.5), color=col, edgecolor='k'
            )
            gantt_chart.text(
                begin, m_idx + (j_idx % 4) * 0.13 - 0.25, j_idx,
                fontsize=10, ha='left', va='bottom',
                color='k', bbox=dict(facecolor='lightblue', alpha=0.8, pad=1)
            )
            # Track the latest operation completion time across all jobs for the x-axis upper bound
            last_output = max(last_output, begin + pt)

    # PART B: Overlay machine breakdown intervals as hatched white bars
    for m_idx, bkd_history in bkd_data.items():
        for begin, end in bkd_history:
            gantt_chart.broken_barh(
                [(begin, end - begin)], (m_idx - 0.25, 0.5),
                color='w', hatch='//', edgecolor='k'
            )

    # PART C: Configure axes, gridlines, saving, and interactive display
    plot_range = max(5.0, float(np.ceil(last_output / 5.0) * 5.0))
    gantt_chart.set_xlabel('Time in simulation')
    gantt_chart.set_ylabel('Machine index')
    gantt_chart.set_title('Operation record of jobs (Gantt Chart)')
    gantt_chart.set_yticks(yticks_pos)

    # Set major and minor grid ticks and place grid behind bars
    fig_major_ticks = np.arange(0, plot_range + 1, 10)
    fig_minor_ticks = np.arange(0, plot_range + 1, 1)
    gantt_chart.set_xticks(fig_major_ticks)
    gantt_chart.set_xticks(fig_minor_ticks, minor=True)
    gantt_chart.grid(which='major', alpha=1.0)
    gantt_chart.grid(which='minor', alpha=0.2, linestyle='--')
    gantt_chart.set_axisbelow(True)
    gantt_chart.set_xlim(0, plot_range)

    # Save figure to the active log directory before closing the figure
    if kwargs.get('save_gantt', False) and logger.handlers:
        save_path = Path(logger.handlers[0].baseFilename).parent / 'gantt_chart.png'
        fig.savefig(save_path, dpi=600, bbox_inches='tight')

    # Display pop-up window for `draw_gantt` seconds if requested
    if kwargs.get('draw_gantt', 0) > 0:
        plt.show(block=False)
        plt.pause(kwargs['draw_gantt'])

    plt.close(fig)

