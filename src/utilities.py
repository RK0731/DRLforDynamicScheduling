#!/usr/bin/env python3
"""
Author: Liu Renke
Utility module providing logging configuration and Gantt chart visualization for the simulator.
"""

import copy
from datetime import datetime as dt
import json
import logging
from logging.config import dictConfig
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

# Root directory where all timestamped simulation log folders are stored
LOG_ROOT_DIR = Path.cwd() / 'logs'
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


def setup_logger(stream: bool = True, keep: int = 5) -> logging.Logger:
    """
    Create a timestamped log directory, prune old log directories beyond `keep`,
    and initialize the 'sim_logger' logger instance.
    """
    # Generate a fresh timestamped folder for this simulation run
    current_log_dir = LOG_ROOT_DIR / dt.now().strftime("%Y%m%d-%H%M%S%f")
    current_log_dir.mkdir(parents=True, exist_ok=True)

    # Prune obsolete log folders from root logging directory (ignore non-directory files such as .DS_Store)
    folders = [f for f in LOG_ROOT_DIR.iterdir() if f.is_dir()]
    folders.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for folder_path in folders[keep:]:
        shutil.rmtree(folder_path, ignore_errors=True)

    # Deep-copy LOG_CONFIG so disabling stream handler does not permanently mutate the template dictionary
    config = copy.deepcopy(LOG_CONFIG)
    config['handlers']['sim_log_file']['filename'] = str(current_log_dir / "sim.log")
    if not stream and "console" in config['loggers']['sim_logger']['handlers']:
        config['loggers']['sim_logger']['handlers'].remove("console")

    dictConfig(config)
    return logging.getLogger("sim_logger")


def create_logger(log_dir: Path = Path('./log'), stream: bool = True, keep: int = 10) -> logging.Logger:
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


def draw_gantt_chart(logger: logging.Logger, recorder, **kwargs) -> None:
    """
    Plot and/or save a Gantt chart of all jobs' operations and machine breakdowns.
    Plotting is skipped when simulation span >= 250 to conserve memory.
    """
    if kwargs.get('span', 0) >= 250:
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

