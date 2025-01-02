import matplotlib.pyplot as plt
import json
import pickle
import re
import numpy as np
import os
import shutil
import logging
from pathlib import Path
import datetime as dt
from typing import List, Optional, Literal, Dict


def create_logger(log_dir = Path('./log'), stream=True, keep=10):
    logger = logging.getLogger(__name__)
    # create a new folder under "log" directory, named by current time
    _current_T = dt.datetime.now().strftime("%Y%m%d-%H%M%S%f")
    # then create new log directory
    if not log_dir.exists(): 
        log_dir.mkdir()
    log_path = log_dir / _current_T
    # check name of folder to avoid name clash
    if not log_path.exists():
        log_path.mkdir()
    else:
        _cnt = 1
        while log_path.exists():
            _cnt += 1
            log_path =  Path('./log', _current_T + f"({_cnt})")
        log_path.mkdir()
    # clear old logger handlers
    for hdlr in logger.handlers[:]:
        logger.removeHandler(hdlr)
        hdlr.close()
    with open(Path.cwd() / "config" / "logger_config.json") as f:
        # load logger cofig and point all log files to log path
        log_config = json.load(f)
        log_config['handlers']['root_file']['filename'] = log_path/'simulation.log'
        # remove the stream logger if not specified
        if not stream:
            log_config['loggers']['sim_logger']['handlers'].pop()
        # set the config and create logger
        logging.config.dictConfig(log_config)
        logger = logging.getLogger("sim_logger")
    logger.info(f"Log path set to [{log_path}]")
    # remove obsolete log
    folders = [f for f in os.listdir(log_dir) if os.path.isdir(os.path.join(log_dir, f))]
    folders.sort(key=lambda x: os.path.getmtime(os.path.join('log', x)), reverse=True)
    # Keep only the latest N (default 20) folders
    folders_to_keep = folders[:keep]
    # Remove the folders that exceed the limit
    for folder in folders:
        if folder not in folders_to_keep:
            folder_path = os.path.join('./log', folder)
            shutil.rmtree(folder_path)
    return logger


def draw_gantt_chart(logger, recorder, **kwargs):
    if kwargs['span']>=250:
        return
    # if duration is ok, draw the figure
    fig = plt.figure(figsize=(15, recorder.m_no+1))
    col_list = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan']
    gantt_chart = fig.add_subplot(1,1,1)
    yticks_pos = np.arange(recorder.m_no) # vertical position of tick labels

    '''
    PART A. jobs' operation history
    '''
    op_data = recorder.j_operation_dict
    for x in op_data.items():
        j_idx = x[0] 
        op_history = x[1]
        col = col_list[j_idx%10]
        for m_idx, begin, pt, wait in op_history:
            gantt_chart.broken_barh(
                [(begin, pt)], (m_idx-0.25, 0.5), color=col, edgecolor='k'
                )
            gantt_chart.text(
                begin, m_idx+(j_idx%3)*0.13-0.25, j_idx, fontsize=10, ha='left', va='bottom', color='k'
                )
            last_output = begin + pt

    '''
    PART B. 
    '''
    bkd_data = recorder.m_bkd_dict
    for x in bkd_data.items():
        m_idx = x[0] 
        bkd_history = x[1]
        for begin, end in bkd_history:
            gantt_chart.broken_barh(
                [(begin, end-begin)], (m_idx-0.25, 0.5), color='w', hatch='//', edgecolor='k'
                )

    '''
    PART C. 
    '''
    plot_range = np.ceil(last_output/5)*5
    gantt_chart.set_xlabel('Time in simulation')
    gantt_chart.set_ylabel('Machine index')
    gantt_chart.set_title('Operation record of jobs (Gantt Chart)')
    gantt_chart.set_yticks(yticks_pos)
    #gantt_chart.set_yticklabels(yticklabels)
    # set grid and set grid behind bars
    fig_major_ticks = np.arange(0, plot_range+1, 10)
    fig_minor_ticks = np.arange(0, plot_range+1, 1)
    gantt_chart.set_xticks(fig_major_ticks)
    gantt_chart.set_xticks(fig_minor_ticks, minor=True)
    # different settings for the grids:
    gantt_chart.grid(which='major', alpha=1)
    gantt_chart.grid(which='minor', alpha=0.2, linestyle='--')
    gantt_chart.set_axisbelow(True)
    # limit
    gantt_chart.set_xlim(0, plot_range)
    #gantt_chart.set_ylim(0, recorder.m_no)


    if 'draw_gantt' in kwargs and kwargs['draw_gantt']>0:
        plt.show(block=False)
        plt.pause(kwargs['draw_gantt'])
        plt.close(fig)
    if 'save_gantt' in kwargs and kwargs['save_gantt']:
        fig.savefig((Path(logger.handlers[1].baseFilename)).parent / 'gantt_chart.png', dpi=600, bbox_inches='tight')
