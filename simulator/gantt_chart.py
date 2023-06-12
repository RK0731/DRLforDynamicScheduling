import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path

class Draw:
    def __init__(self, recorder, **kwargs):
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
            fig.savefig(Path.cwd() / 'log' / 'gantt_chart.png', dpi=600, bbox_inches='tight')