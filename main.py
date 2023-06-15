#!/usr/bin/env python3

from simulator.shopfloor import *


if __name__ == '__main__':
    spf = Shopfloor(m_no = 5, span = 50, pt_range = [1,10], due_tightness = 1.5, E_utliz = 0.85,
                    machine_breakdown = True, MTBF = 100, MTTR = 10, random_MTBF = True, random_MTTR = False,
                    processing_time_variability = False, pt_cv = 0.1,
                    draw_gantt = 20, save_gantt = True, seed = 56578802,
                    sqc_rule = SQC_rule.opt_scheduler
                    )
    spf.run_simulation()