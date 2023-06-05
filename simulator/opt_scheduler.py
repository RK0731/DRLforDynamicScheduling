import pandas as pd
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

class OPT_scheduler:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.schedule = {}


    def solve_problem(self):
        # get the snapshot of system
        _jobs = self.recorder.in_system_jobs
        _machines = self.m_list
        # one job, no optimization
        if len(_jobs) == 1:
            self.single_job_case()
        else:
            self.multiple_job_case()

    
    def multiple_job_case(self):
        with gp.Env(empty=True) as self.grb_env:
            self.grb_env.setParam('LogToConsole', 0)
            self.grb_env.setParam('LogFile', str(Path(__file__).parent/"log"/"gurobi.log"))
            self.grb_env.start()        
            with gp.Model(name="opt_scheduler", env=self.grb_env) as model:
                # get the pairing of job and machine, to build the starting time of operation constraint
                m_idx_list = []
                for j in _jobs:
                    j_idx = j.j_idx
                    remaining_m_list = j.remaining_m
                    pt_list = j.remaining_pt
                    varOpBegin = model.addVars(list(j_idx), m_idx_list, vtype=GRB.CONTINUOUS, name="v_operation_begin_T")
                # the available time of all machines (the time them become available)
                varMachineRel = model.addVars([m.m_idx for m in self.m_list], vtype=GRB.CONTINUOUS, name="v_machine_release_T")
                # build the job pairs for all jobs that need to be processed on a same machine
                varJobPrec = model.addVars([1,2],[1,2,3], vtype=GRB.BINARY, name='v_job_precedence')
                varJobComp = model.addVars([1,2],[1,2,3], vtype=GRB.CONTINUOUS, name='v_job_completion_T')


    def convert_to_schedule(self):
        pass


    def draw_from_schedule(self, jobs, m_idx, *args):
        pass