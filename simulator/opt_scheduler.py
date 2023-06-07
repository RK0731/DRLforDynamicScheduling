import itertools
import pandas as pd
import numpy as np
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

class OPT_scheduler:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.schedule = {m.m_idx:[] for m in self.m_list}


    def solve_problem(self):
        # if there's only one job in system
        if len(self.recorder.in_system_jobs) == 1:
            self.solve_without_optimization()
            return
        # if more than one jobs in system, get remaining operations' info, and check the intersection between them
        self.remaining_trajectories = {}
        self.remaining_pts = {}
        self.job_intersections = {}
        # trajectory and processing time info
        for _j_idx, _job in self.recorder.in_system_jobs.items():
            # remaining trajectory, excluding current machine if being processed
            self.remaining_trajectories[_j_idx] = _job.remaining_machines if _job.status=='queuing' else _job.remaining_machines[1:]
            # processing time of remaining operations, excluding current pt if being processed
            self.remaining_pts[_j_idx] = _job.remaining_pt if _job.status=='queuing' else _job.remaining_pt[1:]
        # get the intersection betwen jobs
        for pair in (itertools.combinations(self.remaining_trajectories.keys(), 2)):
            _rem_traj_1, _rem_traj_2 = self.remaining_trajectories[pair[0]], self.remaining_trajectories[pair[1]]
            _intersec = list(set(_rem_traj_1).intersection(_rem_traj_2))
            if len(_intersec):
                self.job_intersections[pair] = _intersec
        #print(self.job_intersections)
        # if more than one job but no intersection
        if len(self.job_intersections) == 0:
            self.solve_without_optimization()
        else:
            self.solve_by_optimization()


    # no intersection between jobs, no optimization 
    def solve_without_optimization(self):
        for _j_idx, _j_object in self.recorder.in_system_jobs.items():
            for _m_idx in _j_object.remaining_machines:
                self.schedule[_m_idx] = [_j_idx]
        self.logger.info("{} >>> OPT off: new schedule {} / (sequence of jobs)".format(self.env.now, self.schedule))


    # intersection detected, needs optimizaiton
    def solve_by_optimization(self):
        # get machines' release info
        self.machine_release_T = {m.m_idx: m.release_time for m in self.m_list}
        # build the optimization model
        with gp.Env(empty=True) as self.grb_env:
            self.grb_env.setParam('LogToConsole', 0)
            self.grb_env.setParam('LogFile', str(Path.cwd() / "log" / "gurobi.log"))
            self.grb_env.start()
            with gp.Model(name="opt_scheduler", env=self.grb_env) as model:
                ''' PART I: create the decision variables
                '''
                # 1. time of the beginning of operations, for all jobs and their remaining operations
                varOpBeginT_pairing = []
                for _j_idx, _traj in self.remaining_trajectories.items():
                    varOpBeginT_pairing += list(itertools.product([_j_idx], _traj))
                varOpBeginT = model.addVars(varOpBeginT_pairing, vtype=GRB.CONTINUOUS, name="varOpBeginT")
                #print(varOpBeginT)
                # 2. precedence variables for each pair of jobs on relevant machines, if they are intersected
                varJobPrec_pairing = []
                print(self.job_intersections)
                for (_j1, _j2), _intersec in self.job_intersections.items():
                    varJobPrec_pairing += list(itertools.product([_j1], [_j2], _intersec))
                varJobPrec = model.addVars(varJobPrec_pairing, vtype=GRB.BINARY, name='varJobPrec')
                print(varJobPrec)
                ''' PART II: specify the constraints
                '''
                varJobComp = model.addVars([1,2],[1,2,3], vtype=GRB.CONTINUOUS, name='v_job_completion_T')


    def convert_to_schedule(self):
        pass


    def draw_from_schedule(self, jobs:list, m_idx:int, *args) -> int:
        pass