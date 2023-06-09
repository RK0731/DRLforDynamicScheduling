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
        self.in_system_jobs = self.recorder.in_system_jobs
        # overwrite the gurobi log file
        open(Path.cwd() / "log" / "gurobi.log", 'w').close()


    def solve_problem(self, **kwargs):
        # if there's only one job in system
        if len(self.in_system_jobs) == 1:
            self.solve_without_optimization()
            return
        # if more than one jobs in system, get remaining operations' info, and check the intersection between them
        self.remaining_trajectories = {}
        self.remaining_pts = {}
        self.job_intersections = {}
        # extract the trajectory and processing time info of jobs that not yet to be completed
        for _j_idx, _job in self.in_system_jobs.items():
            if _job.status=='queuing':
                self.remaining_trajectories[_j_idx] = _job.remaining_machines
                self.remaining_pts[_j_idx] = _job.remaining_pt
            elif len(_job.remaining_machines) > 1: # excluding current machine if being processed
                self.remaining_trajectories[_j_idx] = _job.remaining_machines[1:]
                self.remaining_pts[_j_idx] = _job.remaining_pt[1:]
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
        for _j_idx, _j_object in self.in_system_jobs.items():
            for _m_idx in _j_object.remaining_machines:
                self.schedule[_m_idx] = [_j_idx]
        self.logger.info("{} >>> OPT off: new schedule {} / (sequence of jobs)".format(self.env.now, self.schedule))


    # intersection detected, needs optimizaiton
    def solve_by_optimization(self):
        # get machines' release info
        self.machine_release_T = {
            m.m_idx: max(m.release_T, self.env.now) for m in self.m_list}
        # get jobs' available time info
        self.job_available_T = {
            _j_idx: max(self.in_system_jobs[_j_idx].available_T, self.env.now) for _j_idx in self.remaining_trajectories.keys()}
        # build the optimization model
        with gp.Env(empty=True) as self.grb_env:
            self.grb_env.setParam('LogToConsole', 0)
            self.grb_env.setParam('LogFile', str(Path.cwd() / "log" / "gurobi.log"))
            self.grb_env.start()
            with gp.Model(name="opt_scheduler", env=self.grb_env) as model:
                ''' 
                PART I: create the variables and necessary pairs
                '''
                # 1. time of the beginning of operations, for all jobs and their remaining operations
                pairOpBeginT = []
                pairOpSqc = []
                pairJobFirstOp = []
                pairJobLastOp = []
                for _j_idx, _traj in self.remaining_trajectories.items():
                    # (j_idx, m_idx) pairing of begin time variables
                    pairOpBeginT += list(itertools.product([_j_idx], _traj))
                    # pairing of consecutive machines
                    pairConsecM = list(zip(_traj, _traj[1:]))
                    # [(j1_idx, m1_idx), (j2_idx, m2_idx)] pairings of begin time variables for consecutive operations
                    pairOpSqc += [list(itertools.product([_j_idx], mp)) for mp in pairConsecM]
                    # job's first and last operation
                    pairJobFirstOp += [(_j_idx, _traj[0])]
                    pairJobLastOp += [(_j_idx, _traj[-1])]
                # continuous variables indicating the beginning time of all operations
                varOpBeginT = model.addVars(pairOpBeginT, vtype=GRB.CONTINUOUS, name="varOpBeginT")
                self.logger.debug('Operation begin time pairs (j, m)\n{}'.format(pairOpBeginT))
                self.logger.debug('Job operations sequence pairs [(j, m1), (j, m2)]\n{}'.format(pairOpSqc))
                # 2. job completion time
                varJobCompT = model.addVars(list(self.remaining_trajectories.keys()), vtype=GRB.CONTINUOUS, name="varJobCompT")
                # discrepency of completion time, can be eother earliness or tardiness
                varJobCompDiscr = model.addVars(list(self.remaining_trajectories.keys()), lb=-1000, vtype=GRB.CONTINUOUS, name="varJobCompDiscr")
                # job tardiness, non-negative
                varJobTardiness = model.addVars(list(self.remaining_trajectories.keys()), vtype=GRB.CONTINUOUS, name="varJobTardiness")
                # schedule makespan, not adjusted by the starting time of a cycle
                varMakespan = model.addVar(vtype=GRB.CONTINUOUS, name="varMakespan")
                # 3. precedence variables for each pair of jobs on relevant machines, if they are intersected
                pairJobPrec = []
                #print(self.job_intersections)
                for (_j1, _j2), _intersec in self.job_intersections.items():
                    pairJobPrec += list(itertools.product([_j1], [_j2], _intersec))
                # binary variable to indicate the precedence between job 1 and job 2 on a machine
                # equals 0 if job 1 preceeds job 2 on that machine, 1 otherwise
                varJobPrec = model.addVars(pairJobPrec, vtype=GRB.BINARY, name='varJobPrec')
                self.logger.debug('Job precedence pairs (j1, j2, m)\n{}'.format(pairJobPrec))
                ''' 
                PART II: specify the constraints
                '''
                # 1. a job's operations must be processed following job's trajectory
                constrOpSqc = model.addConstrs(
                    (varOpBeginT[j, m1] + self.in_system_jobs[j].pt_by_m_idx[m1] <= varOpBeginT[j, m2] for (j,m1), (j,m2) in pairOpSqc),
                    name = 'constrOpSqc')
                # 2. job cannot be processed bafore required machine is released or itself became available
                # 2.1 job's first operation can be processed only after machine release
                constrMachineRelease = model.addConstrs(
                    (varOpBeginT[j, m] >= self.machine_release_T[m] for j, m in pairJobFirstOp),
                    name = 'constrMachineRelease')
                # 2.2 job's first operation can be processed only itself becomes available
                constrJobAvailable = model.addConstrs(
                    (varOpBeginT[j, m] >= self.job_available_T[j] for j, m in pairJobFirstOp),
                    name = 'constrJobAvailable')
                # 3. all operations must be processed following the precedence relations between jobs
                # 3.1 if job 1 preceeds job 2 <--> precedence variable = 0
                constrJobPrec_1 = model.addConstrs(
                    ((varOpBeginT[j1, m] + self.in_system_jobs[j1].pt_by_m_idx[m]) * (1 - varJobPrec[j1, j2, m]) <= varOpBeginT[j2, m] for j1, j2, m in pairJobPrec),
                    name = 'constrJobPrec_0')
                # 3.2 if job 2 preceeds job 1 <--> precedence variable = 1
                constrJobPrec_0 = model.addConstrs(
                    ((varOpBeginT[j2, m] + self.in_system_jobs[j2].pt_by_m_idx[m]) * varJobPrec[j1, j2, m] <= varOpBeginT[j1, m] for j1, j2, m in pairJobPrec),
                    name = 'constrJobPrec_1')
                # 4. dummy variables
                # 4.1 get the completion time of jobs
                constrJobCompT = model.addConstrs(
                    (varJobCompT[j] == varOpBeginT[j, m] + self.in_system_jobs[j].pt_by_m_idx[m] for j, m in pairJobLastOp),
                    name = 'constrJobCompT')
                # 4.2 get the completion discrepency (earliness and tardiness)
                constrJobCompDiscr = model.addConstrs(
                    (varJobCompDiscr[j] == varJobCompT[j] - self.in_system_jobs[j].due for j, m in pairJobLastOp),
                    name = 'constrJobCompDiscr')
                # 4.3 get the job tardiness
                constrJobTardiness = model.addConstrs(
                    (varJobTardiness[j] == gp.max_(varJobCompDiscr[j], constant = 0) for j, m in pairJobLastOp),
                    name = 'constrJobTardiness')
                # 4.4 the makespan of the schedule in this cycle
                constrMakespan = model.addConstr(varMakespan == gp.max_([varJobCompT[j] for j, m in pairJobLastOp]),name = 'constrMakespan')
                ''' 
                PART III: specify the objective of optimization, and run the optimization
                '''
                model.update()
                model.setParam('time_limit', 10)
                # the primary objective of optimization, however, Gurobi can be "lazy"
                # optimization process terminates as soon as Gurobi finds no improvements can be obtained
                model.setObjective(varJobTardiness.sum(), GRB.MINIMIZE)
                # therefore we use the secondary objective in hierachical optimization
                # it can only be optimized without degrading the primary objective
                model.setObjectiveN(expr = varMakespan, index= 1, priority = -1)
                # run the optimization
                model.optimize()
                '''
                PART IV: convert the result to valid schedule
                '''
                for v in model.getVars():
                    print('%s %g' % (v.VarName, v.X))
                self.convert_to_schedule(model.getVarByName("varJobCompT"), model.getVarByName("varJobPrec"))
        # close the environment, release the resource    
        self.grb_env.close()


    def convert_to_schedule(self, varJobCompT, varJobPrec):
        print(varJobCompT)
        for x in varJobCompT:
            print(x)


    def draw_from_schedule(self, jobs:list, m_idx:int, *args) -> int:
        pass