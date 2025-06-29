# standard imports
import collections
import gurobipy as gp
from gurobipy import GRB
import itertools
import json
import logging
import numpy as np
from ortools.sat.python import cp_model
import pandas as pd
from pathlib import Path
import time
from tabulate import tabulate
from typing import Dict, List, Tuple, Union, Literal
# project moduels
from .sequencing_rule import SequencingMethod
from ..simulator.exc import *
from ..simulator.job import Job
from ..simulator.machine import Machine


class CentralScheduler:
    def __init__(self, *args, **kwargs):
        # map the keyword arguments and declare type if necessary
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.schedule = {m.m_idx:[] for m in self.m_list}
        self.logger: logging.Logger
        # get jobs in system for scheduling
        self.in_system_jobs:Dict[int, Job] = self.recorder.in_system_jobs
        # set the log path to record complex scheudling problems
        self.ext_prob_log = {}
        self.ext_prob_log_path = Path(self.logger.handlers[0].baseFilename).parent / "over_extended_problems.json"
        # create the event
        self.build_schedule_event = self.env.event()
        # create the optimizer object
        if self.sqc_method == SequencingMethod.GurobiOptimizer:
            self.scheduler = GurobiOptimizer
        elif self.sqc_method == SequencingMethod.ORTools:
            self.scheduler = ORTools
        # process the build schedule process
        self.env.process(self.solve_problem_process())


    def solve_problem_process(self):
        while True:
            yield self.build_schedule_event
            _begin_T = time.time()
            # if there's only one job in system
            if len(self.in_system_jobs) == 1:
                self.solve_without_optimization()
            # if more than one jobs in system, get all jobs' remaining operation info, and check the intersection between them
            else:
                self.remaining_trajectories = {}
                self.remaining_pts = {}
                self.job_intersections = {}
                # extract the remaining trajectory and processing time info of jobs that not yet completed
                for _j_idx, _job in self.in_system_jobs.items():
                    if _job.status=='queuing':
                        self.remaining_trajectories[_j_idx] = _job.remaining_machines
                        self.remaining_pts[_j_idx] = _job.remaining_pt
                    elif len(_job.remaining_machines) > 1: # excluding current machine if the job is under processing
                        self.remaining_trajectories[_j_idx] = _job.remaining_machines[1:]
                        self.remaining_pts[_j_idx] = _job.remaining_pt[1:]
                # get the potential intersection between jobs' trajectory to define the precedence constraints
                for pair in (itertools.combinations(self.remaining_trajectories.keys(), 2)):
                    # get trajectories of any 2 jobs to infer intersection
                    _rem_traj_1, _rem_traj_2 = self.remaining_trajectories[pair[0]], self.remaining_trajectories[pair[1]]
                    _intersec = list(set(_rem_traj_1).intersection(_rem_traj_2))
                    if len(_intersec):
                        self.job_intersections[pair] = _intersec
                #print(self.job_intersections)
                # if more than one job but no intersection, no math programming is needed
                if len(self.job_intersections) == 0:
                    self.solve_without_optimization()
                # otherwise call the optimizer to solve the problem
                else:
                    _varOpBeginT, over_extended_problem = self.scheduler.solve_scheduling_problem(
                        self.logger, self.env, self.m_list,
                        self.job_intersections, self.remaining_trajectories, self.in_system_jobs)
                    self.convert_to_schedule(_varOpBeginT)
                    # record the over-extended problem instance
                    if not (over_extended_problem is None):
                        self.ext_prob_log[int(self.env.now)] = over_extended_problem 
                self.recorder.opt_time_expense += (time.time() - _begin_T)
            # de-activate the build schedule event
            self.build_schedule_event = self.env.event()


    # no intersection between jobs, no optimization 
    def solve_without_optimization(self):
        for _j_idx, _j_object in self.in_system_jobs.items():
            # remove "processing" operations
            if _j_object.status == "queuing":
                _j_traj = _j_object.remaining_machines
            elif len(_j_object.remaining_machines) > 1:
                _j_traj = _j_object.remaining_machines[1:]
            else:
                continue
            for _m_idx in _j_traj:
                self.schedule[_m_idx] = [_j_idx]
        # if there is a valid schedule
        if [sch for sch in self.schedule.values() if sch != []]:
            self.logger.info("{} > Passive schedule: \n{}".format(self.env.now, tabulate([
                ["Machine"]+list(self.schedule.keys()), ["Schedule"]+list(self.schedule.values())], headers="firstrow", tablefmt="psql")))


    # use the optmized operation begin time to build the schedule
    def convert_to_schedule(self, varOpBeginT: dict):
        # reset the schedule
        self.schedule = {m.m_idx:[] for m in self.m_list}
        self.j_op_by_schedule = {_j_idx:[] for _j_idx in self.remaining_trajectories.keys()}
        # reorder the vaOpBeginT (tuple dict), by the value of variable (begin time of operation)
        _reordered_varOpBeginT: list = sorted(varOpBeginT.items(), key = lambda item: item[1])
        for (_j_idx, _m_idx), T in _reordered_varOpBeginT:
            # add job index to respective machine's new schedule
            self.schedule[_m_idx].append(_j_idx)
            # job's expected operation begin time in schedule
            self.j_op_by_schedule[_j_idx].append((_m_idx, round(T, 1)))
        # log the machines' sequence
        self.logger.debug("Machines' sequence in new schedule: \n{}".format(tabulate(
            [["M.idx", "Job sequence [j_idx]"], 
             *self.schedule.items()], 
             headers="firstrow", tablefmt="psql")))
        # log jobs' operations
        self.logger.debug("Jobs' operations in new schedule: \n{}".format(tabulate(
            [["J.idx", "Remaining Operations (m_idx, opBeginT)"], 
             *[[_j_idx, op] for _j_idx, op in self.j_op_by_schedule.items()]], 
             headers="firstrow", tablefmt="psql")))
        self.update_machine_after_optimization()


    # update the job index that all machines should wait for
    def update_machine_after_optimization(self):
        for m in self.m_list:
            # some machine get an empty schedule
            if not self.schedule[m.m_idx]:
                continue
            #self.logger.debug("Machine {} schedule before {}".format(m.m_idx, self.schedule[m.m_idx]))
            if m.status == "strategic_idle":
                # if the machine is currently in strategic idleness status
                # need to pop from schedule because the sequencing decision is considered made
                m.next_job_in_schedule = self.schedule[m.m_idx].pop(0)
            else:
                # otherwise (idle, down or processing) just copy the job index
                # later the machine will call [draw_from_schedule] function to pop from schedule
                m.next_job_in_schedule = self.schedule[m.m_idx][0]
            #self.logger.debug("Machine {} schedule after {}".format(m.m_idx, self.schedule[m.m_idx]))
            m.update_status_after_new_schedule()


    def draw_from_schedule(self, m_idx:int) -> int:
        self.logger.debug("Draw from schedule, Machine {}, current schedule {}, queue {}".format(m_idx, self.schedule[m_idx], [j.j_idx for j in self.m_list[m_idx].queue]))
        next_job_in_schedule = self.schedule[m_idx].pop(0)
        # returned value is the job index in schedule, not the position of job in queue
        # as job may not yet arrived
        return next_job_in_schedule
    

    def post_simulation(self):
        if self.ext_prob_log:
            print("{} over-extended scheduling problem is recorded, saved to {}".format(len(self.ext_prob_log), self.ext_prob_log_path))
        # after the process, write the over-extended problem instances
        with open(self.ext_prob_log_path, "w") as f:
            json.dump(self.ext_prob_log, f)
        return




class ORTools:
    @classmethod
    def solve_scheduling_problem(cls, logger, env, m_list:List[Machine], 
                                 job_intersections, remaining_trajectories:Dict[int, list], in_system_jobs:Dict[int, Job]
                                 ):
        START_T = time.time()
        # get machines' release time info
        machine_release_T = {m.m_idx: int(max(m.release_T, env.now)) for m in m_list}
        # get jobs' available time info
        job_available_T = {_j_idx: int(max(in_system_jobs[_j_idx].available_T, env.now)) for _j_idx in remaining_trajectories.keys()}
        # build the OR-Tools constrained programming model
        model = cp_model.CpModel()
        ''' 
        PART I: create the variables and pairings
        '''
        # get the lower and upper limit for INT variables, which the sum of all remaining operations' pt
        NOW = int(env.now)
        UL = NOW + int(sum([sum(J.remaining_pt) for J in in_system_jobs.values()]))
        # variable storage
        all_jobs, all_ops = {}, {}
        m_to_ops = {m.m_idx: [] for m in m_list}
        job_tuple = collections.namedtuple("job", ['completion', 'discrepency', 'tardiness'])
        op_tuple = collections.namedtuple("operation", ['begin', 'end', 'interval'])
        # 1. job/operation/machine decision variables
        for j_idx, traj in remaining_trajectories.items():
            # operation-specific variables
            for m_idx in traj:
                suffix = f"_j{j_idx}_m{m_idx}"
                # begin of operation j,m
                _varOpBeginT = model.NewIntVar(NOW, UL, "varOpBeginT" + suffix)
                # end of operation j,m
                _varOpEndT = model.NewIntVar(NOW, UL, "varOpEndT" + suffix)
                # the interval variable represeting operation j,m
                _varOpInterval = model.NewIntervalVar(_varOpBeginT, in_system_jobs[j_idx].pt_by_m_idx[m_idx], _varOpEndT, 
                                                      "varOpInterval" + suffix)
                # store the variables
                all_ops[j_idx, m_idx] = op_tuple(begin = _varOpBeginT, end = _varOpEndT, interval = _varOpInterval)
                m_to_ops[m_idx].append(_varOpInterval)
            # 2. job completion discrepency and tardiness
            _varJobDiscrepency = model.NewIntVar(-1000, 1000, f"varJobDiscrepency_{j_idx}")
            _varJobTardiness = model.NewIntVar(0, 1000, f"varJobTardiness_{j_idx}")
            # store the completion, discrepency, and tardiness variables (dummy for now)
            all_jobs[j_idx] = job_tuple(
                completion = _varOpEndT, 
                discrepency = _varJobDiscrepency,
                tardiness = _varJobTardiness
                )
        # 3. schedule makespan, not adjusted by the starting time of a cycle
        varMakespan = model.NewIntVar(NOW, UL, "varMakespan")
        varCumTardiness = cp_model.LinearExpr.Sum([J.tardiness for J in all_jobs.values()])
        ''' 
        PART II: specify the constraints
        '''
        # math programming model specs
        model_spec = {'op_sqc':0, 'op_overlap':0, 'J_release':0, 'M_release':0, 'J_discrepency':0, 'J_tardiness':0}
        for j_idx, traj in remaining_trajectories.items():
            # 1. a job's operations must be processed following job's trajectory
            for op_sqc in range(len(traj) -1):
                model.Add(all_ops[j_idx, traj[op_sqc + 1]].begin >= all_ops[j_idx, traj[op_sqc]].end)
                model_spec['op_sqc'] += 1
            # 2. job cannot be processed bafore required machine is released or itself became available
            # 2.1 job's all operations can be processed only after machine release
            for m_idx in traj:
                model.Add(all_ops[j_idx, m_idx].begin >= machine_release_T[m_idx])
                model_spec['M_release'] += 1
            # 2.2 job's first operation can be processed only when it becomes available
            model.Add(all_ops[j_idx, traj[0]].begin >= job_available_T[j_idx])
            model_spec['J_release'] += 1
            # 2.3 calculate the completion time discrepency from job completion time
            model.Add(all_jobs[j_idx].discrepency == all_jobs[j_idx].completion - int(in_system_jobs[j_idx].due))
            # 2.4 calculate the tardiness from discrepency
            model.AddMaxEquality(all_jobs[j_idx].tardiness, [0, all_jobs[j_idx].discrepency])
        # 3. no overlaps amongst all operations for a machine
        for machine in m_to_ops:
            model.AddNoOverlap(m_to_ops[machine])
            model_spec['op_overlap'] += len(m_to_ops[machine])
        # 4. system-level performance variables 
        # 4.1 cumulative tardiness
        # 4.2 the makespan of the schedule in this cycle
        #model.AddMaxEquality(varMakespan, [all_ops[j_idx, traj[-1]].end for j_idx, traj in remaining_trajectories.items()])
        ''' 
        PART III: specify the objective of optimization, and run the optimization
        '''
        logger.debug('Problem spec.: [{} Jobs], [{} Ops]. Constraints: [{} op_sqc]; [{} op_overlap]; [{} M_release]; [{} J_release/discrepency/tardiness]'.format(
            len(in_system_jobs), sum(len(x) for x in remaining_trajectories.values()), model_spec['op_sqc'], model_spec['op_overlap'], model_spec['M_release'], model_spec['J_release']))
        model.Minimize(varCumTardiness)
        solver = cp_model.CpSolver()
        status = solver.Solve(model)
        '''
        PART IV: convert the gurobi tupledict to normal Python dict
        '''
        time_expense = round(time.time() - START_T, 3)
        logger.debug("OR_Tools CP Model solving process elapsed, model status: {}, time expense: {}s".format(
            solver.StatusName(status), time_expense))
        # record the overextended problem instance
        if time_expense > 1:
            over_extended_problem = {
                'Jobs': {j_idx: {
                    "ops": [(float(m_idx), float(in_system_jobs[j_idx].pt_by_m_idx[m_idx])) for m_idx in traj],
                    "avail": job_available_T[j_idx]
                    } for j_idx, traj in remaining_trajectories.items()
                },
                'Machines': {m_idx: release_T for m_idx, release_T in machine_release_T.items()},
                'Expense': time_expense
            }
        else:
            over_extended_problem = None
        # extract the value of varOpBeginT variables
        converted_varOpBeginT = {key: solver.Value(op.begin) for key, op in all_ops.items()}
        # return only the operation begin time to build the schedule
        return converted_varOpBeginT, over_extended_problem





class GurobiOptimizer:
    @classmethod
    def solve_scheduling_problem(cls, logger, env, m_list:List[Machine], 
                                 job_intersections, remaining_trajectories:Dict[int, list], in_system_jobs:Dict[int, Job]
                                 ):
        grb_msg = {2:'optimal', 3:'infeasible', 4:'infeasible or unbounded', 9:'time limit', 11:'interrupted'}
        START_T = time.time()
        # get machines' release info
        machine_release_T = {m.m_idx: max(m.release_T, env.now) for m in m_list}
        # get jobs' available time info
        job_available_T = {_j_idx: max(in_system_jobs[_j_idx].available_T, env.now) for _j_idx in remaining_trajectories.keys()}
        # build the optimization model
        with gp.Env(empty=True) as grb_env:
            grb_env.setParam('LogToConsole', 0)
            grb_env.setParam('LogFile', str(Path(logger.handlers[0].baseFilename).parent / "gurobi.log"))
            grb_env.start()
            with gp.Model(name="opt_scheduler", env=grb_env) as model:
                ''' 
                PART I: create the variables and necessary pairs
                '''
                # 1. time of the beginning of operations, for all jobs and their remaining operations
                pairOpBeginT = []
                pairOpSqc = []
                pairJobFirstOp = []
                pairJobLastOp = []
                for _j_idx, _traj in remaining_trajectories.items():
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
                # 2. job completion time
                varJobCompT = model.addVars(list(remaining_trajectories.keys()), vtype=GRB.CONTINUOUS, name="varJobCompT")
                # discrepency of completion time, can be eother earliness or tardiness
                varJobCompDiscr = model.addVars(list(remaining_trajectories.keys()), lb=-1000, vtype=GRB.CONTINUOUS, name="varJobCompDiscr")
                # job tardiness, non-negative
                varJobTardiness = model.addVars(list(remaining_trajectories.keys()), vtype=GRB.CONTINUOUS, name="varJobTardiness")
                # schedule makespan, not adjusted by the starting time of a cycle
                varMakespan = model.addVar(vtype=GRB.CONTINUOUS, name="varMakespan")
                # 3. precedence variables for each pair of jobs on relevant machines, if they are intersected
                pairJobPrec = []
                #print(job_intersections)
                for (_j1, _j2), _intersec in job_intersections.items():
                    pairJobPrec += list(itertools.product([_j1], [_j2], _intersec))
                # binary variable to indicate the precedence between job 1 and job 2 on a machine
                # equals 0 if job 1 preceeds job 2 on that machine, 1 otherwise
                varJobPrec = model.addVars(pairJobPrec, vtype=GRB.BINARY, name='varJobPrec')
                logger.debug('Job in system: {}, Operation begin time pairs: {}, Job operations sequence pairs: {}, Job precedence pairs: {}'.format(
                    len(in_system_jobs), len(pairOpBeginT), len(pairOpSqc), len(pairJobPrec)))
                ''' 
                PART II: create the constraints
                '''
                # 1. a job's operations must be processed following job's trajectory
                constrOpSqc = model.addConstrs(
                    (varOpBeginT[j, m1] + in_system_jobs[j].pt_by_m_idx[m1] <= varOpBeginT[j, m2] for (j,m1), (j,m2) in pairOpSqc),
                    name = 'constrOpSqc')
                # 2. job cannot be processed bafore required machine is released or itself became available
                # 2.1 job's all operations can be processed only after machine release
                constrMachineRelease = model.addConstrs(
                    (varOpBeginT[j, m] >= machine_release_T[m] for j, m in pairOpBeginT),
                    name = 'constrMachineRelease')
                # 2.2 job's first operation can be processed only itself becomes available
                constrJobAvailable = model.addConstrs(
                    (varOpBeginT[j, m] >= job_available_T[j] for j, m in pairJobFirstOp),
                    name = 'constrJobAvailable')
                # 3. all operations must be processed following the precedence relations between jobs
                # 3.1 if job 1 preceeds job 2 <--> precedence variable = 0
                constrJobPrec_1 = model.addConstrs(
                    ((varOpBeginT[j1, m] + in_system_jobs[j1].pt_by_m_idx[m]) * (1 - varJobPrec[j1, j2, m]) <= varOpBeginT[j2, m] for j1, j2, m in pairJobPrec),
                    name = 'constrJobPrec_0')
                # 3.2 if job 2 preceeds job 1 <--> precedence variable = 1
                constrJobPrec_0 = model.addConstrs(
                    ((varOpBeginT[j2, m] + in_system_jobs[j2].pt_by_m_idx[m]) * varJobPrec[j1, j2, m] <= varOpBeginT[j1, m] for j1, j2, m in pairJobPrec),
                    name = 'constrJobPrec_1')
                # 4. performance variables (dummies, not decisional)
                # 4.1 get the completion time of jobs
                constrJobCompT = model.addConstrs(
                    (varJobCompT[j] == varOpBeginT[j, m] + in_system_jobs[j].pt_by_m_idx[m] for j, m in pairJobLastOp),
                    name = 'constrJobCompT')
                # 4.2 get the completion discrepency (earliness and tardiness)
                constrJobCompDiscr = model.addConstrs(
                    (varJobCompDiscr[j] == varJobCompT[j] - in_system_jobs[j].due for j, m in pairJobLastOp),
                    name = 'constrJobCompDiscr')
                # 4.3 get the job tardiness
                constrJobTardiness = model.addConstrs(
                    (varJobTardiness[j] == gp.max_(varJobCompDiscr[j], constant = 0) for j, m in pairJobLastOp),
                    name = 'constrJobTardiness')
                # 4.4 the makespan of the schedule in this cycle
                constrMakespan = model.addConstr(varMakespan == gp.max_([varJobCompT[j] for j, m in pairJobLastOp]),name = 'constrMakespan')
                ''' 
                PART III: create the objective(s), and run the optimization
                '''
                model.update()
                model.setParam('time_limit', 10)
                # the primary (tier 1) objective of optimization, however, Gurobi can be "lazy"
                # optimization process terminates as soon as Gurobi finds no improvements can be obtained
                model.setObjective(varJobTardiness.sum(), GRB.MINIMIZE)
                # therefore we use the secondary objective in hierachical optimization
                # it can only be optimized without compromising the primary objective
                # tier 2 objective, minimize the makespan of entire produiction schedule
                model.setObjectiveN(expr = varMakespan, index = 1, priority = -1)
                # tier 3 objective, let each operation start as early as possible
                model.setObjectiveN(expr = varOpBeginT.sum(), index = 2, priority = -2)
                '''
                for j, m in pairJobLastOp:
                    model.setObjectiveN(expr = varJobCompT[j], index = j+2, priority = -2)
                '''
                # adding this constraint would produce perfect match schedule at the cost fo computation time
                #model.setObjectiveN(expr = varOpBeginT.sum(), index = 1e5, priority = -3)
                # run the optimization
                model.optimize()
                '''
                PART IV: convert the gurobi tupledict to normal Python dict
                '''
                time_expense = round(time.time() - START_T, 3)
                logger.debug("Optimization elapsed, model status: {}, time expense: {}s".format(
                    grb_msg[model.status], time_expense))
                # extract the value of varOpBeginT variables
                converted_varOpBeginT = {key: var.X for key, var in varOpBeginT.items()}
        # record the extended problem instance
        if time_expense > 1:
            over_extended_problem = [[(float(m_idx), float(in_system_jobs[j_idx].pt_by_m_idx[m_idx])) 
                  for m_idx in traj] for j_idx, traj in remaining_trajectories.items()]
        else:
            over_extended_problem = None
        # return only the operation begin time to build the schedule
        return converted_varOpBeginT, over_extended_problem