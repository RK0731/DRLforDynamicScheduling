import gurobipy as gp
from gurobipy import GRB

class Scheduler:
    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.logger.debug('Gurobi scheduler is created')
        pass


    def build_schedule(self, machines, jobs):
        model = gp.Model(name="opt_scheduler")
        # get the pairing of job and machine, to build the starting time of operation constraint
        for j in jobs:
            j_idx = j.j_idx
            m_idx_list = j.remaining_m
            pt_list = j.remaining_pt
            opBeginVar = model.addVars(list(j_idx), m_idx_list, vtype=GRB.CONTINUOUS, name="operation_begin_T")
        # the available time of all machines (the time them become available)
        mAvailVar = model.addVars(list(j_idx), m_idx_list, vtype=GRB.CONTINUOUS, name="machine_avail_T")
        # build the job pairs for all jobs that need to be processed on a same machine
        jobPrecVar = model.addVars([1,2],[1,2,3], vtype=GRB.BINARY, name='job_prec')


    def convert_to_schedule(self):
        pass