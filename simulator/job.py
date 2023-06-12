"""
This is the job class, carries the information of trajectory, processing time, due date, etc.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Union, Literal, Any


@dataclass
class Job:
    env: Any
    logger: Any
    recorder: Any
    j_idx: int
    trajectory: np.array
    pt_by_m_idx: np.array
    pt_range: list[Union[int, float]]
    pt_cv: Union[int, float]
    due_tightness: float
    status: Literal["queuing", "processing", "completed"] = "queuing"
    transfer_t: float = 0


    # create additional features
    def __post_init__(self):
        # new intrinsic attributes
        self.creation_T = self.arrival_T = self.env.now
        self.available_T = self.env.now
        # re-order the processing time by the operatrions
        _pt_by_ops = self.pt_by_m_idx[self.trajectory]
        if self.pt_cv == 0:
            self.remaining_pt = list(_pt_by_ops)
            self.actual_remaining_pt = list(_pt_by_ops) # a stack of actual processing time, equals expected pt
        else:
            self.remaining_pt = list(_pt_by_ops)
            _actual_pt = np.around(np.random.normal(_pt_by_ops, _pt_by_ops*self.pt_cv), decimals=1).clip(*self.pt_range)
            self.actual_remaining_pt = list(_actual_pt) # a stack of actual processing time, different from expected pt
        # a stack of machine indices
        self.remaining_machines = list(self.trajectory) # a stack of machine index that job needs to visit
        # zip remaining machines, expected pt, and actual pt
        self.remaining_operations = list(zip(self.remaining_machines, self.remaining_pt, self.actual_remaining_pt))
        # produce due date for job, which is proportional to the total processing time
        self.due = np.round(self.pt_by_m_idx.sum() * np.random.uniform(1.2, self.due_tightness) + self.env.now)        
        # data recording
        self.operation_record = []
        self.logger.info("{} >>> JOB {} created, trajectory: {}, exp.pt: {}, actual pt: {}, due: {}".format(
            self.env.now, self.j_idx, self.trajectory, self.remaining_pt, self.actual_remaining_pt, self.due))


    def before_operation(self):
        self.arrival_T = self.env.now


    # update the information, get ready for transfer or exit
    def after_operation(self, *args):
        # if job is not completed
        if len(self.remaining_operations) > 1:
            self.logger.debug('Job {} is tagged as "queuing"'.format(self.j_idx))
            self.status = "queuing"
            # pop current operation tuple
            self.remaining_operations.pop(0)
            # and individual lists
            self.remaining_machines.pop(0)
            self.remaining_pt.pop(0)
            self.actual_remaining_pt.pop(0)
            # retrieve machine index from next operation
            _next_m = self.remaining_machines[0]
            return _next_m
        else:
            self.status = "completed"
            self.completion()
            return -1


    def record_operation(self, *args):
        self.operation_record.append([*args])


    # all operations are complete and exit the system
    def completion(self):
        # append the operation histroy to the recorder
        self.recorder.j_operation_dict[self.j_idx] = self.operation_record
        self.recorder.j_tardiness_dict[self.j_idx] = max(0, self.env.now - self.due)
        self.recorder.j_flowtime_dict[self.j_idx] = self.env.now - self.creation_T
        self.recorder.in_system_jobs.pop(self.j_idx)
        self.logger.info("{} >>> JOB {} completed".format(self.env.now, self.j_idx))
        self.tardiness = self.env.now - self.due