"""
This is the job class, carries the information of trajectory, processing time, due date, etc.
"""

import numpy as np


class Job:
    def __init__(self, *args, **kwargs):
        # user specified attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # new intrinsic attributes
        self.creation_t = self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.trajectory = kwargs['trajectory']
        self.pt_by_m_idx = kwargs['processing_time_list'] # processing time ordered by machine index #1, 2, ... N, but by operations
        # dynamic stack to store incomplete operations' processing time, variance may apply
        seed = self.pt_by_m_idx[self.trajectory]
        if kwargs['pt_cv'] == 0:
            self.remaining_pt = list(seed)
            self.actual_remaining_pt = list(seed)
        else:
            self.remaining_pt = list(seed)
            actual_pt = np.around(np.random.normal(seed, seed*kwargs['pt_cv']), decimals=1).clip(*kwargs['pt_range'])
            self.actual_remaining_pt = list(actual_pt)
        # a stack of machine indices
        self.remaining_m = list(self.trajectory)
        # produce due date for job, which is proportional to the total processing time
        self.due = np.round(self.pt_by_m_idx.sum() * np.random.uniform(1.2, kwargs['due_tightness']) + self.env.now)        
        # optional attributes
        if 'transfer_time' in kwargs:
            self.transfer_t = kwargs['transfer_time']
        # data recording
        self.operation_record = []
        self.logger.info("{} >>> JOB {} created, trajectory: {}, exp.pt: {}, actual pt: {}, due: {}".format(
            self.env.now, self.j_idx, self.trajectory , self.remaining_pt, self.actual_remaining_pt, self.due))


    def before_operation(self):
        self.arrival_t = self.env.now


    # update the information, get ready for transfer or exit
    def after_operation(self, *args):
        # if job is not completed
        if len(self.remaining_m) > 1:
            self.remaining_pt.pop(0)
            self.actual_remaining_pt.pop(0)
            self.remaining_m.pop(0)
            next = self.remaining_m[0] # next station's index
            return next
        else:
            self.completion()
            # append the operation histroy to the recorder
            self.recorder.j_op_dict[self.j_idx] = self.operation_record
            return -1


    def record_operation(self, *args):
        self.operation_record.append([*args])


    # all operations are complete and exit the system
    def completion(self):
        self.logger.info("{} >>> JOB {} completed".format(self.env.now, self.j_idx))
        self.tardiness = self.env.now - self.due


