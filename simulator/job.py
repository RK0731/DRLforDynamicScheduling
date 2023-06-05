"""
This is the job class, carries the information of trajectory, processing time, due date, etc.
"""

import numpy as np


class Job:
    def __init__(self, *args, **kwargs):
        # all inherited attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # new intrinsic attributes
        self.creation_t = self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.trajectory = kwargs['trajectory']
        self.pt_by_m_idx = kwargs['processing_time_list'] # processing time ordered by machine index #1, 2, ... N, but by operations
        # processing time seed
        seed = self.pt_by_m_idx[self.trajectory]
        if kwargs['pt_cv'] == 0:
            remaining_pt = list(seed)
            actual_remaining_pt = list(seed) # a stack of actual processing time, equals expected pt
        else:
            remaining_pt = list(seed)
            actual_pt = np.around(np.random.normal(seed, seed*kwargs['pt_cv']), decimals=1).clip(*kwargs['pt_range'])
            actual_remaining_pt = list(actual_pt) # a stack of actual processing time, different from expected pt
        # a stack of machine indices
        remaining_machines = list(self.trajectory) # a stack of machine index that job needs to visit
        # zip remaining machines, expected pt, and actual pt
        self.remaining_operations = list(zip(remaining_machines, remaining_pt, actual_remaining_pt))
        # produce due date for job, which is proportional to the total processing time
        self.due = np.round(self.pt_by_m_idx.sum() * np.random.uniform(1.2, kwargs['due_tightness']) + self.env.now)        
        # optional attributes
        if 'transfer_time' in kwargs:
            self.transfer_t = kwargs['transfer_time']
        # data recording
        self.operation_record = []
        self.logger.info("{} >>> JOB {} created, trajectory: {}, exp.pt: {}, actual pt: {}, due: {}".format(
            self.env.now, self.j_idx, self.trajectory , remaining_pt, actual_remaining_pt, self.due))


    def before_operation(self):
        self.arrival_t = self.env.now


    # update the information, get ready for transfer or exit
    def after_operation(self, *args):
        # if job is not completed
        if len(self.remaining_operations) > 1:
            # pop current operation
            self.remaining_operations.pop(0)
            # retrieve machine index from next operation
            next = self.remaining_operations[0][0]
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


