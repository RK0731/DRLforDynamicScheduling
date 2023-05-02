"""
This is the job object, carries the information of route, processing time, due date, etc.
"""

import numpy as np


class Job:
    def __init__(self, env, logger, recorder, *args, **kwargs):
        self.env = env
        self.logger = logger
        self.recorder = recorder
        # associated attributes
        self.creation_t = self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.trajectory = kwargs['trajectory']
        self.pt_by_m_idx = kwargs['processing_time_list'] # processing time here corresponds to machines, but by operations
        # dynamic list to store incomplete operations
        self.remaining_pt = list(self.pt_by_m_idx[self.trajectory])
        self.remaining_m = list(self.trajectory)
        # produce due date for job, which is proportional to the total processing time
        self.due = np.round(self.pt_by_m_idx.sum() * np.random.uniform(1.2, kwargs['tightness']) + self.env.now)        
        # optional attributes
        if 'transfer_time' in kwargs:
            self.transfer_t = kwargs['transfer_time']
        # data recording
        self.operation_record = []
        self.logger.info("JOB {} created at time: {}, trajectory: {}, exp.pt: {}, due: {}".format(self.j_idx, self.env.now, self.trajectory , self.remaining_pt, self.due))


    def before_operation(self):
        self.arrival_t = self.env.now


    # update the information, get ready for transfer or exit
    def after_operation(self, *args):
        # if job is not completed
        if len(self.remaining_m) > 1:
            self.remaining_pt.pop(0)
            self.remaining_m.pop(0)
            next = self.remaining_m[0] # next station's index
            return next
        else:
            self.completion()
            return -1


    def record_operation(self, *args):
        self.operation_record.append([*args])


    # all operations are complete and exit the system
    def completion(self):
        self.logger.info("JOB {} completed at time {}".format(self.j_idx, self.env.now))
        self.tardiness = self.env.now - self.due


    def __del__(self):
        self.logger.debug("JOB {} instance deleted at time {}".format(self.j_idx, self.env.now))
        # append the operation histroy to the recorder
        self.recorder.j_op_dict[self.j_idx] = self.operation_record
