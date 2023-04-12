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
        self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.trajectory = kwargs['trajectory']
        self.pt_by_machine = kwargs['processing_time_list'] # processing time here corresponds to machines, but by operations
        # reorder the processing time list to get remaining pt list
        # so we can simply delete the first element after each operation
        self.remaining_pt = self.pt_by_machine[self.trajectory]
        # produce due date for job, which is proportional to the total processing time
        self.due = np.round(self.pt_by_machine.sum() * np.random.uniform(1.2, kwargs['tightness']) + self.env.now)        
        # optional attributes
        if 'transfer_time' in kwargs:
            self.transfer_t = kwargs['transfer_time']
        # data recording
        self.processing_record = []
        self.logger.info("JOB {} created at time: {}, trajectory: {}, pt: {}, due: {}".format(self.j_idx, self.env.now, self.trajectory , self.remaining_pt, self.due))


    def record_operation(self, details):
        self.processing_record.append(details)


    def completion(self):
        self.tardiness = self.env.now - self.due


    def __del__(self):
        pass