"""
This is the job object, carries the information of route, processing time, due date, etc.
"""

import numpy as np


class Job:
    def __init__(self, env, logger, *args, **kwargs):
        self.env = env
        self.logger = logger
        # associated attributes
        self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.pt = kwargs['processing_time']
        self.due = kwargs['due_date']
        # optional attributes
        if 'transfer_time' in kwargs:
            self.transfer_t = kwargs['transfer_time']
        # data recording
        self.processing_record = []


    def arrival(self, machine):
        pass


    def departure(self):
        pass


    def record_operation(self, details):
        self.processing_record.append(details)


    def completion(self):
        self.tardiness = self.env.now - self.due