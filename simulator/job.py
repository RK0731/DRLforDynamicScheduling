"""
This is the job object, carries the information of route, processing time, due date, etc.
"""

import numpy as np


class Job:
    def __init__(self, env, *args, **kwargs):
        self.env = env
        # associated attributes
        self.arrival_t = self.env.now
        self.j_idx = kwargs['job_index']
        self.pt = kwargs['processing_time']
        self.due = kwargs['due_date']
        # data recording
        self.processing_record = []


    def departure(self):
        pass


    def completion(self):
        self.tardiness = self.env.now - self.due