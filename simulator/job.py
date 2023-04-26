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
        self.logger.info("JOB {} created at time: {}, trajectory: {}, exp.pt: {}, due: {}".format(self.j_idx, self.env.now, self.trajectory , self.remaining_pt, self.due))


    # record the details of this operation, and join next machine's queue
    def after_operation(self, details):
        # check if this is the last operation of job
        # if the sequence is not empty, any value > 0 is True
            #print('OPERATION: Job %s output from machine %s at time %s'%(self.queue[self.position], self.m_idx, self.env.now))
            next_wc = self.sequence_list[self.position][0]
            # add the job to next work center's queue
            self.wc_list[next_wc].queue.append(self.queue.pop(self.position))
            # add the information of this job to next work center's storage
            self.wc_list[next_wc].sequence_list.append(np.delete(self.sequence_list.pop(self.position),0))
            self.wc_list[next_wc].pt_list.append(self.pt_list.pop(self.position))
            # get the expected processing time of remaining processes
            remaining_ptl = self.remaining_pt_list.pop(self.position)
            # and activate the dispatching of next work center
            try:
                self.wc_list[next_wc].routing_event.succeed()
            except:
                pass
            # after transfered the job, update information of queuing jobs
            self.state_update_all()
            # clear some global information
            self.update_global_info_after_operation()
            # check if sequencing learning mode is on, and queue is not 0
            if self.routing_learning_event.triggered:
                try:
                    self.wc.build_routing_experience(self.job_idx,self.slack_change, self.critical_level_R)
                except:
                    pass
            if self.sequencing_learning_event.triggered:
                self.complete_experience


    # all operations are complete and exit the system
    def completion(self):
        self.tardiness = self.env.now - self.due



    def __del__(self):
        self.logger.info("Completion, job {} exits system at time {}".format(self.j_idx, self.env.now))