import numpy as np
import random
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from job import *

'''
The module that creates dynamic events 
Able to simulate job arrival/cancellation, machine breakdown, processing time variability, etc.
'''

class Narrator:
    def __init__(self, env, logger, recorder, **kwargs):
        '''
        0. Shared features
        '''
        self.env = env
        self.logger = logger
        self.recorder = recorder
        self.span = kwargs['span']
        self.kwargs = kwargs
        self.logger.debug("Event narrator created")
        # look for random seed
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            self.logger.debug("Random seed is specified, seed: {}".format(kwargs['seed']))
        else:
            self.logger.warning("Random seed is not specified, do this only for training!")
        '''
        1. Compulsory part: machines and dynamic job arrivals
        '''
        self.m_list = kwargs['machine_list'] # get the list of all machines
        self.m_no = len(self.m_list) # related to the number of operations
        self.pt_range = kwargs['pt_range'] # lower and upper bound of processing time
        self.exp_pt = np.average(self.pt_range) # expected processing time of individual operations
        # variables to track the job related system status
        self.in_system_job_no = 0
        self.j_idx = 0
        self.tightness = kwargs['due_tightness'] # due date tightness
        self.E_utliz = kwargs['E_utliz'] # expected utlization rate of machines
        # produce the feature of new job arrivals by Poison distribution
        # draw the time interval betwen job arrivals from an exponential distribution
        # The mean of an exp random variable X with rate parameter λ is given by:
        # 1/λ (which equals the term "beta" in np exp function)
        self.beta = self.exp_pt / self.E_utliz # beta is the average time interval between job arrivals
        self.logger.debug("The expected utilization rate (excluding machine down time) is: {}%".format(self.E_utliz*100))
        self.logger.debug("Converted expected interval between job arrival is: {} (m_no: {}, pt_range: {}, exp_pt: {})".format(self.beta, self.m_no, self.pt_range, self.exp_pt))
        # number of new jobs arrive within simulation, with 10% extra jobs as buffer
        self.total_no = np.round(1.1*self.span/self.beta).astype(int)
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        # process the job arrival function
        self.env.process(self.job_creation())
        '''
        2. Optional part I: machine breakdown
        '''
        if kwargs['machine_breakdown'] == True:
            self.MTBF = kwargs['MTBF']
            self.MTTR = kwargs['MTTR']
            for m_idx, m in enumerate(self.m_list):
                self.env.process(self.machine_breakdown(m_idx, kwargs['random_bkd']))
            self.logger.debug("Machine breakdown mode is ON, MTBF: {}, MTTR: {}".format(self.MTBF, self.MTTR))
        '''
        3. Optional part II: processing time variablity
        '''
        if kwargs['processing_time_variability'] and kwargs['pt_cv'] > 0:
            self.pt_cv = kwargs['pt_cv']
            self.logger.debug("Variable processing time mode is ON, coefficient of variance: {}".format(kwargs['pt_cv']))
        else:
            self.pt_cv = 0
        # initialize the information associated with jobs that are being processed
        # note that the updates to these data are initiated by job or machine instances
        self.available_time_list = np.array([0 for m in self.m_list]) 
        self.release_time_list = np.array([self.exp_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.m_no)
        self.next_machine_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.exp_pt for m in self.m_list])
        self.arriving_job_ttd_list = np.array([self.exp_pt * self.m_no for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])


    # continuously creating new jobs
    def job_creation(self):
        # jobs are assumed to go through all machines
        trajectory_seed = np.arange(self.m_no)
        while self.j_idx < self.total_no:
            # draw the interval from pre-produced list
            time_interval = self.arrival_interval[self.j_idx]
            yield self.env.timeout(time_interval)
            # produce the trajectory of job, by shuffling the sequence seed
            np.random.shuffle(trajectory_seed)
            # randomly a produce processing time array of job, this is THEORATICAL value, not actual value if variance exists
            ptl = np.random.randint(self.pt_range[0], self.pt_range[1]+1, size = [self.m_no])
            # new job instance
            job_instance = Job(
                self.env, self.logger, self.recorder,
                job_index = self.j_idx, trajectory = trajectory_seed.copy(), processing_time_list = ptl.copy(),
                pt_range = self.pt_range, pt_cv = self.pt_cv, tightness = self.tightness)
            # after creating a job, assign it to the first machine along its trajectory
            first_m = trajectory_seed[0]
            self.m_list[first_m].job_arrival(job_instance)
            # update the index of job
            self.j_idx += 1


    # periodicall disable machines
    def machine_breakdown(self, m_idx, random_bkd):
        while self.env.now < self.span:
            # draw the time interval between two break downs
            if random_bkd:
                time_interval = np.around(np.random.exponential(self.MTBF), decimals = 1)
                bkd_t = np.around(np.random.uniform(self.MTTR*0.5, self.MTTR*1.5), decimals = 1)
            else:
                time_interval = self.MTBF
                bkd_t = self.MTTR
            yield self.env.timeout(time_interval)           
            self.m_list[m_idx].working_event = self.env.event()
            self.logger.debug("{} >>> BKD created, mahcine {} will be down for {}".format(self.env.now, m_idx, bkd_t))
            # if machine is currently running, the breakdown will commence after current operation
            actual_begin = max(self.m_list[m_idx].release_time, self.env.now)
            actual_end = actual_begin + bkd_t
            yield self.env.timeout(actual_end - self.env.now)
            self.recorder.m_bkd_dict[m_idx].append([actual_begin, actual_end])
            self.m_list[m_idx].working_event.succeed()


    def build_sqc_experience_repository(self,m_list): # build two dictionaries
        self.incomplete_experience = {}
        self.rep_memo = {}
        for m in m_list: # each machine will have a list in the dictionaries
            self.incomplete_experience[m.m_idx] = {} # used for storing s0 and a0
            self.rep_memo[m.m_idx] = [] # after transition, store r0 and s1, complete the experience


    def complete_experience(self, m_idx, decision_point, r_t): # turn incomplete experience to complete experience
        self.incomplete_experience[m_idx][decision_point] += [r_t]
        complete_exp = self.incomplete_experience[m_idx].pop(decision_point)
        self.rep_memo[m_idx].append(complete_exp)
        self.reward_record[m_idx][0].append(self.env.now)
        self.reward_record[m_idx][1].append(r_t)


# Retains all records
class Recorder:
    def __init__(self, **kwargs):
        self.m_no = kwargs['m_no']
        self.span = kwargs['span']
        # record the job's journey
        self.j_arrival_dict = {}
        self.j_departure_dict = {}
        self.j_op_dict = {}
        self.m_bkd_dict = {idx: [] for idx in range(kwargs['m_no'])}
        self.m_cum_runtime_dict = {}
        self.pt_mean_dict = {}
        self.pt_std_dict = {}
        self.expected_tardiness_dict = {}