import numpy as np
import random
import os
from tabulate import tabulate
import matplotlib.pyplot as plt
from job import Job
from data_recorder import Recorder

'''
The module that creates dynamic events 
Able to simulate job arrival/cancellation, machine breakdown, processing time variability, etc.
'''

class Director:
    def __init__(self, env, logger, span:int, **kwargs
            ):
        '''
        0. Important features
        '''
        self.env = env
        self.logger = logger
        self.span = span
        # look for random seed
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            self.logger.info("Random seed is specified, seed: {}".format(kwargs['seed']))
        else:
            self.logger.warning("Random seed is not specified, should be only used for the training!")
        '''
        1. Essential part: machines and dynamic job arrivals
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
        # number of new jobs arrive within simulation, with 10% extra jobs as buffer
        self.total_no = np.round(1.1*self.span/self.beta).astype(int)
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        # process the job arrival function
        self.env.process(self.new_job_arrival())
        '''
        2. Optional part I: machine breakdown
        '''
        if kwargs['machine_breakdown'] == True:
            self.env.process(self.machine_breakdown())
        '''
        3. Optional part II: processing time variablity
        '''
        if kwargs['processing_time_variability'] == True:
            self.pt_var = kwargs['pt_variance']
        else:
            self.pt_var = 0
        # initialize the information associated with jobs that are being processed
        # note that the updates to these data are initiated by job or machine instances
        self.available_time_list = np.array([0 for m in self.m_list]) 
        self.release_time_list = np.array([self.exp_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.m_no)
        self.next_machine_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.exp_pt for m in self.m_list])
        self.arriving_job_ttd_list = np.array([self.exp_pt * self.no_machines for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        '''
        4. Data recorder
        '''
        self.recorder = Recorder()


    def new_job_arrival(self):
        # jobs are assumed to go through all machines
        trajectory_seed = np.arange(self.no_machines)
        while self.env.now < self.span:
            # draw the interval from pre-produced list
            time_interval = self.arrival_interval[self.j_idx]
            yield self.env.timeout(time_interval)
            # produce the trajectory of job, by shuffling the sequence seed
            np.random.shuffle(trajectory_seed)
            # produce processing time of job
            ptl = np.random.randint(*self.pt_range, size = [self.no_machines])
            job_instance = Job(
                self.env, self.logger, self.recorder,
                job_index = self.j_idx, trajectory = trajectory_seed.copy(), processing_time_list = ptl.copy(), tightness = self.tightness
                )
            # after creating a job, assign it to the first machine along its trajectory\
            first_m = trajectory_seed[0]
            self.m_list[first_m].arrival(job_instance)
            # update the index of job
            self.j_idx += 1




    def build_sqc_experience_repository(self,m_list): # build two dictionaries
        self.incomplete_rep_memo = {}
        self.rep_memo = {}
        for m in m_list: # each machine will have a list in the dictionaries
            self.incomplete_rep_memo[m.m_idx] = {} # used for storing s0 and a0
            self.rep_memo[m.m_idx] = [] # after the transition and reward is given, store r0 and s1, complete the experience

    def complete_experience(self, m_idx, decision_point, r_t): # turn incomplete experience to complete experience
        self.incomplete_rep_memo[m_idx][decision_point] += [r_t]
        complete_exp = self.incomplete_rep_memo[m_idx].pop(decision_point)
        self.rep_memo[m_idx].append(complete_exp)
        self.reward_record[m_idx][0].append(self.env.now)
        self.reward_record[m_idx][1].append(r_t)

