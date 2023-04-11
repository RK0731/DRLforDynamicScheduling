import numpy as np
import random
import os
from tabulate import tabulate
import matplotlib.pyplot as plt

'''
The module that creates dynamic events 
Able to simulate job arrival/cancellation, machine breakdown, processing time variability, etc.
'''

class Director:
    def __init__ (self, env, logger, span:int, **kwargs
            ):
        # important features
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
        Essential part: machines and dynamic job arrivals
        '''
        self.m_list = kwargs['machine_list'] # get a list of all machines
        self.m_no = len(self.m_list) # related to the number of operations
        self.pt_range = kwargs['pt_range'] # lower and upper bound of processing time
        self.exp_pt = np.average(self.pt_range) # expected processing time of individual operations
        # variables to track the job related system status
        self.in_system_job_no = 0
        self.j_idx = 0
        self.tightness = kwargs['due_tightness'] # due date tightness
        self.E_utliz = kwargs['E_utliz'] # expected utlization rate of machines
        # produce the feature of new job arrivals
        self.beta = self.exp_pt / self.E_utliz # beta is the average time interval between job arrivals
        # number of new jobs arrive within simulation
        self.total_no = np.round(self.span/self.beta).astype(int)
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(self.beta, self.total_no).round()
        self.initial_job_assignment()
        ### process the job arrival function
        self.env.process(self.new_job_arrival())
        '''
        Optional part I: machine breakdown
        '''
        if kwargs['machine_breakdown'] == True:
            self.env.process(self.machine_breakdown())
        '''
        Optional part II: processing time variablity
        '''
        if kwargs['processing_time_variability'] == True:
            self.pt_var = kwargs['pt_variance']
        else:
            self.pt_var = 0
        # initialize the information associated with jobs that are being processed
        # note that the updates to these data are initiated by job or machine instances
        self.available_time_list = np.array([0 for m in self.m_list]) 
        self.release_time_list = np.array([self.exp_pt for m in self.m_list])
        self.current_j_idx_list = np.arange(self.no_machines)
        self.next_machine_list = np.array([-1 for m in self.m_list])
        self.next_pt_list = np.array([self.exp_pt for m in self.m_list])
        self.arriving_job_ttd_list = np.array([self.exp_pt * self.no_machines for m in self.m_list])
        self.arriving_job_rempt_list = np.array([0 for m in self.m_list])
        self.arriving_job_slack_list = np.array([0 for m in self.m_list])
        # and create an empty, initial array of sequence
        self.sequence_list = []
        self.pt_list = []
        self.remaining_pt_list = []
        self.create_time = []
        self.due_list = []
        self.schedule = []
        # record the rewards that agents received
        self.reward_record = {}
        for m in self.m_list:
            self.reward_record[m.m_idx] = [[],[]]
        # record the arrival and departure information
        self.arrival_dict = {}
        self.departure_dict = {}
        self.mean_dict = {}
        self.std_dict = {}
        self.expected_tardiness_dict = {}


    def new_job_arrival(self):
        # main process
        sqc_seed = np.arange(self.no_machines)
        while self.j_idx < self.total_no:
            # draw the time interval betwen job arrivals from exponential distribution
            # The mean of an exp random variable X with rate parameter λ is given by:
            # 1/λ (which equals the term "beta" in np exp function)
            time_interval = self.arrival_interval[self.j_idx]
            yield self.env.timeout(time_interval)
            # produce sequence of job, first shuffle the sequence seed
            np.random.shuffle(sqc_seed)
            self.sequence_list.append(sqc_seed.copy())
            # produce processing time of job, get corresponding remaining_pt_list
            ptl = np.random.randint(self.pt_range[0], self.pt_range[1], size = [self.no_machines])
            self.pt_list.append(ptl)
            self.record_job_feature(self.j_idx,ptl)
            # rearrange the order of ptl to get remaining pt list, so we can simply delete the first element after each stage of production
            remaining_ptl = ptl[sqc_seed]
            self.remaining_pt_list.append(remaining_ptl.copy())
            # produce due date for job
            exp_pt = ptl.mean()
            due = np.round(exp_pt*self.no_machines*np.random.uniform(1, self.tightness) + self.env.now)
            # record the creation time
            self.create_time.append(self.env.now)
            self.due_list.append(due)
            # add job to system and create the data repository for job
            self.record_job_arrival()
            # operation record, path, wait time, decision points, slack change
            self.production_record[self.j_idx] = [[],[],[],[]]
            '''after creation of new job, add it to machine'''
            # first machine of that job
            first_m = sqc_seed[0]
            # add job to machine
            self.m_list[first_m].queue.append(self.j_idx)
            self.m_list[first_m].sequence_list.append(np.delete(sqc_seed,0)) # the added sequence is the one without first element, coz it's been dispatched
            self.m_list[first_m].remaining_pt_list.append(remaining_ptl)
            self.m_list[first_m].due_list.append(due)
            self.m_list[first_m].slack_upon_arrival.append(due - self.env.now - remaining_ptl.sum())
            self.m_list[first_m].arrival_time_list.append(self.env.now)
            # and update some information
            self.m_list[first_m].state_update_after_job_arrival(remaining_ptl[0])
            # after assigned the nwe job to machine, try activate its sufficient stock event
            try:
                self.m_list[first_m].sufficient_stock.succeed()
            except:
                pass
            self.j_idx += 1 # and update the index of job
            self.logger.info("JOB {} arrival at time: {}, trajectory: {}, pt: {}, due: {}"%(self.j_idx, self.env.now, sqc_seed , ptl, due))

    def dynamic_seed_change(self, interval):
        while self.env.now < self.span:
            yield self.env.timeout(interval)
            seed = np.random.randint(2000000000)
            np.random.seed(seed)
            print('change random seed to {} at time {}'.format(seed,self.env.now))

    def change_setting(self,pt_range):
        print('Heterogenity changed at time',self.env.now)
        self.pt_range = pt_range
        self.exp_pt = np.average(self.pt_range)-0.5
        self.beta = self.exp_pt / (2*self.E_utliz)

    def get_global_exp_tard_rate(self):
        x = []
        for m in self.m_list:
            x = np.append(x, m.slack)
        rate = x[x<0].size / x.size
        return rate

    # this fucntion record the time and number of new job arrivals
    def record_job_arrival(self):
        self.in_system_job_no += 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.arrival_dict[self.env.now] += 1
        except:
            self.arrival_dict[self.env.now] = 1

    # this function is called upon the completion of a job, by machine agent
    def record_job_departure(self):
        self.in_system_job_no -= 1
        self.in_system_job_no_dict[self.env.now] = self.in_system_job_no
        try:
            self.departure_dict[self.env.now] += 1
        except:
            self.departure_dict[self.env.now] = 1

    def record_job_feature(self,idx,ptl):
        self.mean_dict[idx] = (self.env.now, ptl.mean())
        self.std_dict[idx] = (self.env.now, ptl.std())

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

    def initial_output(self):
        print('job information are as follows:')
        job_info = [[i,self.sequence_list[i], self.pt_list[i], \
        self.create_time[i], self.due_list[i]] for i in range(self.j_idx)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due']))
        print('--------------------------------------')
        return job_info

    def final_output(self):
        # information of job output time and realized tardiness
        output_info = []
        print(self.production_record)
        for item in self.production_record:
            output_info.append(self.production_record[item][4])
        job_info = [[i,self.sequence_list[i], self.pt_list[i], self.create_time[i],\
        self.due_list[i], output_info[i][0], output_info[i][1]] for i in range(self.j_idx)]
        print(tabulate(job_info, headers=['idx.','sqc.','proc.t.','in','due','out','tard.']))
        realized = np.array(output_info)[:,1].sum()
        exp_tard = sum(self.expected_tardiness_dict.values())

    def tardiness_output(self):
        # information of job output time and realized tardiness
        tard_info = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard_info.append(self.production_record[item][4])
        # now tard_info is an ndarray of objects, cannot be sliced. need covert to common np array
        # if it's a simple ndarray, can't sort by index
        dt = np.dtype([('output', float),('tardiness', float)])
        tard_info = np.array(tard_info, dtype = dt)
        tard_info = np.sort(tard_info, order = 'output')
        # now tard_info is an ndarray of objects, cannot be sliced, need covert to common np array
        tard_info = np.array(tard_info.tolist())
        tard_info = np.array(tard_info)
        output_time = tard_info[:,0]
        tard = np.absolute(tard_info[:,1])
        cumulative_tard = np.cumsum(tard)
        tard_max = np.max(tard)
        tard_mean = np.cumsum(tard) / np.arange(1,len(cumulative_tard)+1)
        tard_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return output_time, cumulative_tard, tard_mean, tard_max, tard_rate

    def record_printout(self):
        print(self.production_record)

    def timing_output(self):
        return self.arrival_dict, self.departure_dict, self.in_system_job_no_dict

    def feature_output(self):
        return self.mean_dict, self.std_dict

    def reward_output(self, m_idx):
        plt.scatter(np.array(self.reward_record[m_idx][0]), np.array(self.reward_record[m_idx][1]), s=3, c='r')
        plt.show()
        return

    def all_tardiness(self):
        # information of job output time and realized tardiness
        tard = []
        #print(self.production_record)
        for item in self.production_record:
            #print(item,self.production_record[item])
            tard.append(self.production_record[item][4][1])
        #print(tard)
        tard = np.array(tard)
        mean_tardiness = tard.mean()
        tardy_rate = tard.clip(0,1).sum() / tard.size
        #print(output_time, cumulative_tard, tard_mean)
        return mean_tardiness, tardy_rate