import numpy as np
from tabulate import tabulate
from job import *
from sequencing_rule import *
from opt_scheduler import *

'''
The module that creates dynamic events 
Able to simulate job arrival/cancellation, machine breakdown, processing time variability, etc.
'''

class Narrator:
    def __init__(self, **kwargs):
        '''
        0. Shared features
        '''
        for k, v in kwargs.items():
            setattr(self, k, v)
        self.kwargs = kwargs
        self.logger.debug("Event narrator created")
        # look for random seed
        if 'seed' in kwargs:
            np.random.seed(kwargs['seed'])
            self.logger.debug("Random seed is specified, seed: {}".format(kwargs['seed']))
        else:
            self.logger.warning("Random seed is not specified, do this only for training!")
        '''
        1.1 Core components: machines and dynamic job arrivals
        '''
        self.m_no = len(self.m_list) # related to the number of operations
        _E_pt = np.average(self.pt_range) # expected processing time of individual operations
        self.j_idx = 0 # job index, start from 0
        # produce the feature of new job arrivals by Poison distribution
        # draw the time interval betwen job arrivals from an exponential distribution
        # The mean of an exp random variable X with rate parameter λ is given by:
        # 1/λ (which equals the term "beta" in np exp function)
        _beta = _E_pt / self.E_utliz # beta is the average time interval between job arrivals
        self.logger.debug("The expected utilization rate (excluding machine down time) is: [{}%]".format(self.E_utliz*100))
        self.logger.debug("Converted expected interval between job arrival is: [{}] (m_no: [{}], pt_range: {}, exp_pt: [{}])".format(_beta, self.m_no, self.pt_range, _E_pt))
        # number of new jobs arrive within simulation, with 10% extra jobs as buffer
        self.total_no = np.round(1.1*self.span/_beta).astype(int)
        # the interval between job arrivals by exponential distribution
        self.arrival_interval = np.random.exponential(_beta, self.total_no).round()
        # process the job arrival function
        self.env.process(self.process_job_creation())
        ''' 
        1.2 Machine initialization: knowing each other and specify the sequencing rule
        '''
        self.opt_mode = False
        if 'sqc_rule' in kwargs:
            if kwargs['sqc_rule'] == 'complete_schedule': # follow a complete schedule
                pass
                #self.job_sequencing_func = complete_schedule.who_is_next()
            elif kwargs['sqc_rule'] == SQC_rule.opt_scheduler: # or using mathematical optimization to produce dynamic schedule
                self.central_scheduler = OPT_scheduler(**self.kwargs)
                self.opt_mode = True
                job_sequencing_func = self.central_scheduler.draw_from_schedule
                self.logger.info("* Optimization mode is ON, A [centralized Gurobi scheduler] is created, all machines use a central schedule")
            else: # otherwise a valid sequencing rule must be specified
                try:
                    job_sequencing_func = kwargs['sqc_rule']
                    self.logger.info("* Machine use [{}] sequencing rule".format(job_sequencing_func.__name__))
                except Exception as e:
                    self.logger.error("Sequencing rule is invalid! Invalid entry: [{}]".format(kwargs['sqc_rule']))
                    self.logger.error(str(e))
                    raise Exception
        else:
            # if no argument is given, default sequencing rule is FIFO
            self.logger.info("* Machine {} uses default FIFO rule".format(self.m_idx))
            job_sequencing_func = SQC_rule.FIFO
        # initialization, let all machines know each other and pass the sqc rule to them
        for m in self.m_list:
            m.initialization(machine_list = self.m_list, sqc_rule = job_sequencing_func)
        '''
        2. Optional part I: machine breakdown
        '''
        if self.machine_breakdown == True:
            for m_idx, m in enumerate(self.m_list):
                self.env.process(self.process_machine_breakdown(m_idx, kwargs['random_bkd']))
            self.logger.debug("Machine breakdown mode is ON, MTBF: [{}], MTTR: [{}]".format(self.MTBF, self.MTTR))
        '''
        3. Optional part II: processing time variablity
        '''
        if kwargs['processing_time_variability'] and kwargs['pt_cv'] > 0:
            self.pt_cv = kwargs['pt_cv']
            self.logger.debug("Variable processing time mode is ON, coefficient of variance: [{}]".format(kwargs['pt_cv']))
        else:
            self.pt_cv = 0


    # continuously creating new jobs
    def process_job_creation(self):
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
                env = self.env, logger = self.logger, recorder = self.recorder,
                j_idx = self.j_idx, trajectory = trajectory_seed.copy(), pt_by_m_idx = ptl.copy(),
                pt_range = self.pt_range, pt_cv = self.pt_cv, due_tightness = self.due_tightness)
            # track thjis job
            self.recorder.in_system_jobs[self.j_idx] = job_instance
            # build a new schedule if optimization mode is on
            if self.opt_mode:
                self.logger.debug("New job arrived, call central scheduler to build schedule")
                self.central_scheduler.solve_problem()
            # after creating a job, assign it to the first machine along its trajectory
            first_m = trajectory_seed[0]
            self.m_list[first_m].job_arrival(job_instance)
            # update the index of job
            self.j_idx += 1


    # periodicall disable machines
    def process_machine_breakdown(self, m_idx:int, random_bkd:bool):
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
            self.m_list[m_idx].release_time = actual_begin + self.MTTR
            # time of breakdown
            yield self.env.timeout(actual_end - self.env.now)
            self.recorder.m_bkd_dict[m_idx].append([actual_begin, actual_end])
            self.m_list[m_idx].working_event.succeed()

    
    def post_simulation(self):
        self.logger.info('Simulation Ended, here is the shopfloor configuration:\n\n{}\n'.format(
            tabulate([["Category","Number", "Attributes"],
                      ["Machine", self.m_no, "(1) Machine Breakdown: {}; (2) Random bkd: {}".format(self.kwargs['machine_breakdown'], self.kwargs['random_bkd'])],
                      ["Job", self.j_idx, "(1) pt range: {}; (2) pt cv: {}".format(self.pt_range, self.pt_cv)]],
                      headers="firstrow")))


    def build_sqc_experience_repository(self, m_list): # build two dictionaries
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


# Retain all records
class Recorder:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # record the job's journey
        self.in_system_jobs = {}
        self.j_arrival_dict = {}
        self.j_departure_dict = {}
        self.j_op_dict = {}
        self.m_bkd_dict = {idx: [] for idx in range(kwargs['m_no'])}
        self.m_cum_runtime_dict = {}
        self.pt_mean_dict = {}
        self.pt_std_dict = {}
        self.expected_tardiness_dict = {}
