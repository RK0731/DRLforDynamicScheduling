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
        # specify the random seed
        if 'seed' in kwargs:
            self.logger.debug("Random seed is specified, seed: {}".format(kwargs['seed']))
        else:
            self.seed = np.random.randint(0, 1e10)
            self.logger.warning("Random seed is not specified, generated seed: {}, do this only for training!".format(self.seed))
        self.rng = np.random.default_rng(seed = self.seed)
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
        self.arrival_interval = self.rng.exponential(_beta, self.total_no).round()
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
                self.env.process(self.process_machine_breakdown(m_idx, self.random_MTBF, self.random_MTTR))
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
            self.rng.shuffle(trajectory_seed)
            # randomly a produce processing time array of job, this is THEORATICAL value, not actual value if variance exists
            ptl = self.rng.integers(low = self.pt_range[0], high = self.pt_range[1]+1, size = [self.m_no])
            # new job instance
            job_instance = Job(
                env = self.env, logger = self.logger, recorder = self.recorder, rng = self.rng,
                j_idx = self.j_idx, trajectory = trajectory_seed.copy(), pt_by_m_idx = ptl.copy(),
                pt_range = self.pt_range, pt_cv = self.pt_cv, due_tightness = self.due_tightness)
            # track thjis job
            self.recorder.in_system_jobs[self.j_idx] = job_instance
            # build a new schedule if optimization mode is on
            if self.opt_mode:
                self.logger.debug("New job arrived, call central scheduler to build schedule\n"+"-"*88)
                self.central_scheduler.solve_problem()
            # after creating a job, assign it to the first machine along its trajectory
            first_m = trajectory_seed[0]
            self.m_list[first_m].job_arrival(job_instance)
            # update the index of job
            self.j_idx += 1


    # periodicall disable machines
    def process_machine_breakdown(self, m_idx:int, random_MTBF:bool, random_MTTR:bool):
        while self.env.now < self.span:
            # draw the time interval between two break downs and the down time
            if random_MTBF:
                time_interval = np.around(self.rng.exponential(self.MTBF), decimals = 1)
            else:
                time_interval = self.MTBF
            if random_MTTR:
                bkd_t = np.around(self.rng.uniform(self.MTTR*0.5, self.MTTR*1.5), decimals = 1)
            else:
                bkd_t = self.MTTR
            # if machine is currently running, the breakdown will commence after current operation
            # but get the actual beging and end time first
            yield self.env.timeout(time_interval)
            actual_begin = max(self.m_list[m_idx].hidden_release_T, self.env.now)
            actual_end = actual_begin + bkd_t
            # then when we arrive at the actual breakdown time
            yield self.env.timeout(actual_begin - self.env.now)
            self.m_list[m_idx].working_event = self.env.event()
            self.logger.debug("{} > BKD: mahcine {} will be down for {}".format(self.env.now, m_idx, bkd_t))
            # summation of actual begin time and expected down time (MTTR)
            self.m_list[m_idx].release_T = actual_begin + self.MTTR
            if self.opt_mode:
                self.logger.debug("Machine breakdown, call central scheduler to rebuild schedule")
                self.central_scheduler.solve_problem()
            # time of breakdown
            yield self.env.timeout(actual_end - self.env.now)
            self.recorder.m_bkd_dict[m_idx].append([actual_begin, actual_end])
            self.m_list[m_idx].working_event.succeed()

    
    def post_simulation(self):
        # compare the number of completed job and created job
        if len(self.recorder.j_operation_dict) != self.j_idx:
            msg = "Simulation FAILED, not all jobs have successfully complete their operations"
            self.logger.error(msg)
        # compare each operation in schedule and execution
        # mismatch doesn't mean simulation failed, but indicate likely "under-optimization"
        # only meaningful when processing variablity and breakdown time (if any) variablity are 0
        if self.opt_mode and self.pt_cv == 0 and (self.machine_breakdown and self.random_MTTR) == False:
            self.logger.debug("Schedule <-> Execution match check needed")
            _mismatch = {}
            for _j_idx, ops in self.central_scheduler.j_op_by_schedule.items():
                compare = zip(ops, self.recorder.j_operation_dict[_j_idx][-len(ops):])
                for E, A in compare: # E, A are both (m_idx, opBeginT), one is expected, one is actual 
                    if E[1]!=A[1]: 
                        try:
                            _mismatch[_j_idx].append("M{}, ET:{}, AT:{}".format(E[0], E[1], A[1]))
                        except:
                            _mismatch[_j_idx] = ["M{}, ET:{}, AT:{}".format(E[0], E[1], A[1])]
            if len(_mismatch): 
                self.logger.warning("Schedule and execution MISMATCH!:\n{}\n".format(
                    tabulate([["Job", "Mismatch"],
                            *[[_j_idx, description] for _j_idx, description in _mismatch.items()]],
                            headers="firstrow", tablefmt="psql")))
        # simulation configurations to be printed in console
        header = ["Category", "Number", "Attributes"]
        # machine breakdown info
        if self.machine_breakdown:
            m_config = ["Machine", self.m_no, "Machine Breakdown: {}".format(self.machine_breakdown)]
            if self.random_MTBF: 
                m_config[-1] += "\nMTBF: {}, random: {}".format(self.MTBF, self.random_MTBF) 
            else: m_config[-1] += "\nMTBF: {}, deterministic".format(self.MTBF)
            if self.random_MTTR: 
                m_config[-1] += "\nMTTR: {}, random: {}".format(self.MTTR, self.random_MTTR) 
            else: m_config[-1] += "\nMTTR: {}, deterministic".format(self.MTTR)
        else:
            m_config = ["Machine", self.m_no, "Machine Breakdown: False"]
        # job info
        if self.processing_time_variability and self.pt_cv > 0:
            j_config = ["Job", self.j_idx, "pt range: {}\npt cv: {}\ndue tightness: {}".format(self.pt_range, self.pt_cv, self.due_tightness)]
        else:
            j_config = ["Job", self.j_idx, "pt range: {}, deterministic\ndue tightness: {}".format(self.pt_range, self.due_tightness)]            
        # sequencing decision maker
        sqc_config = ['Sqc', "N.A.", self.sqc_rule.__name__]
        # simulation info
        sim_config = ["Sim", "N.A.", "Span: {}\nUtilization rate: {}%\nRandom seed: {} / {}".format(self.span, self.E_utliz*100, self.seed, self.rng)]
        if self.opt_mode:
            tt= time.time()-self.program_start_T
            opt_tt = self.recorder.opt_time_expense
            sim_config[-1]+= "\nTime: Toal: {}s, OPT: {}s, {}%".format(round(tt,2), round(opt_tt,2), round(100*(opt_tt/tt),1))
        else:
            sim_config[-1]+= "\nTime: {}s".format(round(time.time()-self.program_start_T,2))
        # print to console
        self.logger.info('Simulation Configurations:\n{}\n'.format(
            tabulate([header, m_config, j_config, sqc_config, sim_config],
                    headers="firstrow", tablefmt="grid")))
        # performance metrics
        cum_tard = sum(self.recorder.j_tardiness_dict.values())
        max_tard = max(self.recorder.j_tardiness_dict.values())
        cum_flow = sum(self.recorder.j_flowtime_dict.values())
        max_flow = max(self.recorder.j_flowtime_dict.values())
        self.logger.info('Performance:\n{}\n'.format(tabulate(
            [["Category", "value"],
            ["Tardiness", "sum: {}, max:{}, mean: {}".format(cum_tard, max_tard, round(cum_tard / (self.j_idx), 2))],
            ["Flowtime", "sum:{}, max: {}, mean: {}".format(cum_flow, max_flow, round(cum_flow / (self.j_idx), 2))]],
            headers="firstrow", tablefmt="grid")))


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
        # sim data
        self.opt_time_expense = 0
        # record the job's journey
        self.in_system_jobs = {}
        self.j_operation_dict = {}
        self.j_tardiness_dict = {}
        self.j_flowtime_dict = {}
        self.m_bkd_dict = {idx: [] for idx in range(kwargs['m_no'])}
        self.m_cum_runtime_dict = {}
        self.pt_mean_dict = {}
        self.pt_std_dict = {}
        self.expected_tardiness_dict = {}
        # performance metric
        self.cumulative_tardiness = 0
