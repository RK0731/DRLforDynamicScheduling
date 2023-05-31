'''
This is the simulation model of machine
When machine becomes idle and more than one jobs are queuing to be processed
The machine/sequencing agent would pick one job from its queue for the next operation
Either by a sequencing rule or a set of trained parameters
'''

import traceback
import simpy
import numpy as np
import torch
from tabulate import tabulate


class Machine:
    def __init__(self, *args, **kwargs):
        # user specified attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # the time that agent make current and next decision
        self.decision_time = 0
        self.release_time = 0
        # Initialize the possible events during production
        self.queue = []
        self.sufficient_stock = self.env.event()
        # working condition in shut down or breakdown
        self.working_event = self.env.event()
        self.restoration_time = 0
        self.cumulative_runtime = 0
        # no shutdown, no breakdown at beginning
        self.working_event.succeed()
        # initialize the data for learning and recordiing
        self.breakdown_record = []
        # set the sequencing decision-maker
        # record extra data for learning, initially not activated, can be activated by brains
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()
        self.env.process(self.process_production())


    def initialization(self, **kwargs):
        # let all machines know each other so they can pass the jobs around
        self.m_list = kwargs['machine_list']
        # specify the sequencing decision maker
        self.job_sequencing = kwargs['sqc_rule']

    # The main function, simulates the production
    def process_production(self):
        # at the begining of simulation, check the initial queue/stock level
        if len(self.queue) < 1:
            yield self.env.process(self.idle())
        # the loop that will run till the end of simulation
        while True:
            # record the time of the sequencing decision, used as the index of produciton record in job creator
            self.decision_time = self.env.now
            # if we have more than one queuing jobs, sequencing is required
            if len(self.queue) > 1:
                # the returned value is picked job's position in machine's queue
                self.sqc_decision_pos = self.job_sequencing(self.queue)
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.logger.info("{} >>> SQC on: Machine {} picks Job {}".format(
                    self.env.now, self.m_idx, self.picked_j_instance.j_idx))
            # otherwise simply select the first(only) one
            else:
                self.sqc_decision_pos = 0
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.logger.info("{} >>> SQC off: Machine {} processes Job {}".format(
                    self.env.now, self.m_idx, self.picked_j_instance.j_idx))
            # retrive the information of job
            pt = self.picked_j_instance.actual_remaining_pt[0] # the actual processing time in this stage, can be different from expected value
            wait = self.env.now - self.picked_j_instance.arrival_t # time that job waited before being picked
            self.after_decision(pt, wait)
            self.logger.debug("{} >>> PT: Job {} on Machine {} proc.t, expected: {}, actual: {}".format(
                self.env.now, self.picked_j_instance.j_idx, self.m_idx, self.picked_j_instance.remaining_pt[0], pt))
            # The production process (yield the processing time of operation)
            yield self.env.timeout(pt)
            self.logger.info("{} >>> DEP: Job {} departs from Machine {}".format(
                self.env.now, self.picked_j_instance.j_idx, self.m_idx))
            # transfer job to next station or remove it from system
            self.after_operation()
            # check if machine is shut down/broken
            if not self.working_event.triggered:
                yield self.env.process(self.breakdown())
                self.state_update_all()
            # check the stock level
            if len(self.queue) == 0:
                # start the idle process
                yield self.env.process(self.idle())
                self.state_update_all()
    

    # when there's no job queueing, machine becomes idle
    def idle(self):
        self.logger.info("{} >>> IDL on: Machine {} became idle".format(self.env.now, self.m_idx))
        # set the self.sufficient_stock event to untriggered
        self.sufficient_stock = self.env.event()
        # proceed only if the sufficient_stock event is triggered by new job arrival
        yield self.sufficient_stock
        # examine whether the scheduled shutdown is triggered
        if not self.working_event.triggered:
            yield self.env.process(self.breakdown())
        self.logger.info("{} >>> IDL off: Machine {} replenished".format(self.env.now, self.m_idx))


    # or when machine failure happens
    def breakdown(self):
        self.logger.info("{} >>> BKD on: Machine {} is broken".format(self.env.now, self.m_idx))
        start = self.env.now
        # suspend the production here, untill the working_event is triggered
        yield self.working_event
        self.breakdown_record.append([(self.m_idx, start, self.env.now - start)])
        self.logger.info("{} >>> BKD off: Machine {} restored, delayed the production for {} units".format(self.env.now, self.m_idx, self.env.now - start))


    # a new job (instance) arrives
    def job_arrival(self, arriving_job):
        # add the job instance to queue
        self.queue.append(arriving_job)
        arriving_job.before_operation()
        self.logger.info("{} >>> ARV: Job {} arrived at Machine {}".format(self.env.now, arriving_job.j_idx, self.m_idx))
        # change the stocking status if machine is currently idle
        if not self.sufficient_stock.triggered:
            self.sufficient_stock.succeed()


    # update information that will be used for calculating the rewards
    def before_operation(self):
        pass


    def after_decision(self, pt, wait):
        self.picked_j_instance.record_operation(self.m_idx, self.env.now, pt, wait)
        self.release_time = self.env.now + pt
        self.cumulative_runtime += pt


    def after_operation(self):
        leaving_job = self.queue.pop(self.sqc_decision_pos)
        next = leaving_job.after_operation()
        if next > -1: # if returned index is valid
            self.m_list[next].job_arrival(leaving_job)


    def __del__(self):
        self.logger.debug("{} >>> MACHINE {} instance deleted".format(self.env.now, self.m_idx))
        # append the operation histroy to the recorder
        self.recorder.m_cum_runtime_dict[self.m_idx] = self.cumulative_runtime


    '''
    3. downwards are functions that related to information update and exchange
       especially the information that will be used by other agents on shop floor
    '''

    # call this function after the completion of operation
    def state_update_all(self):
        pass


    # update the information of progression, eralized and expected tardiness to event_creator !!!
    def update_global_info_progression(self):
        # realized: 0 if already tardy; exp: 0 is slack time is negative
        realized = self.time_till_due.clip(0,1)
        exp = self.slack.clip(0,1)
        # update the machine's corresponding record in job creator, and several rates
        self.event_creator.comp_rate_list[self.m_idx] = self.completion_rate
        self.event_creator.comp_rate = np.concatenate(self.event_creator.comp_rate_list).mean()
        self.event_creator.realized_tard_list[self.m_idx] = realized
        self.event_creator.realized_tard_rate = 1 - np.concatenate(self.event_creator.realized_tard_list).mean()
        self.event_creator.exp_tard_list[self.m_idx] = exp
        self.event_creator.exp_tard_rate = 1 - np.concatenate(self.event_creator.exp_tard_list).mean()
        self.event_creator.available_time_list[self.m_idx] = self.available_time


    '''
    4. downwards are functions related to the calculation of reward and construction of state
       only be called if the sequencing learning mode is activated
       the options of reward function are listed at bottom
    '''


    # this function is called only if self.sequencing_learning_event is triggered
    # when this function is called upon the completion of an operation
    # it add received data to corresponding record in job creator's incomplete_rep_memo
    def complete_experience(self):
        # it's possible that not all machines keep memory for learning
        # machine that needs to keep memory don't keep record for all jobs
        # only when they have to choose from several queuing jobs
        try:
            # check whether corresponding experience exists, if not, ends at this line
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_time]
            #print('PARAMETERS',self.m_idx,self.decision_time,self.env.now)
            #print('BEFORE\n',self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_time])
            # if yes, get the global state
            local_data = self.sequencing_data_generation()
            s_t = self.build_state(local_data)
            #print(self.m_idx,s_t)
            r_t = self.reward_function() # can change the reward function, by sepecifying before the training
            #print(self.env.now, r_t)
            self.event_creator.sqc_reward_record.append([self.env.now, r_t])
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_time] += [s_t, r_t]
            #print(self.event_creator.incomplete_rep_memo[self.m_idx])
            #print(self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_time])
            complete_exp = self.event_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_time)
            # and add it to rep_memo
            self.event_creator.rep_memo[self.m_idx].append(complete_exp)
            #print(self.event_creator.rep_memo[self.m_idx])
            #print('AFTER\n',self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_time])
            #print(self.m_idx,self.env.now,'state: ',s_t,'reward: ',r_t)
        except:
            pass
