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
from sequencing_rule import *


class Machine:
    def __init__(self, env, logger, recorder, *args, **kwargs):
        # initialize the environment of simulation
        self.env = env
        self.logger = logger
        self.recorder = recorder
        self.m_idx = kwargs['m_idx']
        # the time that agent make current and next decision
        self.decision_time = 0
        self.release_time = 0

        # Initialize the possible events during production
        self.queue = []
        self.sufficient_stock = self.env.event()
        # working condition in shut down or breakdown
        self.working_event = self.env.event()
        self.restoration_time = 0
        # no shutdown, no breakdown at beginning
        self.working_event.succeed()
        # initialize the data for learning and recordiing
        self.breakdown_record = []
        # set the sequencing rule
        if 'sqc_rule' in kwargs:
            try:
                self.job_sequencing = eval(kwargs['sqc_rule'])
                self.logger.info("Machine {} uses {} sequencing rule".format(self.m_idx, kwargs['sqc_rule']))
            except Exception as e:
                self.logger.error("Sequencing rule assigned to machine {} is invalid! Invalid entry: {}".format(self.m_idx, kwargs['sqc_rule']))
                self.logger.error(str(e))
                raise Exception
        else:
            # default sequencing rule is FIFO
            self.logger.info("Machine {} uses default FIFO rule".format(self.m_idx))
            self.job_sequencing = FIFO
        # record extra data for learning, initially not activated, can be activated by brains
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()
        self.env.process(self.production())

    def initialization(self, **kwargs):
        self.m_list = kwargs['machine_list']


    # The main function, simulates the production
    def production(self):
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
                self.logger.info("SQC: Machine %s choose job %s at time %s"%(self.m_idx, self.picked_j_instance.j_idx, self.env.now))
            # otherwise simply select the first(only) one
            else:
                self.sqc_decision_pos = 0
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.logger.info("One queue: Machine %s process job %s at time %s"%(self.m_idx, self.picked_j_instance.j_idx, self.env.now))
            # retrive the information of job
            pt = self.picked_j_instance.remaining_pt[0] # processing time of the picked job in this stage
            wait = self.env.now - self.picked_j_instance.arrival_t # time that job waited before being picked
            self.picked_j_instance.record_operation(self.m_idx, pt, wait) # record these information
            # The production process (yield the processing time of operation)
            yield self.env.timeout(pt)
            #self.cumulative_run_time += pt
            self.logger.info("OPN: Job {} leave Machine {} at time {}".format(self.picked_j_instance.j_idx, self.m_idx, self.env.now))
            # transfer job to next workcenter or delete it, and update information
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
        self.logger.info("IDL on: Machine {} becomes idle at time {}".format(self.m_idx, self.env.now))
        # set the self.sufficient_stock event to untriggered
        self.sufficient_stock = self.env.event()
        # proceed only if the sufficient_stock event is triggered by new job arrival
        yield self.sufficient_stock
        # examine whether the scheduled shutdown is triggered
        if not self.working_event.triggered:
            yield self.env.process(self.breakdown())
        self.logger.info("IDL off: Machine {} replenished at time {}".format(self.m_idx, self.env.now))


    # or when machine failure happens
    def breakdown(self):
        self.logger.info("BKD on: Machine {} broke at time {}".format(self.m_idx, self.env.now))
        start = self.env.now
        # simply update the available time of that machines
        self.available_time = self.restoration_time + self.cumulative_pt
        # suspend the production here, untill the working_event is triggered
        yield self.working_event
        self.breakdown_record.append([(start, self.env.now - start), self.m_idx])
        self.logger.info("BKD off: Machine {} restored at time {}, down for {} units".format(self.m_idx, self.env.now, self.env.now - start))


    '''
    2. downwards are functions the called before and after each operation
       to maintain some record, and transit the finished job to next workcenter or out of system
    '''
    # a new job (instance) arrives
    def job_arrival(self, arriving_job):
        # add the job instance to queue
        self.queue.append(arriving_job)
        arriving_job.before_operation()
        self.logger.info("ARV: Job {} arrived at Machine {}".format(arriving_job.j_idx, self.m_idx))
        # change the stocking status if machine is currently idle
        if not self.sufficient_stock.triggered:
            self.sufficient_stock.succeed()


    # update information that will be used for calculating the rewards
    def before_operation(self):
        pass


    def after_operation(self):
        leaving_job = self.queue.pop(self.sqc_decision_pos)
        next = leaving_job.after_operation()
        if next > -1: # if returned index is valid
            self.m_list[next].job_arrival(leaving_job)


    '''
    3. downwards are functions that related to information update and exchange
       especially the information that will be used by other agents on shop floor
    '''

    # call this function after the completion of operation
    def state_update_all(self):
        pass

    # available time is a bit tricky, jobs may come when the operation is ongoing
    # or when the machine is already in idle (availble time is earlier than now)
    # hence we can't simply let available time = now + cumulative_pt
    def state_update_after_job_arrival(self, increased_available_time):
        self.current_pt = np.array([x[self.m_idx] for x in self.pt_list])
        self.cumulative_pt = self.current_pt.sum()
        # add the new job's pt to current time / current available time
        self.available_time = max(self.available_time, self.env.now) + increased_available_time
        self.que_size = len(self.queue)

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

    # update the information of the job that being processed to event_creator !!!
    def update_global_info_anticipation(self,pt):
        current_j_idx = self.queue[self.sqc_decision_pos]
        self.event_creator.current_j_idx_list[self.m_idx] = current_j_idx
        next_wc = self.sequence_list[self.sqc_decision_pos][0] if len(self.sequence_list[self.sqc_decision_pos]) else -1 # next workcenter of the job
        self.event_creator.next_wc_list[self.m_idx] = next_wc # update the next wc info (hold by job creator)
        self.release_time = self.env.now + pt
        self.event_creator.release_time_list[self.m_idx] = self.release_time # update the time of completion of current operation
        job_rempt = self.remaining_job_pt[self.sqc_decision_pos].sum() - pt
        self.event_creator.arriving_job_rempt_list[self.m_idx] = job_rempt # update the remaining pt of job under processing
        job_slack = self.slack[self.sqc_decision_pos]
        self.event_creator.arriving_job_slack_list[self.m_idx] = job_slack # update the slack time of processing job (hold by job creator)

    # must call this after operation otherwise the record persists, lead to error
    def update_global_info_after_operation(self):
        self.event_creator.next_wc_list[self.m_idx] = -1 # after each operation, clear the record in job creator

    # give out the information related to routing decision
    def routing_data_generation(self):
        # note that we subtract current time from available_time
        # becasue state_update_all function may be called at a different time
        self.routing_data = [self.cumulative_pt, max(0,self.available_time-self.env.now), self.que_size, self.cumulative_run_time]
        return self.routing_data

    # give ou the information related to sequencing decision
    def sequencing_data_generation(self):
        self.sequencing_data = \
        [self.current_pt, self.remaining_job_pt, np.array(self.due_list), self.env.now, self.completion_rate, \
        self.time_till_due, self.slack, self.winq, self.avlm, self.next_pt, self.remaining_no_op, self.waited_time, \
        self.wc_idx, self.queue, self.m_idx]
        #print(self.sequencing_data)
        return self.sequencing_data


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
