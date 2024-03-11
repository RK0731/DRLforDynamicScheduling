'''
This is the simulation model of machine
When machine becomes idle and more than one jobs are queuing to be processed
The machine/sequencing agent would pick one job from its queue for the next operation
Either by a sequencing rule or a set of trained parameters
'''

from typing import Optional, Union, Literal, Any
import simpy
import numpy as np
from .sequencing_rule import *
from .exc import *


class Machine:
    def __init__(self, *args, **kwargs):
        # user specified attributes
        for k, v in kwargs.items():
            setattr(self, k, v)
        # the time that agent make current and next decision
        self.decision_T = 0
        self.release_T = self.hidden_release_T = 0
        self.current_job = None
        self.status: Literal["idle", "processing", "strategic_idle", "down"] = "idle"
        self.next_job_in_schedule = -1
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
        # events
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()
        self.required_job_in_queue_event = self.env.event()


    def initialization(self, **kwargs):
        # let all machines know each other so they can pass the jobs around
        self.m_list = kwargs['machine_list']
        # specify the sequencing decision maker
        self.job_sequencing = kwargs['sqc_rule']
        # follow a subset of centralized production schedule which allows strategic idleness
        # or act in a reactive way that always process the queuing job as soon as possible?
        self.schedule_mode = False
        # [self.job_sequencing] is assigned by event.Narrator
        if self.job_sequencing.__name__ == "draw_from_schedule":
            self.schedule_mode = True
        # activate the produciton
        self.production_proc = self.env.process(self.process_production())


    # The main function, simulates the production
    def process_production(self):
        # at the begining of simulation, check the initial queue/stock level
        if len(self.queue) < 1:
            yield self.env.process(self.process_idle())
        # the loop that will run till the end of simulation
        while True:
            """
            PART I: when sequencing decision is needed, draw a job by specified rule/schedule
            """
            # record the time of the sequencing decision, used as the index of produciton record in job creator
            self.decision_T = self.env.now
            _decision_type = None
            # TYPE I: if strategic idleness is allowed
            if self.schedule_mode:
                # the returned value is the first job's index in pre-developed schedule
                # i.e. the next job that should be processed by this machine
                # WARNING: a sequencing decision has been made, pop the first element from the schedule
                self.next_job_in_schedule = self.job_sequencing(m_idx = self.m_idx)
                # if the job in schedule is NOT in queue, activate the strategic idleness process
                self.check_strategic_idleness()
                yield self.required_job_in_queue_event
                # when job required is in queue (with ot without strategic idleness)
                self.sqc_decision_pos = [j.j_idx for j in self.queue].index(self.next_job_in_schedule)
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.recorder.sqc_cnt_opt += 1
                _decision_type = 'Scheduled'
            # TYPE II: if sequencing is reactive
            # and we have more than one queuing jobs, sequencing is required
            elif len(self.queue) > 1:
                # the returned value is picked job's position in machine's queue
                self.sqc_decision_pos = self.job_sequencing(jobs = self.queue)
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.recorder.sqc_cnt_reactive += 1
                _decision_type = 'Reactive'
            # otherwise simply select the first(only) one
            else:
                self.sqc_decision_pos = 0
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.recorder.sqc_cnt_passive += 1
                _decision_type = 'Passive'
            """
            PART II. after the decision, update infromation and perform the operation
            """
            # update job instance, and get the time of operation
            self.status = "processing"
            actual_pt = self.after_decision()
            self.logger.debug("{}. Machine {} process Job {}, expected PT: {}, actual: {}".format(
                _decision_type, self.m_idx, self.picked_j_instance.j_idx, self.picked_j_instance.remaining_operations[0][1], actual_pt))
            # The production process (yield the actual processing time of operation)
            yield self.env.timeout(actual_pt)
            self.logger.info("{} > DEP: Job {} departs Machine {}".format(
                self.env.now, self.picked_j_instance.j_idx, self.m_idx))
            """
            PART III: after operation, update information and check for machine breakdown and idleness
            """
            # transfer job to next station or remove it from system
            self.after_operation()
            # check if machine is shut down/broken
            if not self.working_event.triggered:
                self.status = "down"
                yield self.env.process(self.process_breakdown())
            # check the stock level
            if len(self.queue) == 0:
                self.status = "idle"
                yield self.env.process(self.process_idle())
    

    # when there's no job queueing, machine becomes idle
    def process_idle(self):
        self.logger.info("{} > IDL on: Machine {} became idle".format(self.env.now, self.m_idx))
        # set the self.sufficient_stock event to untriggered
        self.sufficient_stock = self.env.event()
        # proceed only if the sufficient_stock event is triggered by new job arrival
        yield self.sufficient_stock
        # examine whether the scheduled shutdown is triggered
        if not self.working_event.triggered:
            yield self.env.process(self.process_breakdown())
        self.logger.info("{} > IDL off: Machine {} replenished".format(self.env.now, self.m_idx))


    # or when machine failure happens
    def process_breakdown(self):
        self.logger.info("{} > BKD on: Machine {} is broken".format(self.env.now, self.m_idx))
        start = self.env.now
        # suspend the production here, untill the working_event is triggered
        yield self.working_event
        self.breakdown_record.append([(self.m_idx, start, self.env.now - start)])
        self.logger.info("{} > BKD off: Machine {} repaired, delayed the production for {} units".format(
            self.env.now, self.m_idx, self.env.now - start))


    # a new job (instance) arrives
    def job_arrival(self, arriving_job: object):
        # add the job instance to queue
        self.queue.append(arriving_job)
        arriving_job.after_arrival()
        # change the stocking status if machine is currently idle (empty stock or strategic)
        if not self.sufficient_stock.triggered:
            self.sufficient_stock.succeed()
        common_msg = "{} > ARV: Job {} arrived at Machine {}, current status {}, queue: {}".format(
            self.env.now, arriving_job.j_idx, self.m_idx, self.status, [j.j_idx for j in self.queue])
        extra_msg = ", next job in schedule: {}".format(self.next_job_in_schedule)
        self.logger.info(common_msg + extra_msg if self.schedule_mode else common_msg)
        # if schedule mode is ON, need to check if arrived job match the required job
        if self.schedule_mode:
            # check if arriving job matches the [next_job_in_schedule]
            if arriving_job.j_idx == self.next_job_in_schedule:
                # if so, end strategic idleness and reactivate machine
                if not self.required_job_in_queue_event.triggered:
                    self.required_job_in_queue_event.succeed()
                    self.logger.info("{} > Str.Idle end: Machine {} reactivated".format(self.env.now, self.m_idx))


    # suspend the machine if strategic idleness is needed
    def check_strategic_idleness(self):
        # if the next job in schedule is now queuing
        if self.next_job_in_schedule in [j.j_idx for j in self.queue]:
            if not self.required_job_in_queue_event.triggered:
                self.required_job_in_queue_event.succeed() 
        # otherwise need to wait for the arrival of required job
        else:
            self.status = "strategic_idle" # and change the status
            self.recorder.sqc_cnt_SI += 1
            self.logger.info("{} > STR.IDL. on: Machine {} suspended, waiting for Job {}, current queue: {}".format(
                self.env.now, self.m_idx, self.next_job_in_schedule, [j.j_idx for j in self.queue]))
            self.required_job_in_queue_event = self.env.event()


    def update_status_after_new_schedule(self):
        # after change the [next_job_in_schedule], check the match again
        if self.next_job_in_schedule in [j.j_idx for j in self.queue]:
            if not self.required_job_in_queue_event.triggered:
                self.required_job_in_queue_event.succeed() 


    def after_decision(self) -> int:
        # get data of upcoming operation
        expected_pt = self.picked_j_instance.remaining_pt[0] # the expected processing time
        actual_pt = self.picked_j_instance.actual_remaining_pt[0] # the actual processing time in this stage, can be different from expected value
        wait = self.env.now - self.picked_j_instance.arrival_T # time that job queued before being picked
        # record this decision/operation
        self.picked_j_instance.after_decision(self.m_idx, wait)
        # update status of picked job and machine
        self.release_T = self.env.now + expected_pt
        self.hidden_release_T = self.env.now + actual_pt # invisible to decision maker
        self.cumulative_runtime += actual_pt
        self.current_job = self.picked_j_instance.j_idx
        return actual_pt


    def after_operation(self):
        leaving_job = self.queue.pop(self.sqc_decision_pos)
        # reset the decision
        self.sqc_decision_pos = None
        self.current_job = None
        next = leaving_job.after_operation()
        if next > -1: # if returned index is valid
            self.m_list[next].job_arrival(leaving_job)


    def __del__(self):
        # append the operation histroy to the recorder
        self.recorder.m_cum_runtime_dict[self.m_idx] = self.cumulative_runtime


    def overstay_check(self):
        if not self.queue:
            return
        for j in self.queue:
            overstay = max(0, self.env.now - j.due)
            if overstay > 200:
                j.overstay()


    # this function is called only if self.sequencing_learning_event is triggered
    # when this function is called upon the completion of an operation
    # it add received data to corresponding record in job creator's incomplete_rep_memo
    def complete_experience(self):
        # it's possible that not all machines keep memory for learning
        # machine that needs to keep memory don't keep record for all jobs
        # only when they have to choose from several queuing jobs
        try:
            # check whether corresponding experience exists, if not, ends at this line
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T]
            #print('PARAMETERS',self.m_idx,self.decision_T,self.env.now)
            #print('BEFORE\n',self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T])
            # if yes, get the global state
            local_data = self.sequencing_data_generation()
            s_t = self.build_state(local_data)
            #print(self.m_idx,s_t)
            r_t = self.reward_function() # can change the reward function, by sepecifying before the training
            #print(self.env.now, r_t)
            self.event_creator.sqc_reward_record.append([self.env.now, r_t])
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T] += [s_t, r_t]
            #print(self.event_creator.incomplete_rep_memo[self.m_idx])
            #print(self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T])
            complete_exp = self.event_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_T)
            # and add it to rep_memo
            self.event_creator.rep_memo[self.m_idx].append(complete_exp)
            #print(self.event_creator.rep_memo[self.m_idx])
            #print('AFTER\n',self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T])
            #print(self.m_idx,self.env.now,'state: ',s_t,'reward: ',r_t)
        except:
            pass
