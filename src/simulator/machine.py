"""
Simulation model of a Machine (workstation) on the shopfloor.
When a machine becomes idle and multiple jobs are queuing to be processed,
the machine / sequencing agent picks one job from its queue for the next operation,
either by following a heuristic sequencing rule, a centralized schedule, or a trained DRL policy.
"""

# standard imports
from typing import List, Union, Literal
import simpy

# project modules
from .exc import SimulatorError
from .job import Job


class Machine:
    def __init__(self, *args, **kwargs):
        # Map user-specified keyword arguments to instance attributes
        for k, v in kwargs.items():
            setattr(self, k, v)

        # Timestamps tracking when the agent makes the current decision and when the machine is released
        self.decision_T: float = 0.0
        self.release_T: float = 0.0
        self.hidden_release_T: float = 0.0  # actual release time (invisible to scheduler when PT/MTTR is stochastic)
        self.current_job = None
        self.m_idx: int
        self.status: Literal["idle", "processing", "strategic_idle", "down"] = "idle"
        self.next_job_in_schedule: int = -1

        # Queue of jobs waiting at this workstation and stock-availability event
        self.queue: List[Union[Job, None]] = []
        self.sufficient_stock = self.env.event()

        # Working condition during normal operation vs. machine breakdown
        self.working_event = self.env.event()
        self.restoration_time: float = 0.0
        self.cumulative_runtime: float = 0.0
        # Machine starts in operational (non-broken) condition at t = 0
        self.working_event.succeed()

        # Data containers for learning and breakdown history recording
        self.breakdown_record = []

        # SimPy events for DRL learning hooks and strategic idleness synchronization
        self.env: simpy.Environment
        self.sequencing_learning_event = self.env.event()
        self.routing_learning_event = self.env.event()
        self.required_job_in_queue_event = self.env.event()
        self.required_job_in_queue_event.succeed()

    def initialization(self, **kwargs) -> None:
        """
        Link this machine to all other machines on the shopfloor, configure its sequencing
        decision mechanism (heuristic rule vs. centralized schedule), and start its SimPy process.
        """
        # Let all machines know each other so they can route completed operations to the next station
        self.m_list = kwargs['machine_list']
        # Specify the sequencing decision function
        self.job_sequencing = kwargs['sqc_method']
        self.central_scheduler = kwargs.get('central_scheduler', None)

        # Determine whether to follow a centralized production schedule (which supports strategic idleness)
        # or act reactively (always processing a queuing job as soon as the machine becomes available)
        self.schedule_mode = getattr(self.job_sequencing, '__name__', '') == "draw_from_schedule"

        # Activate the main production SimPy process loop
        self.production_proc = self.env.process(self.process_production())

    def process_production(self):
        """Main SimPy process loop simulating workstation operation, breakdowns, and job sequencing."""
        # At the beginning of the simulation, wait if the initial queue is empty
        if len(self.queue) < 1:
            yield self.env.process(self.process_idle())

        # Continuous production loop running until the end of the simulation
        while True:
            # Check if the machine is currently broken down before making a sequencing decision
            if not self.working_event.triggered:
                yield self.env.process(self.process_breakdown())

            # Check if the queue became empty while waiting
            if len(self.queue) == 0:
                self.status = "idle"
                yield self.env.process(self.process_idle())
                continue

            # If a centralized schedule rebuild is already triggered at this exact timestamp,
            # yield a zero-time step so the central scheduler updates the schedule first
            if (
                self.schedule_mode
                and self.central_scheduler is not None
                and self.central_scheduler.build_schedule_event.triggered
            ):
                yield self.env.timeout(0)
                continue

            """
            PART I: Make a sequencing decision (pick a job following a schedule, heuristic rule, or DRL policy)
            """
            # Record the timestamp of the sequencing decision (used as index in DRL experience replay)
            self.decision_T = self.env.now
            decision_type = None

            # TYPE I: Centralized schedule mode (allows strategic idleness to wait for an incoming job)
            if self.schedule_mode:
                # Wait until the machine is operational, no reschedule is pending, and the scheduled job is in queue
                while (
                    not self.working_event.triggered
                    or len(self.queue) == 0
                    or (self.central_scheduler is not None and self.central_scheduler.build_schedule_event.triggered)
                    or self.next_job_in_schedule not in [j.j_idx for j in self.queue]
                ):
                    if self.central_scheduler is not None and self.central_scheduler.build_schedule_event.triggered:
                        yield self.env.timeout(0)
                    elif not self.working_event.triggered:
                        yield self.env.process(self.process_breakdown())
                    elif len(self.queue) == 0:
                        self.status = "idle"
                        yield self.env.process(self.process_idle())
                    else:
                        # The next scheduled job has not yet arrived at this machine: enter strategic idleness
                        self.check_strategic_idleness()
                        yield self.required_job_in_queue_event

                # Refresh decision timestamp after any strategic idleness wait
                self.decision_T = self.env.now
                # Pop the scheduled job index from the central schedule now that the machine commits to processing it
                scheduled_job_idx = self.job_sequencing(m_idx=self.m_idx)
                try:
                    self.sqc_decision_pos = [j.j_idx for j in self.queue].index(scheduled_job_idx)
                    self.picked_j_instance = self.queue[self.sqc_decision_pos]
                except ValueError as exc:
                    raise SimulatorError(
                        f"Machine {self.m_idx} trying to pick Job {scheduled_job_idx} that is not in queue!"
                    ) from exc

                self.recorder.sqc_cnt_opt += 1
                decision_type = 'Scheduled'

            # TYPE II: Reactive sequencing strategy with multiple queuing jobs (decision required)
            elif len(self.queue) > 1:
                # The returned value is the picked job's position in this machine's queue
                self.sqc_decision_pos = self.job_sequencing(jobs=self.queue)
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.recorder.sqc_cnt_reactive += 1
                decision_type = 'Reactive'

            # TYPE III: Reactive sequencing strategy with only a single job in queue (passive selection)
            else:
                self.sqc_decision_pos = 0
                self.picked_j_instance = self.queue[self.sqc_decision_pos]
                self.recorder.sqc_cnt_passive += 1
                decision_type = 'Passive'

            """
            PART II: After the sequencing decision, update entity states and execute the operation
            """
            self.status = "processing"
            actual_pt = self.after_decision()
            self.logger.debug(
                "{} > {}. Machine {} process Job {}, expected PT: {}, actual: {}".format(
                    self.env.now,
                    decision_type,
                    self.m_idx,
                    self.picked_j_instance.j_idx,
                    self.picked_j_instance.remaining_operations[0][1],
                    actual_pt,
                )
            )
            # Simulate the operation processing duration (yield actual processing time)
            yield self.env.timeout(actual_pt)
            self.logger.info(
                "{} > OUT: Job {} left Machine {}".format(
                    self.env.now, self.picked_j_instance.j_idx, self.m_idx
                )
            )

            """
            PART III: After operation completion, route job to next station and check breakdown / idleness
            """
            # Transfer job to its next workstation or mark it completed in the system
            self.after_operation()
            # Yield a zero-time step so concurrent events at the exact same timestamp (e.g., breakdown start) settle
            yield self.env.timeout(0)
            # Check if machine breakdown was triggered upon completion of this operation
            if not self.working_event.triggered:
                yield self.env.process(self.process_breakdown())
            # Check queue stock level; if empty, transition to idle state
            if len(self.queue) == 0:
                self.status = "idle"
                yield self.env.process(self.process_idle())

    def process_idle(self):
        """SimPy sub-process suspending the machine while its queue is empty."""
        self.status = "idle"
        self.logger.info("{} > IDL on: Machine {} became idle".format(self.env.now, self.m_idx))
        # Reset `sufficient_stock` event and wait until a new job arrives in `self.queue`
        self.sufficient_stock = self.env.event()
        yield self.sufficient_stock
        # If a machine breakdown is active when the job arrives, wait for repair before resuming
        if not self.working_event.triggered:
            yield self.env.process(self.process_breakdown())
        self.logger.info("{} > IDL off: Machine {} replenished".format(self.env.now, self.m_idx))

    def process_breakdown(self):
        """SimPy sub-process suspending production while the machine undergoes repair."""
        start = self.env.now
        # Suspend production until `working_event` is triggered by `Narrator.process_machine_breakdown`
        yield self.working_event
        delay = round(self.env.now - start, 2)
        self.breakdown_record.append([(self.m_idx, start, delay)])
        self.logger.info(
            f"{self.env.now} > BKD end: Machine {self.m_idx} repaired, delayed the production for {delay} units"
        )

    def job_arrival(self, arriving_job: Job) -> None:
        """Handle a job arriving at this machine's queue (either newly created or routed from another machine)."""
        # Add the arriving job instance to the queue and record its arrival time
        self.queue.append(arriving_job)
        arriving_job.after_arrival()

        # Wake up the machine if it was waiting for stock in `process_idle`
        if not self.sufficient_stock.triggered:
            self.sufficient_stock.succeed()

        common_msg = "{} > IN: Job {} arrived at Machine {}, current status {}, queue: {}".format(
            self.env.now,
            arriving_job.j_idx,
            self.m_idx,
            self.status,
            [j.j_idx for j in self.queue],
        )
        extra_msg = ", scheduled next op/job: {}".format(
            self.next_job_in_schedule if self.next_job_in_schedule >= 0 else 'None'
        )
        self.logger.info(common_msg + extra_msg if self.schedule_mode else common_msg)

        # If centralized schedule mode is active and the machine is waiting in strategic idleness,
        # check whether the newly arrived job is the required next scheduled job
        if self.schedule_mode and arriving_job.j_idx == self.next_job_in_schedule:
            if not self.required_job_in_queue_event.triggered:
                self.required_job_in_queue_event.succeed()
                if self.status == "strategic_idle":
                    self.logger.info("{} > Str.Idle end: Machine {} reactivated".format(self.env.now, self.m_idx))

    def check_strategic_idleness(self) -> None:
        """
        Check whether the next scheduled job is already in the queue.
        If not, place the machine into strategic idleness until the required job arrives or the schedule changes.
        """
        if self.next_job_in_schedule in [j.j_idx for j in self.queue]:
            if not self.required_job_in_queue_event.triggered:
                self.required_job_in_queue_event.succeed()
        else:
            if self.status != "strategic_idle":
                self.status = "strategic_idle"
                self.recorder.sqc_cnt_SI += 1
                self.logger.info(
                    "{} > STR.IDL. on: Machine {} suspended, waiting for Job {}, current queue: {}".format(
                        self.env.now,
                        self.m_idx,
                        self.next_job_in_schedule,
                        [j.j_idx for j in self.queue],
                    )
                )
            self.required_job_in_queue_event = self.env.event()

    def update_status_after_new_schedule(self) -> None:
        """
        Called by CentralScheduler after a schedule rebuild updates `self.next_job_in_schedule`.
        If the newly assigned next job is already waiting in `self.queue`, reactivate the machine.
        """
        if self.next_job_in_schedule in [j.j_idx for j in self.queue]:
            if not self.required_job_in_queue_event.triggered:
                self.required_job_in_queue_event.succeed()
                if self.status == "strategic_idle":
                    self.logger.info("{} > Str.Idle end: Machine {} reactivated".format(self.env.now, self.m_idx))

    def after_decision(self) -> float:
        """
        Update machine and job timing attributes immediately after a job is selected for processing.
        Returns the actual processing time of the upcoming operation.
        """
        expected_pt = self.picked_j_instance.remaining_pt[0]  # nominal processing time visible to scheduler
        actual_pt = self.picked_j_instance.actual_remaining_pt[0]  # actual processing time (may differ if pt_cv > 0)
        wait = self.env.now - self.picked_j_instance.arrival_T  # queuing wait time before being picked

        # Record this operation on the job instance
        self.picked_j_instance.after_decision(self.m_idx, wait)

        # Update machine release times and cumulative runtime
        self.release_T = self.env.now + expected_pt
        self.hidden_release_T = self.env.now + actual_pt  # invisible to decision maker
        self.cumulative_runtime = round(self.cumulative_runtime + actual_pt, 2)
        self.current_job = self.picked_j_instance.j_idx
        self.recorder.m_cum_runtime_dict[self.m_idx] = self.cumulative_runtime
        return actual_pt

    def after_operation(self) -> None:
        """
        Pop the processed job from this machine's queue upon operation completion
        and route it to its next required workstation (if any).
        """
        leaving_job = self.queue.pop(self.sqc_decision_pos)
        # Reset machine's active decision and release times to current timestamp (if not broken down)
        self.sqc_decision_pos = None
        self.current_job = None
        if self.working_event.triggered:
            self.status = "idle"
            self.release_T = self.env.now
            self.hidden_release_T = self.env.now

        next_machine = leaving_job.after_operation()
        if next_machine > -1:  # route job to its next machine along its trajectory
            self.m_list[next_machine].job_arrival(leaving_job)

    def __del__(self):
        # Ensure cumulative runtime is recorded in the shared recorder
        if hasattr(self, 'recorder') and hasattr(self, 'm_idx'):
            self.recorder.m_cum_runtime_dict[self.m_idx] = self.cumulative_runtime

    def overstay_check(self) -> None:
        """Check queued jobs and remove any job whose tardiness exceeds 200 time units."""
        if not self.queue:
            return
        for j in list(self.queue):
            overstay = max(0.0, self.env.now - j.due)
            if overstay > 200:
                self.queue.remove(j)
                j.overstay()

    def complete_experience(self) -> None:
        """
        DRL experience replay hook: called upon completion of an operation when `sequencing_learning_event`
        is active. Appends the resulting state `s_t` and reward `r_t` to form a complete transition tuple.
        """
        try:
            # Verify whether an incomplete experience exists at `self.decision_T`
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T]
            local_data = self.sequencing_data_generation()
            s_t = self.build_state(local_data)
            r_t = self.reward_function()
            self.event_creator.sqc_reward_record.append([self.env.now, r_t])
            self.event_creator.incomplete_rep_memo[self.m_idx][self.decision_T] += [s_t, r_t]
            complete_exp = self.event_creator.incomplete_rep_memo[self.m_idx].pop(self.decision_T)
            self.event_creator.rep_memo[self.m_idx].append(complete_exp)
        except Exception:
            pass

