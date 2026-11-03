"""
Job entity class for the discrete-event simulator.
Carries job-specific attributes including routing trajectory, nominal (expected) and actual
processing times, due date, current status, and operation execution history.
"""

from dataclasses import dataclass
from logging import Logger
from typing import Optional, Union, Literal, Any, List

import numpy as np
from simpy import Environment


@dataclass
class Job:
    env: Environment
    logger: Logger
    recorder: Any
    rng: np.random.Generator
    j_idx: int
    trajectory: np.ndarray
    pt_by_m_idx: np.ndarray
    pt_range: List[Union[int, float]]
    pt_cv: Union[int, float]
    due_tightness: float
    status: Literal["queuing", "processing", "completed", "overstay"] = "queuing"
    transfer_t: float = 0.0

    def __post_init__(self):
        """Initialize derived timing, routing stacks, stochastic processing times, and due date."""
        # Intrinsic time attributes: creation time, arrival time at current station, and expected availability time
        self.creation_T = self.arrival_T = self.env.now
        self.available_T = self.env.now
        self.completion_T: Optional[float] = None

        # Re-order the processing time array by the sequence of operations in `self.trajectory`
        pt_by_ops = self.pt_by_m_idx[self.trajectory]
        if self.pt_cv == 0:
            self.remaining_pt = [float(x) for x in pt_by_ops]
            # Stack of actual processing times; equals expected processing times when variability is off
            self.actual_remaining_pt = [float(x) for x in pt_by_ops]
        else:
            self.remaining_pt = [float(x) for x in pt_by_ops]
            # Sample actual processing times from Normal(mean=pt, std=pt*pt_cv), rounded to 0.1 and clipped to pt_range
            actual_pt = np.around(
                self.rng.normal(pt_by_ops, pt_by_ops * self.pt_cv), decimals=1
            ).clip(*self.pt_range)
            self.actual_remaining_pt = [float(x) for x in actual_pt]

        # Stack of machine indices that the job still needs to visit
        self.remaining_machines = [int(m) for m in self.trajectory]
        # Preserve immutable copies of the full trajectory and PTs for the `jobs` database table
        self.initial_trajectory = list(self.remaining_machines)
        self.initial_exp_pt = list(self.remaining_pt)
        self.initial_actual_pt = list(self.actual_remaining_pt)

        # Zip remaining machines, expected processing times, and actual processing times
        self.remaining_operations = list(
            zip(self.remaining_machines, self.remaining_pt, self.actual_remaining_pt)
        )

        # Generate due date for the job, proportional to its total nominal processing time
        self.due = float(
            np.round(self.pt_by_m_idx.sum() * self.rng.uniform(1.2, self.due_tightness) + self.env.now)
        )
        self.tardiness = 0.0

        # Operation history recording: list of [m_idx, begin_time, actual_pt, wait_time]
        self.operation_record = []
        self.recorder.all_jobs_dict[self.j_idx] = self

        msg = "{} > Job {} created, trajectory: {}, exp.pt: {}, actual pt: {}, due: {}".format(
            self.env.now,
            self.j_idx,
            self.trajectory,
            self.remaining_pt,
            self.actual_remaining_pt,
            self.due,
        )
        self.logger.info(msg)
        self.recorder.record_event(self.env.now, "INFO", "JOB_CREATED", j_idx=self.j_idx, message=msg)

    def after_arrival(self) -> None:
        """Record the timestamp when the job arrives at a machine's queue."""
        self.arrival_T = self.env.now

    def after_decision(
        self,
        m_idx: int,
        wait: float,
        decision_type: str = "Passive",
        queue_len: int = 1,
        scheduled_begin_t: Optional[float] = None,
    ) -> None:
        """
        Called immediately after a machine picks this job for processing.
        Records the operation start (using actual processing time) and updates expected availability.
        """
        op_seq = len(self.operation_record)
        expected_pt = float(self.remaining_pt[0])
        actual_pt = float(self.actual_remaining_pt[0])
        begin_t = round(float(self.env.now), 2)
        end_t = round(float(self.env.now + actual_pt), 2)

        # The recorded operation history uses the actual processing time, NOT the expected value
        self.record_operation(m_idx, self.env.now, actual_pt, wait)
        # Also append structured operation tuple for the `operations` SQLite table
        self.recorder.operations_list.append(
            (
                int(self.j_idx),
                int(op_seq),
                int(m_idx),
                str(decision_type),
                int(queue_len),
                round(float(self.arrival_T), 2),
                begin_t,
                end_t,
                round(float(wait), 2),
                expected_pt,
                actual_pt,
                round(float(scheduled_begin_t), 2) if scheduled_begin_t is not None else None,
            )
        )
        # Update job status to processing
        self.status = "processing"
        # Expected available time visible to the centralized scheduler (uses nominal/expected PT)
        self.available_T = round(self.env.now + expected_pt, 1)

    def after_operation(self, *args) -> int:
        """
        Called upon completion of the current operation on a machine.
        Pops the finished operation and returns the next machine index, or -1 if the job is completed.
        """
        # Refresh availability time to current simulation time (handles early/late completion under pt_cv > 0)
        self.available_T = self.env.now

        # If the job still has subsequent operations to complete
        if len(self.remaining_operations) > 1:
            self.status = "queuing"
            # Pop the completed operation from all tracking stacks
            self.remaining_operations.pop(0)
            self.remaining_machines.pop(0)
            self.remaining_pt.pop(0)
            self.actual_remaining_pt.pop(0)
            # Retrieve the machine index for the next operation
            next_m = self.remaining_machines[0]
            return next_m
        else:
            # All operations are finished; mark completed and exit the system
            self.status = "completed"
            self.completion()
            return -1

    def record_operation(self, *args) -> None:
        """Append an operation entry [m_idx, begin_T, actual_pt, wait_T] to the job's history."""
        self.operation_record.append([*args])

    def completion(self) -> None:
        """Finalize metrics in the shared Recorder when all operations of the job are complete."""
        self.completion_T = round(float(self.env.now), 2)
        self.tardiness = max(0.0, round(self.env.now - self.due, 2))
        self.recorder.j_operation_dict[self.j_idx] = self.operation_record
        self.recorder.j_tardiness_dict[self.j_idx] = self.tardiness
        self.recorder.j_flowtime_dict[self.j_idx] = round(self.env.now - self.creation_T, 2)
        self.recorder.last_job_comp_T = self.env.now
        self.recorder.in_system_jobs.pop(self.j_idx, None)
        msg = "{} > END: Job {} completed".format(self.env.now, self.j_idx)
        self.logger.info(msg)
        self.recorder.record_event(self.env.now, "INFO", "JOB_END", j_idx=self.j_idx, message=msg)

    def overstay(self) -> None:
        """Remove a job that has exceeded the maximum allowable tardiness threshold from the system."""
        self.status = "overstay"
        self.completion_T = round(float(self.env.now), 2)
        self.tardiness = max(0.0, round(self.env.now - self.due, 2))
        self.recorder.j_operation_dict[self.j_idx] = self.operation_record
        self.recorder.j_tardiness_dict[self.j_idx] = self.tardiness
        self.recorder.j_flowtime_dict[self.j_idx] = round(self.env.now - self.creation_T, 2)
        self.recorder.last_job_comp_T = self.env.now
        self.recorder.in_system_jobs.pop(self.j_idx, None)
        msg = "{} > Job {} is removed from system due to over-stay!".format(self.env.now, self.j_idx)
        self.logger.warning(msg)
        self.recorder.record_event(self.env.now, "WARNING", "JOB_OVERSTAY", j_idx=self.j_idx, message=msg)

