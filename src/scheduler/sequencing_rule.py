"""
Heuristic sequencing rules and placeholders for centralized mathematical optimization
and Deep Reinforcement Learning (DRL) schedulers.
Each heuristic method accepts `jobs` (a list of `Job` instances waiting in a machine's queue)
and returns the queue index (`int`) of the selected job.
"""

import inspect
import numpy as np


class SequencingMethod:
    @classmethod
    def FIFO(cls, jobs, *args, **kwargs) -> int:
        """First-In, First-Out: pick the earliest arriving job in the queue."""
        return 0

    @classmethod
    def LIFO(cls, jobs, *args, **kwargs) -> int:
        """Last-In, First-Out: pick the most recently arrived job in the queue."""
        return -1

    @classmethod
    def Slack(cls, jobs, *args, **kwargs) -> int:
        """Minimum Slack / Earliest Due Date (EDD): pick the job with the smallest time remaining until its due date."""
        s = np.array([j.due for j in jobs]) - jobs[0].env.now
        return int(np.argmin(s))

    @classmethod
    def CR(cls, jobs, *args, **kwargs) -> int:
        """
        Critical Ratio (CR): pick the job with the smallest ratio of time-to-due (TTD)
        to total remaining processing time.
        """
        ttd = np.array([j.due for j in jobs]) - jobs[0].env.now
        sum_pt = np.maximum(np.array([sum(j.remaining_pt) for j in jobs]), 1e-6)
        critical_ratio = ttd / sum_pt
        return int(np.argmin(critical_ratio))

    @classmethod
    def SPT(cls, jobs, *args, **kwargs) -> int:
        """Shortest Processing Time (SPT): pick the job with the shortest processing time on the current machine."""
        pt = np.array([j.remaining_pt[0] for j in jobs])
        return int(np.argmin(pt))

    @classmethod
    def LWKR(cls, jobs, *args, **kwargs) -> int:
        """Least Work Remaining (LWKR): pick the job with the smallest total remaining processing time."""
        rem_work = np.array([sum(j.remaining_pt) for j in jobs])
        return int(np.argmin(rem_work))

    @classmethod
    # Placeholder for Gurobi optimizer; Narrator replaces this with `CentralScheduler.draw_from_schedule`
    def GurobiOptimizer(cls, jobs, *args, **kwargs):
        return

    @classmethod
    # Placeholder for Google OR-Tools; Narrator replaces this with `CentralScheduler.draw_from_schedule`
    def ORTools(cls, jobs, *args, **kwargs):
        return

    @classmethod
    # Placeholder for DRL scheduler; replaced by DRL policy action selection during training/inference
    def DRL_scheduler(cls, jobs, *args, **kwargs):
        return


if __name__ == '__main__':
    print(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
