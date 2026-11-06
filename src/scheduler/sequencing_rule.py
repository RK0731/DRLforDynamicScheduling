"""
Reactive heuristic sequencing rules for job selection at a workstation.

Each rule accepts `jobs` (the list of `Job` instances waiting in a machine's queue) and
returns the queue index (`int`) of the selected job.

Centralized mathematical-optimization schedulers (Gurobi MILP / OR-Tools CP-SAT) are
deliberately NOT defined here. They live in `src/scheduler/scheduler.py` and are looked
up through `OPTIMIZATION_SOLVERS`, so this class stays a clean collection of pure
dispatching heuristics. The DRL policy is likewise supplied as a callable by `Brain`
(it sets each machine's `job_sequencing` directly), not as a dummy rule here.
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
        # Return a concrete (non-negative) queue index rather than Python's -1 alias:
        # the index is later reused by `Machine.after_operation` to pop the processed
        # job, and a stale `-1` would pop whatever happens to be last at that time.
        return len(jobs) - 1

    @classmethod
    def Slack(cls, jobs, *args, **kwargs) -> int:
        """
        Minimum Slack: pick the job with the smallest slack,
        ``slack = time_to_due - total_remaining_processing_time``.

        Note: the previous implementation returned the earliest due date (EDD), not
        slack, because it omitted the remaining work term. EDD is still available as a
        special case when all remaining work is equal; for a true slack rule the work
        term must be subtracted.
        """
        ttd = np.array([j.due for j in jobs]) - jobs[0].env.now
        remaining_work = np.array([sum(j.remaining_pt) for j in jobs])
        slack = ttd - remaining_work
        return int(np.argmin(slack))

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

if __name__ == '__main__':
    print(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))
