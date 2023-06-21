import inspect
import numpy as np

class SQC_rule:
    @classmethod
    def FIFO(cls, jobs, *args, **kwargs):
        return 0

    @classmethod
    def LIFO(cls, jobs, *args, **kwargs):
        return -1

    @classmethod
    def Slack(cls, jobs, *args, **kwargs):
        s = np.array([j.due for j in jobs]) - jobs[0].env.now
        return np.argmin(s)

    @classmethod
    def CR(cls, jobs, *args, **kwargs):
        ttd = np.array([j.due for j in jobs]) - jobs[0].env.now
        sum_pt = np.array([sum(j.remaining_pt) for j in jobs])
        critical_ratio = ttd / sum_pt
        return np.argmin(critical_ratio)
    
    @classmethod 
    # dummy function, will use the draw_from_schedule function after creating a central scheduler object
    def opt_scheduler(cls, jobs, *args, **kwargs): 
        pass

    @classmethod 
    # dummy function, will use the function after creating a DRL scheduler
    def DRL_scheduler(cls, jobs, *args, **kwargs): 
        pass


if __name__ == '__main__':
    print(inspect.getmembers(SQC_rule, predicate=inspect.ismethod))