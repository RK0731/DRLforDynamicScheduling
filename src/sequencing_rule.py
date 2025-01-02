import inspect
import numpy as np

class SequencingMethod:
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
    # place holder for Gurobi optimizer, will use the draw_from_schedule function after creating a central scheduler object
    def GurobiOptimizer(cls, jobs, *args, **kwargs): 
        pass

    @classmethod 
    # place holder for Google OR-Tools, will use the draw_from_schedule function after creating a central scheduler object
    def ORTools(cls, jobs, *args, **kwargs): 
        pass

    @classmethod 
    # place holder, will use the function after creating a DRL scheduler
    def DRL_scheduler(cls, jobs, *args, **kwargs): 
        pass


if __name__ == '__main__':
    print(inspect.getmembers(SequencingMethod, predicate=inspect.ismethod))