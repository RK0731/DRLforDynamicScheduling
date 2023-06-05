import numpy as np
from dataclasses import dataclass

def FIFO(jobs, *args, **kwargs):
    return 0

def LIFO(jobs, *args, **kwargs):
    return -1

def Slack(jobs, *args, **kwargs):
    s = np.array([j.due for j in jobs]) - jobs[0].env.now
    return np.argmin(s)

def CR(jobs, *args, **kwargs):
    ttd = np.array([j.due for j in jobs]) - jobs[0].env.now
    sum_pt = np.array([sum(j.remaining_pt) for j in jobs])
    critical_ratio = ttd / sum_pt
    return np.argmin(critical_ratio)