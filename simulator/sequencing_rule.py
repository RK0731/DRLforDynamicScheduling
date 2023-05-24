import numpy as np

def FIFO(jobs):
    return 0

def LIFO(jobs):
    return -1

def Slack(jobs):
    s = np.array([j.due for j in jobs]) - jobs[0].env.now
    return np.argmin(s)
