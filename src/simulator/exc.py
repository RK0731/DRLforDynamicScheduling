"""
Customized exception classes
"""

class InvalidRequestError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message
    
class SimulatorError(Exception):
    def __init__(self, message, status_code=400):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message

# 500 series are used for server problems
class ResultError(Exception):
    job_id = 'NOT_DEFINED'
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message
