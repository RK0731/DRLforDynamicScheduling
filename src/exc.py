"""
Customized exception classes
"""

# 400 series are used for errors in input data
class DataError(Exception):
    job_id = 'NOT_DEFINED'
    def __init__(self, message, status_code=460):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message


class InvalidRequestError(Exception):
    job_id = 'NOT_DEFINED'
    def __init__(self, message, status_code=480):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message


# 500 series are used for server problems
class UndefinedServerError(Exception):
    job_id = 'NOT_DEFINED'
    def __init__(self, message, status_code=500):
        self.message = message
        self.status_code = status_code
    def __str__(self):
        return self.message
