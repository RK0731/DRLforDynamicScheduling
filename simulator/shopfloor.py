import simpy
import pandas as pd
import logging
import time
import os
import argparse

import simulator.job as JOB
import simulator.machine as MACHINE

class Shopfloor:
    def __init__(self, **kwargs):
        # create the common logger
        self.sim_logger = logging.getLogger(__name__)
        ct = '_'.join([str(x) for x in time.strftime("%Y,%m,%d,%H,%M,%S").split(',')])
        filehandler = logging.FileHandler(os.path.join(['log', '{}_simulation.log'.format(ct)]))
        filehandler.setFormatter(
            logging.Formatter('%(asctime)s [%(module)s: %(lineno)-3d] %(levelname)-5s >>> %(message)s'))
        self.sim_logger.addHandler(filehandler)
        # then create all assets on shop floor
        self.initialize_assets()


    def initialize_assets(self):
        self.env = simpy.Environment()

    
    def event_creator(self, m_list, j_parameter):
        pass
        

    def run_simulation(self):
        self.env.run()


if __name__ == '__main__':
    pass