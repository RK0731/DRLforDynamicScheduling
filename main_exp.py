import logging
import os

import simulator.job as JOB
import simulator.machine as MACHINE

class Factory:
    def __init__(self):
        # create the common logger
        self.sim_logger = logging.getLogger(__name__)
        filehandler = logging.FileHandler(os.path.join(['log', 'simulation.log']))
        filehandler.setFormatter(
            logging.Formatter('%(asctime)s [%(module)s: %(lineno)-3d] %(levelname)-5s >>> %(message)s'))
        self.sim_logger.addHandler(filehandler)
        # then create all assets on shop floor
        self.initialize_assets()

    
    def initialize_assets(self):
        pass


if __name__ == '__main__':
    pass