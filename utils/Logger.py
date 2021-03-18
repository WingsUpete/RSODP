"""
Logger that logs to both stdout and a file
"""
import logging
from datetime import datetime
import sys
import os

LOGGING_FOLDER = '../log'
if not os.path.isdir(LOGGING_FOLDER):
    os.mkdir(LOGGING_FOLDER)


class Logger:
    def __init__(self):
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Create a File Handler
        curT = datetime.now()
        curTStr = curT.strftime("%Y%m%d_%H_%M_%S")
        fName = '{}.log'.format(curTStr)
        print(os.path.join(LOGGING_FOLDER, fName))
        self.f_handler = logging.FileHandler(os.path.join(LOGGING_FOLDER, fName))

        # Create an stdout Stream Handler
        self.stdout_handler = logging.StreamHandler(sys.stdout)

        self.logger.addHandler(self.f_handler)
        self.logger.addHandler(self.stdout_handler)

    """
        Log a message into stdout and log file simultaneously
    """
    def log(self, message):
        self.logger.debug(message)

    def close(self):
        self.f_handler.close()
        self.stdout_handler.close()


if __name__ == '__main__':
    """
        Test
    """
    logger = Logger()
    logger.log('Test test')
    logger.log('Test 0')
    logger.log('Test 1')
    logger.close()
