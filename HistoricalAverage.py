"""
A baseline model called HA - Historical Average: Calculate average values according to temporal feature sets
"""
import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from utils import Logger, RMSE, MAE, MAPE
from RSODPDataSet import RSODPDataSet

import Config


def HA(bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, logr=Logger(activate=False),
       data_dir=Config.DATA_DIR_DEFAULT, total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H,
       scale_factor_d=Config.SCALE_FACTOR_DEFAULT_D, scale_factor_g=Config.SCALE_FACTOR_DEFAULT_G):
    pass


if __name__ == '__main__':
    """ 
        Usage Example:
        python HistoricalAverage.py -dr data/ny2016_0101to0331/ -th 1064 -ts 1 -c 4 -bs 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a batch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-th', '--hours', type=int, default=Config.DATA_TOTAL_H, help='Specify the number of hours for data, default = {}'.format(Config.DATA_TOTAL_H))
    parser.add_argument('-ts', '--start_hour', type=int, default=Config.DATA_START_H, help='Specify the starting hour for data, default = {}'.format(Config.DATA_START_H))
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-sfd', '--scale_factor_d', type=float, default=Config.SCALE_FACTOR_DEFAULT_D, help='scale factor for model output d, default = {}'.format(Config.SCALE_FACTOR_DEFAULT_D))
    parser.add_argument('-sfg', '--scale_factor_g', type=float, default=Config.SCALE_FACTOR_DEFAULT_G, help='scale factor for model output g, default = {}'.format(Config.SCALE_FACTOR_DEFAULT_G))
    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    # HA
    HA(bs=FLAGS.batch_size, num_workers=FLAGS.cores, logr=logger,
       data_dir=FLAGS.data_dir, total_H=FLAGS.hours, start_H=FLAGS.start_hour,
       scale_factor_d=FLAGS.scale_factor_d, scale_factor_g=FLAGS.scale_factor_g)
    logger.close()
