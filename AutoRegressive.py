"""
A baseline model called AR - Auto-regressive: Predict with the considerations of the past p records
"""
import os
import sys
import argparse

import numpy as np
import torch
from statsmodels.tsa.ar_model import AutoReg

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from utils import Logger, batch2device, evalMetrics
from RSODPDataSet import RSODPDataSet

import Config


def AR(data_train, lag, n_channels, start, end):
    pred = np.array([AutoReg(data_train[:, fi], lags=lag).fit().predict(start=start, end=end)
                     for fi in range(n_channels)])

    return pred


def cal(recordGD, device):
    bs, num_nodes = recordGD['St'][-1][0].shape
    nRec = len(recordGD['St'])

    Dnp = np.array([recordGD['St'][i][0].reshape(-1).cpu().numpy() for i in range(nRec)])
    resD = torch.from_numpy(AR(Dnp, lag=0, n_channels=int(bs * num_nodes), start=nRec, end=nRec).reshape(bs, num_nodes)).to(device)

    Gnp = np.array([recordGD['St'][i][1].reshape(-1).cpu().numpy() for i in range(nRec)])
    resG = torch.from_numpy(AR(Gnp, lag=0, n_channels=int(bs * num_nodes * num_nodes), start=nRec, end=nRec).reshape(bs, num_nodes, num_nodes)).to(device)

    return resD, resG


def batch2res(batch, device, args):
    recordGD, target_G, target_D = batch['record_GD'], batch['target_G'], batch['target_D']
    if device:
        _, recordGD, _, target_G, target_D = batch2device(record=None, record_GD=recordGD, query=None,
                                                          target_G=target_G, target_D=target_D, device=device)

    res_D, res_G = cal(recordGD, device)
    return res_D, res_G, target_D, target_G


def evaluate(bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, logr=Logger(activate=False),
       use_gpu=True, gpu_id=Config.GPU_ID_DEFAULT,
       data_dir=Config.DATA_DIR_DEFAULT, total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H):
    """
        Evaluate using AR model
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include RMSE, MAPE, MAE
    """
    # CUDA if needed
    device = torch.device('cuda:%d' % gpu_id if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Autoregressive
    logr.log('> Using AR (Auto-Regressive) baseline model.\n')

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir, his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT, time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT, total_H=total_H, start_at=start_H)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = GraphDataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # 1.
    evalMetrics(validloader, 'Validation', batch2res, device, logr)

    # 2.
    evalMetrics(testloader, 'Test', batch2res, device, logr)


if __name__ == '__main__':
    """ 
        Usage Example:
        python AutoRegressive.py -dr data/ny2016_0101to0331/ -th 1064 -ts 1 -c 4 -bs 5
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a batch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-th', '--hours', type=int, default=Config.DATA_TOTAL_H, help='Specify the number of hours for data, default = {}'.format(Config.DATA_TOTAL_H))
    parser.add_argument('-ts', '--start_hour', type=int, default=Config.DATA_START_H, help='Specify the starting hour for data, default = {}'.format(Config.DATA_START_H))
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-gid', '--gpu_id', type=int, default=Config.GPU_ID_DEFAULT, help='Specify which GPU to use, default = {}'.format(Config.GPU_ID_DEFAULT))
    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    # HA
    evaluate(bs=FLAGS.batch_size, num_workers=FLAGS.cores, logr=logger, use_gpu=(FLAGS.gpu == 1), gpu_id=FLAGS.gpu_id,
             data_dir=FLAGS.data_dir, total_H=FLAGS.hours, start_H=FLAGS.start_hour)
    logger.close()
