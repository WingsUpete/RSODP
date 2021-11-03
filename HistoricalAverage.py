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


def batch2device(record: dict, target_G: torch.Tensor, target_D: torch.Tensor, device):
    """ Transfer all sample data into the device (cpu/gpu) """
    # Transfer record
    for temp_feat in Config.TEMP_FEAT_NAMES:
        record[temp_feat] = [(D.to(device), G.to(device)) for (D, G) in record[temp_feat]]

    # Transfer target
    target_G = target_G.to(device)
    target_D = target_D.to(device)

    return record, target_G, target_D


def avgRec(records: dict):
    # Aggregate features for each temporal feature set
    res0 = {}
    for temp_feat in Config.TEMP_FEAT_NAMES:
        curDList = [records[temp_feat][i][0] for i in range(len(records[temp_feat]))]
        curGList = [records[temp_feat][i][1] for i in range(len(records[temp_feat]))]
        avgD = sum(curDList) / len(curDList)
        avgG = sum(curGList) / len(curGList)
        res0[temp_feat] = (avgD, avgG)

    # Aggregate features altogether
    allD = [res0[temp_feat][0] for temp_feat in res0]
    allG = [res0[temp_feat][1] for temp_feat in res0]
    avgD = sum(allD) / len(allD)
    avgG = sum(allG) / len(allG)

    return avgD, avgG


def HA(bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, logr=Logger(activate=False), use_gpu=True,
       data_dir=Config.DATA_DIR_DEFAULT, total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include RMSE, MAPE, MAE
    """
    # Historical Average
    logr.log('> Using HA (Historical Average) baseline model.\n')

    # CUDA if needed
    device = torch.device('cuda:0' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir, his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT, time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT, total_H=total_H, start_at=start_H)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = GraphDataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # 1.
    # Metrics with thresholds
    num_metrics_threshold = len(Config.EVAL_METRICS_THRESHOLD_SET)
    metrics_res = {'Demand': {}, 'OD': {}}
    for metrics_for_what in metrics_res:
        metrics_res[metrics_for_what] = {
            'RMSE': torch.zeros(num_metrics_threshold),
            'MAPE': torch.zeros(num_metrics_threshold),
            'MAE': torch.zeros(num_metrics_threshold),
        }
    if device:
        metrics_thresholds = [torch.Tensor([threshold]).to(device) for threshold in Config.EVAL_METRICS_THRESHOLD_SET]
    # Clean GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    with torch.no_grad():
        for j, val_batch in enumerate(validloader):
            val_record, val_target_G, val_target_D = val_batch['record_GD'], val_batch['target_G'], val_batch['target_D']
            if device:
                val_record, val_target_G, val_target_D = batch2device(val_record, val_target_G, val_target_D, device)

            val_res_D, val_res_G = avgRec(val_record)

            for mi in range(num_metrics_threshold):     # for the (mi)th threshold
                metrics_res['Demand']['RMSE'][mi] += RMSE(val_res_D, val_target_D, metrics_thresholds[mi]).item()
                metrics_res['Demand']['MAPE'][mi] += MAPE(val_res_D, val_target_D, metrics_thresholds[mi]).item()
                metrics_res['Demand']['MAE'][mi] += MAE(val_res_D, val_target_D, metrics_thresholds[mi]).item()
                metrics_res['OD']['RMSE'][mi] += RMSE(val_res_G, val_target_G, metrics_thresholds[mi]).item()
                metrics_res['OD']['MAPE'][mi] += MAPE(val_res_G, val_target_G, metrics_thresholds[mi]).item()
                metrics_res['OD']['MAE'][mi] += MAE(val_res_G, val_target_G, metrics_thresholds[mi]).item()

        for metrics_for_what in metrics_res:
            for metrics in metrics_res[metrics_for_what]:
                metrics_res[metrics_for_what][metrics] /= len(validloader)

        logr.log('> Metrics Evaluations for Validation Set:\n')
        for metrics_for_what in metrics_res:
            logr.log('%s:\n' % metrics_for_what)
            for metrics in metrics_res[metrics_for_what]:
                for mi in range(num_metrics_threshold):
                    cur_threshold = Config.EVAL_METRICS_THRESHOLD_SET[mi]
                    logr.log('%s-%d = %.4f%s' % (metrics,
                                                 int(cur_threshold * cur_threshold) if metrics == 'RMSE' else cur_threshold,
                                                 metrics_res[metrics_for_what][metrics][mi],
                                                 (', ' if mi != num_metrics_threshold - 1 else '\n')))

    # 2.
    # Metrics with thresholds
    for metrics_for_what in metrics_res:
        metrics_res[metrics_for_what] = {
            'RMSE': torch.zeros(num_metrics_threshold),
            'MAPE': torch.zeros(num_metrics_threshold),
            'MAE': torch.zeros(num_metrics_threshold),
        }
    # Clean GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    with torch.no_grad():
        for k, test_batch in enumerate(testloader):
            test_record, test_target_G, test_target_D = test_batch['record_GD'], test_batch['target_G'], test_batch['target_D']
            if device:
                test_record, test_target_G, test_target_D = batch2device(test_record, test_target_G, test_target_D, device)

            test_res_D, test_res_G = avgRec(test_record)

            for mi in range(num_metrics_threshold):     # for the (mi)th threshold
                metrics_res['Demand']['RMSE'][mi] += RMSE(test_res_D, test_target_D, metrics_thresholds[mi]).item()
                metrics_res['Demand']['MAPE'][mi] += MAPE(test_res_D, test_target_D, metrics_thresholds[mi]).item()
                metrics_res['Demand']['MAE'][mi] += MAE(test_res_D, test_target_D, metrics_thresholds[mi]).item()
                metrics_res['OD']['RMSE'][mi] += RMSE(test_res_G, test_target_G, metrics_thresholds[mi]).item()
                metrics_res['OD']['MAPE'][mi] += MAPE(test_res_G, test_target_G, metrics_thresholds[mi]).item()
                metrics_res['OD']['MAE'][mi] += MAE(test_res_G, test_target_G, metrics_thresholds[mi]).item()

        for metrics_for_what in metrics_res:
            for metrics in metrics_res[metrics_for_what]:
                metrics_res[metrics_for_what][metrics] /= len(testloader)

        logr.log('> Metrics Evaluations for Test Set:\n')
        for metrics_for_what in metrics_res:
            logr.log('%s:\n' % metrics_for_what)
            for metrics in metrics_res[metrics_for_what]:
                for mi in range(num_metrics_threshold):
                    cur_threshold = Config.EVAL_METRICS_THRESHOLD_SET[mi]
                    logr.log('%s-%d = %.4f%s' % (metrics,
                                                 int(cur_threshold * cur_threshold) if metrics == 'RMSE' else cur_threshold,
                                                 metrics_res[metrics_for_what][metrics][mi],
                                                 (', ' if mi != num_metrics_threshold - 1 else '\n')))


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
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    # HA
    HA(bs=FLAGS.batch_size, num_workers=FLAGS.cores, logr=logger, use_gpu=(FLAGS.gpu == 1),
       data_dir=FLAGS.data_dir, total_H=FLAGS.hours, start_H=FLAGS.start_hour)
    logger.close()
