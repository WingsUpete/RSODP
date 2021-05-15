import os
import sys
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.profiler as profiler

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.dataloading import GraphDataLoader
sys.stderr.close()
sys.stderr = stderr

from utils import Logger
from RSODPDataSet import RSODPDataSet
from model import Gallat

import Config


def batch2device(record: dict, query: torch.Tensor, target_G: torch.Tensor, target_D: torch.Tensor, device):
    """ Transfer all sample data into the device (cpu/gpu) """
    # Transfer record
    for temp_feat in Config.TEMP_FEAT_NAMES:
        record[temp_feat] = [(fg.to(device), bg.to(device), gg.to(device)) for (fg, bg, gg) in record[temp_feat]]

    # Transfer query
    query = query.to(device)

    # Transfer target
    target_G = target_G.to(device)
    target_D = target_D.to(device)

    return record, query, target_G, target_D


def train(lr=Config.LEARNING_RATE_DEFAULT, bs=Config.BATCH_SIZE_DEFAULT, ep=Config.MAX_EPOCHS_DEFAULT,
          eval_freq=Config.EVAL_FREQ_DEFAULT, opt=Config.OPTIMIZER_DEFAULT, num_workers=Config.WORKERS_DEFAULT,
          use_gpu=True, data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False), model=Config.NETWORK_DEFAULT,
          model_save_dir=Config.MODEL_SAVE_DIR_DEFAULT, pretrain=False, metrics_threshold=Config.METRICS_THRESHOLD_DEFAULT,
          total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H, hidden_dim=Config.HIDDEN_DIM_DEFAULT,
          feat_dim=Config.FEAT_DIM_DEFAULT, query_dim=Config.QUERY_DIM_DEFAULT,
          scale_factor_d=Config.SCALE_FACTOR_DEFAULT_D, scale_factor_g=Config.SCALE_FACTOR_DEFAULT_G):
    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir, his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT, time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT, total_H=total_H, start_at=start_H)
    trainloader = GraphDataLoader(dataset.train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Total Hours: {}, staring from {}\n'.format(total_H, start_H))
    logr.log('> Training batches: {}, Validation batches: {}\n'.format(len(trainloader), len(validloader)))

    # Initialize the Model
    logr.log('> Initializing the Training Model: {}, Pretrain = {}\n'.format(model, pretrain))
    predict_G = (not pretrain)
    net = Gallat(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim)
    if model == 'Gallat':
        net = Gallat(feat_dim=feat_dim, query_dim=query_dim, hidden_dim=hidden_dim)
    logr.log('> Model Structure:\n{}\n'.format(net))

    # Select Optimizer
    logr.log('> Constructing the Optimizer: {}\n'.format(opt))
    optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)  # Default: Adam + L2 Norm
    if opt == 'ADAM':
        optimizer = torch.optim.Adam(net.parameters(), lr, weight_decay=Config.WEIGHT_DECAY_DEFAULT)    # Adam + L2 Norm

    # Loss Function
    logr.log('> Using SmoothL1Loss as the Loss Function.\n')
    criterion_D = nn.SmoothL1Loss()
    criterion_G = nn.SmoothL1Loss()

    # CUDA if possible
    device = torch.device('cuda:0' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Scale Factor
    if device:
        scale_factor_d = scale_factor_d.to(device)
        scale_factor_g = scale_factor_g.to(device)

    # Model Saving Directory
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)

    # Metrics
    metrics_threshold_val = metrics_threshold.item()
    if device:
        metrics_threshold = metrics_threshold.to(device)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}, num_workers = {}\n'.format(lr, ep, num_workers))
    logr.log('eval_freq = {}, batch_size = {}, optimizer = {}\n'.format(eval_freq, bs, opt))
    logr.log('scaling Factor for: d - %.2f, g - %.2f' % (scale_factor_d.item(), scale_factor_g.item()))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    min_eval_loss = float('inf')

    for epoch_i in range(ep):
        # train one round
        net.train()
        train_loss = 0
        train_rmse = 0
        train_mape = 0
        train_mae = 0
        time_start_train = time.time()
        for i, batch in enumerate(trainloader):
            record, query, target_G, target_D = batch['record'], batch['query'], batch['target_G'], batch['target_D']
            if device:
                record, query, target_G, target_D = batch2device(record, query, target_G, target_D, device)

            # Avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()

            # with profiler.profile(profile_memory=True, use_cuda=True) as prof:
            #     with profiler.record_function('model_inference'):
            #         res_D, res_G = net(record, query, predict_G=predict_G)   # if pretrain, res_G = None
            #         loss = criterion_D(res_D, target_D) if pretrain else (criterion_D(res_D, target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(res_G, target_G) * Config.G_PERCENTAGE_DEFAULT)
            # logr.log(prof.key_averages().table(sort_by="cuda_time_total"))

            res_D, res_G = net(record, query, predict_G=predict_G)  # if pretrain, res_G = None
            loss = criterion_D(res_D * scale_factor_d, target_D) if pretrain else (criterion_D(res_D * scale_factor_d, target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(res_G * scale_factor_g, target_G) * Config.G_PERCENTAGE_DEFAULT)

            loss.backward()
            optimizer.step()

            # Analysis
            with torch.no_grad():
                train_loss += loss.item()
                train_rmse += RMSE(res_D * scale_factor_d, target_D, metrics_threshold).item() if pretrain else ((RMSE(res_D * scale_factor_d, target_D, metrics_threshold) + RMSE(res_G * scale_factor_g, target_G, metrics_threshold)) / 2).item()
                train_mape += MAPE(res_D * scale_factor_d, target_D, metrics_threshold).item() if pretrain else ((MAPE(res_D * scale_factor_d, target_D, metrics_threshold) + MAPE(res_G * scale_factor_g, target_G, metrics_threshold)) / 2).item()
                train_mae += MAE(res_D * scale_factor_d, target_D, metrics_threshold).item() if pretrain else ((MAE(res_D * scale_factor_d, target_D, metrics_threshold) + MAE(res_G * scale_factor_g, target_G, metrics_threshold)) / 2).item()

            # if i == 0:    # DEBUG
            #     break

        # Analysis after one training round in the epoch
        train_loss /= len(trainloader)
        train_rmse /= len(trainloader)
        train_mape /= len(trainloader)
        train_mae /= len(trainloader)
        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        train_time_per_sample = total_train_time / len(dataset.train_set)
        logr.log('Training Round %d: loss = %.6f, time_cost = %.4f sec (%.4f sec per sample), RMSE-%d = %.4f, MAPE-%d = %.4f, MAE-%d = %.4f\n' % (epoch_i, train_loss, total_train_time, train_time_per_sample, metrics_threshold_val, train_rmse, metrics_threshold_val, train_mape, metrics_threshold_val, train_mae))

        # eval_freq: Evaluate on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            val_loss_total = 0
            val_rmse = 0
            val_mape = 0
            val_mae = 0
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            with torch.no_grad():
                for j, val_batch in enumerate(validloader):
                    val_record, val_query, val_target_G, val_target_D = val_batch['record'], val_batch['query'], val_batch['target_G'], val_batch['target_D']
                    if device:
                        val_record, val_query, val_target_G, val_target_D = batch2device(val_record, val_query, val_target_G, val_target_D, device)

                    val_res_D, val_res_G = net(val_record, val_query, predict_G=True)
                    val_loss = criterion_D(val_res_D * scale_factor_d, val_target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(val_res_G * scale_factor_g, val_target_G) * Config.G_PERCENTAGE_DEFAULT

                    val_loss_total += val_loss.item()
                    val_rmse += ((RMSE(val_res_D * scale_factor_d, val_target_D, metrics_threshold) + RMSE(val_res_G * scale_factor_g, val_target_G, metrics_threshold)) / 2).item()
                    val_mape += ((MAPE(val_res_D * scale_factor_d, val_target_D, metrics_threshold) + MAPE(val_res_G * scale_factor_g, val_target_G, metrics_threshold)) / 2).item()
                    val_mae += ((MAE(val_res_D * scale_factor_d, val_target_D, metrics_threshold) + MAE(val_res_G * scale_factor_g, val_target_G, metrics_threshold)) / 2).item()

                val_loss_total /= len(validloader)
                val_rmse /= len(validloader)
                val_mape /= len(validloader)
                val_mae /= len(validloader)
                logr.log('!!! Validation : loss = %.6f, RMSE-%d = %.4f, MAPE-%d = %.4f, MAE-%d = %.4f\n' % (val_loss_total, metrics_threshold_val, val_rmse, metrics_threshold_val, val_mape, metrics_threshold_val, val_mae))

                if val_loss_total < min_eval_loss:
                    min_eval_loss = val_loss_total
                    model_name = os.path.join(model_save_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_name)
                    logr.log('Model: {} has been saved since it achieves smaller loss.\n'.format(model_name))

        # if epoch_i == 0:    # break
        #     break

    # End Training
    logr.log('Training finished.\n')


# TODO: put these methods into utils
def filter_with_threshold(x: torch.Tensor, threshold: torch.Tensor):
    """
    Filter out values below the threshold (they will become the threshold)
    :param x: a tensor
    :param threshold: single-value tensor containing the threshold
    :return: filtered tensor
    """
    return torch.max(x, threshold)


def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=Config.ZERO_TENSOR):
    """
    RMSE (Root Mean Squared Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is RMSE-3)
    :return: RMSE-threshold
    """
    y_pred_filter = filter_with_threshold(y_pred, threshold)
    y_true_filter = filter_with_threshold(y_true, threshold)
    return torch.sqrt(torch.mean(torch.pow((y_true_filter - y_pred_filter), 2)))


def MAE(y_pred, y_true, threshold=Config.ZERO_TENSOR):
    """
    MAE (Mean Absolute Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is MAE-3)
    :return: MAE-threshold
    """
    y_pred_filter = filter_with_threshold(y_pred, threshold)
    y_true_filter = filter_with_threshold(y_true, threshold)
    return torch.mean(torch.abs(y_true_filter - y_pred_filter))


def MAPE(y_pred, y_true, threshold=Config.ZERO_TENSOR):
    """
    MAPE (Mean Absolute Percentage Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is MAPE-3)
    :return: MAPE-threshold
    """
    y_pred_filter = filter_with_threshold(y_pred, threshold)
    y_true_filter = filter_with_threshold(y_true, threshold)
    return torch.mean(torch.abs((y_true_filter - y_pred_filter)/(y_true_filter + 1)))


def evaluate(model_name, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, use_gpu=True,
             data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False),
             total_H=Config.DATA_TOTAL_H, start_H=Config.DATA_START_H,
             scale_factor_d=Config.SCALE_FACTOR_DEFAULT_D, scale_factor_g=Config.SCALE_FACTOR_DEFAULT_G):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include RMSE, MAPE, MAE
    """
    # Load Model
    logr.log('Loading {}\n'.format(model_name))
    net = torch.load(model_name)

    # CUDA if needed
    device = torch.device('cuda:0' if (use_gpu and torch.cuda.is_available()) else 'cpu')
    logr.log('> device: {}\n'.format(device))

    if device:
        net.to(device)
        logr.log('> Model sent to {}\n'.format(device))

    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir, his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT,
                           time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT, total_H=total_H, start_at=start_H)
    validloader = GraphDataLoader(dataset.valid_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    testloader = GraphDataLoader(dataset.test_set, batch_size=bs, shuffle=False, num_workers=num_workers)
    logr.log('> Validation batches: {}, Test batches: {}\n'.format(len(validloader), len(testloader)))

    # # 1.
    # net.eval()
    # val_rmse = 0
    # val_mape = 0
    # val_mae = 0
    # if device.type == 'cuda':
    #     torch.cuda.empty_cache()
    # with torch.no_grad():
    #     for j, val_batch in enumerate(validloader):
    #         val_record, val_query, val_target_G, val_target_D = val_batch['record'], val_batch['query'], val_batch['target_G'], val_batch['target_D']
    #         if device:
    #             val_record, val_query, al_target_G, val_target_D = batch2device(val_record, val_query, val_target_G, val_target_D, device)
    #
    #         val_res_D, val_res_G = net(val_record, val_query)
    #
    #         val_rmse += ((RMSE(val_res_D, val_target_D, metrics_threshold) + RMSE(val_res_G, val_target_G, metrics_threshold)) / 2).item()
    #         val_mape += ((MAPE(val_res_D, val_target_D, metrics_threshold) + MAPE(val_res_G, val_target_G, metrics_threshold)) / 2).item()
    #         val_mae += ((MAE(val_res_D, val_target_D, metrics_threshold) + MAE(val_res_G, val_target_G, metrics_threshold)) / 2).item()
    #
    #     val_rmse /= len(validloader)
    #     val_mape /= len(validloader)
    #     val_mae /= len(validloader)
    #     logr.log('!!! Validation : loss = %.4f, RMSE-%d = %.4f, MAPE-%d = %.4f, MAE-%d = %.4f\n' % (val_loss_total, metrics_threshold_val, val_rmse, metrics_threshold_val, val_mape, metrics_threshold_val, val_mae))


if __name__ == '__main__':
    """ 
        Usage Example:
        python Trainer.py -dr data/ny2016_0101to0331/ -th 1064 -st 1 -c 4 -m train -net Gallat -pre 0
        python Trainer.py -dr data/ny2016_0101to0331/ -th 1064 -st 1 -c 4 -m eval -e model/20221225_06_06_06.pth
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-lr', '--learning_rate', type=float, default=Config.LEARNING_RATE_DEFAULT, help='Learning rate, default = {}'.format(Config.LEARNING_RATE_DEFAULT))
    parser.add_argument('-me', '--max_epochs', type=int, default=Config.MAX_EPOCHS_DEFAULT, help='Number of epochs to run the trainer, default = {}'.format(Config.MAX_EPOCHS_DEFAULT))
    parser.add_argument('-ef', '--eval_freq', type=int, default=Config.EVAL_FREQ_DEFAULT, help='Frequency of evaluation on the validation set, default = {}'.format(Config.EVAL_FREQ_DEFAULT))
    parser.add_argument('-bs', '--batch_size', type=int, default=Config.BATCH_SIZE_DEFAULT, help='Size of a minibatch, default = {}'.format(Config.BATCH_SIZE_DEFAULT))
    parser.add_argument('-opt', '--optimizer', type=str, default=Config.OPTIMIZER_DEFAULT, help='Optimizer to be used [ADAM], default = {}'.format(Config.OPTIMIZER_DEFAULT))
    parser.add_argument('-dr', '--data_dir', type=str, default=Config.DATA_DIR_DEFAULT, help='Root directory of the input data, default = {}'.format(Config.DATA_DIR_DEFAULT))
    parser.add_argument('-ld', '--log_dir', type=str, default=Config.LOG_DIR_DEFAULT, help='Specify where to create a log file. If log files are not wanted, value will be None'.format(Config.LOG_DIR_DEFAULT))
    parser.add_argument('-c', '--cores', type=int, default=Config.WORKERS_DEFAULT, help='number of workers (cores used), default = {}'.format(Config.WORKERS_DEFAULT))
    parser.add_argument('-gpu', '--gpu', type=int, default=Config.USE_GPU_DEFAULT, help='Specify whether to use GPU, default = {}'.format(Config.USE_GPU_DEFAULT))
    parser.add_argument('-net', '--network', type=str, default=Config.NETWORK_DEFAULT,  help='Specify which model/network to use, default = {}'.format(Config.NETWORK_DEFAULT))
    parser.add_argument('-m', '--mode', type=str, default=Config.MODE_DEFAULT, help='Specify which mode the discriminator runs in (train, eval), default = {}'.format(Config.MODE_DEFAULT))
    parser.add_argument('-e', '--eval', type=str, default=Config.EVAL_DEFAULT, help='Specify the location of saved network to be loaded for evaluation, default = {}'.format(Config.EVAL_DEFAULT))
    parser.add_argument('-md', '--model_save_dir', type=str, default=Config.MODEL_SAVE_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_SAVE_DIR_DEFAULT))
    parser.add_argument('-pre', '--pretrain', type=int, default=Config.PRETRAIN_DEFAULT, help='Specify whether to pretrain the model (only predict demands), default = {}'.format(Config.PRETRAIN_DEFAULT))
    parser.add_argument('-mt', '--metrics_threshold', type=int, default=Config.METRICS_THRESHOLD_DEFAULT, help='Specify the metrics threshold, default = {}'.format(Config.METRICS_THRESHOLD_DEFAULT))
    parser.add_argument('-th', '--hours', type=int, default=Config.DATA_TOTAL_H, help='Specify the number of hours for data, default = {}'.format(Config.DATA_TOTAL_H))
    parser.add_argument('-ts', '--start_hour', type=int, default=Config.DATA_START_H, help='Specify the starting hour for data, default = {}'.format(Config.DATA_START_H))
    parser.add_argument('-hd', '--hidden_dim', type=int, default=Config.HIDDEN_DIM_DEFAULT, help='Specify the hidden dimension, default = {}'.format(Config.HIDDEN_DIM_DEFAULT))
    parser.add_argument('-fd', '--feature_dim', type=int, default=Config.FEAT_DIM_DEFAULT, help='Specify the feature dimension, default = {}'.format(Config.FEAT_DIM_DEFAULT))
    parser.add_argument('-qd', '--query_dim', type=int, default=Config.QUERY_DIM_DEFAULT, help='Specify the query dimension, default = {}'.format(Config.QUERY_DIM_DEFAULT))
    parser.add_argument('-sfd', '--scale_factor_d', type=float, default=Config.SCALE_FACTOR_DEFAULT_D, help='scale factor for model output d, default = {}'.format(Config.SCALE_FACTOR_DEFAULT_D))
    parser.add_argument('-sfg', '--scale_factor_g', type=float, default=Config.SCALE_FACTOR_DEFAULT_G, help='scale factor for model output g, default = {}'.format(Config.SCALE_FACTOR_DEFAULT_G))
    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), data_dir=FLAGS.data_dir, logr=logger, model=FLAGS.network,
              model_save_dir=FLAGS.model_save_dir, pretrain=(FLAGS.pretrain == 1),
              metrics_threshold=torch.Tensor([FLAGS.metrics_threshold]),
              total_H=FLAGS.hours, start_H=FLAGS.start_hour, hidden_dim=FLAGS.hidden_dim,
              feat_dim=FLAGS.feature_dim, query_dim=FLAGS.query_dim,
              scale_factor_d=torch.Tensor([FLAGS.scale_factor_d]), scale_factor_g=torch.Tensor([FLAGS.scale_factor_g]))
        logger.close()
    elif working_mode == 'eval':
        eval_file = FLAGS.eval
        # Abnormal: file not found
        if (not eval_file) or (not os.path.isfile(eval_file)):
            sys.stderr.write('File for evaluation not found, please check!\n')
            logger.close()
            exit(-1)
        # Normal
        evaluate(eval_file, bs=FLAGS.batch_size, num_workers=FLAGS.cores, use_gpu=(FLAGS.gpu == 1),
                 data_dir=FLAGS.data_dir, logr=logger, total_H=FLAGS.hours, start_H=FLAGS.start_hour,
                 scale_factor_d=torch.Tensor([FLAGS.scale_factor_d]),
                 scale_factor_g=torch.Tensor([FLAGS.scale_factor_g]))
        logger.close()
    else:
        sys.stderr.write('Please specify the running mode (train/eval)\n')
        logger.close()
        exit(-2)
