import os
import sys
import argparse
import time

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
from dgl.data import DGLDataset
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
          model_dir=Config.MODEL_DIR_DEFAULT, pretrain=False):
    # Load DataSet
    logr.log('> Loading DataSet from {}\n'.format(data_dir))
    dataset = RSODPDataSet(data_dir, his_rec_num=Config.HISTORICAL_RECORDS_NUM_DEFAULT, time_slot_endurance=Config.TIME_SLOT_ENDURANCE_DEFAULT)
    trainloader = GraphDataLoader(dataset.train_set, batch_size=bs, shuffle=True, num_workers=num_workers)
    validloader = GraphDataLoader(dataset.train_set, batch_size=bs, shuffle=False, num_workers=num_workers)

    # Initialize the Model
    logr.log('> Initializing the Training Model: {}, Pretrain = {}\n'.format(model, pretrain))
    net = Gallat(feat_dim=Config.FEAT_DIM_DEFAULT, query_dim=Config.QUERY_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, predict_G=(not pretrain))
    if model == 'Gallat':
        net = Gallat(feat_dim=Config.FEAT_DIM_DEFAULT, query_dim=Config.QUERY_DIM_DEFAULT, hidden_dim=Config.HIDDEN_DIM_DEFAULT, predict_G=(not pretrain))

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

    # Model Saving Directory
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # Summarize Info
    logr.log('\nlearning_rate = {}, epochs = {}, num_workers = {}\n'.format(lr, ep, num_workers))
    logr.log('eval_freq = {}, batch_size = {}, optimizer = {}\n'.format(eval_freq, bs, opt))

    # Start Training
    logr.log('\nStart Training!\n')
    logr.log('------------------------------------------------------------------------\n')

    min_eval_loss = float('inf')

    for epoch_i in range(ep):
        # train one round
        net.train()
        train_loss = 0
        time_start_train = time.time()
        for i, batch in enumerate(trainloader):
            record, query, target_G, target_D = batch['record'], batch['query'], batch['target_G'], batch['target_D']
            if device:
                record, query, target_G, target_D = batch2device(record, query, target_G, target_D, device)

            # Avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=Config.MAX_NORM_DEFAULT)

            optimizer.zero_grad()
            if pretrain:
                res_D = net(record, query)
                loss = criterion_D(res_D, target_D)
            else:
                res_D, res_G = net(record, query)
                loss = criterion_D(res_D, target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(res_G, target_G) * Config.G_PERCENTAGE_DEFAULT
            loss.backward()
            optimizer.step()

            # Analysis
            train_loss += loss.item()
        train_loss /= len(trainloader)
        time_end_train = time.time()
        total_train_time = (time_end_train - time_start_train)
        train_time_per_sample = total_train_time / len(dataset.train_set)
        logr.log('Training Round %d: loss = %.4f, time_cost = %.4f sec (%.4f sec per sample)\n' % (epoch_i, train_loss, total_train_time, train_time_per_sample))

        # Evaluate Frequency on validation set
        if (epoch_i + 1) % eval_freq == 0:
            net.eval()
            val_loss_total = 0
            with torch.no_grad():
                for j, val_batch in enumerate(validloader):
                    val_record, val_query, val_target_G, val_target_D = val_batch['record'], val_batch['query'], val_batch['target_G'], val_batch['target_D']
                    if device:
                        val_record, val_query, al_target_G, val_target_D = batch2device(val_record, val_query, val_target_G, val_target_D, device)

                    if pretrain:
                        val_res_D = net(val_record, val_query)
                        val_loss = criterion_D(val_res_D, val_target_D)
                    else:
                        val_res_D, val_res_G = net(val_record, val_query)
                        val_loss = criterion_D(val_res_D, val_target_D) * Config.D_PERCENTAGE_DEFAULT + criterion_G(val_res_G, val_target_G) * Config.G_PERCENTAGE_DEFAULT
                    val_loss_total += val_loss.item()
                val_loss_total /= len(validloader)
                logr.log('!!! Validation : loss = %.4f\n' % val_loss_total)

                if val_loss_total < min_eval_loss:
                    min_eval_loss = val_loss_total
                    model_name = os.path.join(model_dir, '{}.pth'.format(logr.time_tag))
                    torch.save(net, model_name)
                    logr.log('Model: {} has been saved since it achieves smaller loss.\n'.format(model_name))


def evaluate(model_name, bs=Config.BATCH_SIZE_DEFAULT, num_workers=Config.WORKERS_DEFAULT, use_gpu=True,
             data_dir=Config.DATA_DIR_DEFAULT, logr=Logger(activate=False)):
    """
        Evaluate using saved best model (Note that this is a Test API)
        1. Re-evaluate on the validation set
        2. Re-evaluate on the test set
        The evaluation metrics include RMSE, MAPE, MAE
    """
    pass


if __name__ == '__main__':
    """ 
        Usage Example:
        python Trainer.py -dr data/ny2016_0101to0331/ -c 4 -m train -net Gallat -pre 0
        python Trainer.py -dr data/ny2016_0101to0331/ -c 4 -m eval -e model/20221225_06_06_06.pth
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
    parser.add_argument('-md', '--model_dir', type=str, default=Config.MODEL_DIR_DEFAULT, help='Specify the location of network to be saved, default = {}'.format(Config.MODEL_DIR_DEFAULT))
    parser.add_argument('-pre', '--pretrain', type=int, default=Config.PRETRAIN_DEFAULT, help='Specify whether to pretrain the model (only predict demands), default = {}'.format(Config.PRETRAIN_DEFAULT))

    FLAGS, unparsed = parser.parse_known_args()

    # Starts a log file in the specified directory
    logger = Logger(activate=True, logging_folder=FLAGS.log_dir) if FLAGS.log_dir else Logger(activate=False)

    working_mode = FLAGS.mode
    if working_mode == 'train':
        train(lr=FLAGS.learning_rate, bs=FLAGS.batch_size, ep=FLAGS.max_epochs,
              eval_freq=FLAGS.eval_freq, opt=FLAGS.optimizer, num_workers=FLAGS.cores,
              use_gpu=(FLAGS.gpu == 1), data_dir=FLAGS.data_dir, logr=logger, model=FLAGS.network,
              model_dir=FLAGS.model_dir, pretrain=(FLAGS.pretrain == 1))
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
                 data_dir=FLAGS.data_dir, logr=logger)
        logger.close()
    else:
        sys.stderr.write('Please specify the running mode (train/eval)\n')
        logger.close()
        exit(-2)