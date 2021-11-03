"""
Utility functions
"""
import math

import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ZERO_TENSOR = torch.Tensor([0])


def haversine(c0, c1):
    """
    :param c0: coordinate 0 in form (lat0, lng0) with degree as unit
    :param c1: coordinate 1 in form (lat1, lng1) with degree as unit
    :return: The haversine distance of c0 and c1 in km
    Compute the haversine distance between
    https://en.wikipedia.org/wiki/Haversine_formula
    """
    dLat = math.radians(c1[0] - c0[0])
    dLng = math.radians(c1[1] - c0[1])
    lat0 = math.radians(c0[0])
    lat1 = math.radians(c1[0])
    form0 = math.pow(math.sin(dLat / 2), 2)
    form1 = math.cos(lat0) * math.cos(lat1) * math.pow(math.sin(dLng / 2), 2)
    radius_of_earth = 6371  # km
    dist = 2 * radius_of_earth * math.asin(math.sqrt(form0 + form1))
    return dist


def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=ZERO_TENSOR):
    """
    RMSE (Root Mean Squared Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is RMSE-9)
    :return: RMSE-threshold
    """
    threshold = threshold * threshold
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    return torch.sqrt(torch.mean(torch.pow((y_true_filter - y_pred_filter), 2)))


def MAE(y_pred, y_true, threshold=ZERO_TENSOR):
    """
    MAE (Mean Absolute Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is MAE-3)
    :return: MAE-threshold
    """
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    return torch.mean(torch.abs(y_true_filter - y_pred_filter))


def MAPE(y_pred, y_true, threshold=ZERO_TENSOR):
    """
    MAPE (Mean Absolute Percentage Error)
    :param y_pred: prediction tensor
    :param y_true: target tensor
    :param threshold: single-value tensor - only values not below the threshold are considered (if threshold=3, result is MAPE-3)
    :return: MAPE-threshold
    """
    y_true_mask = y_true > threshold
    y_pred_filter = y_pred[y_true_mask]
    y_true_filter = y_true[y_true_mask]
    return torch.mean(torch.abs((y_true_filter - y_pred_filter)/(y_true_filter + 1)))


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(path)[0]


def trainLog2LossCurve(logfn='train.log'):
    if not os.path.isfile(logfn):
        print('{} is not a valid file.'.format(logfn))
        exit(-1)

    x_epoch = []
    y_loss_train = []
    train_time_list = []

    print('Analyzing log file: {}'.format(logfn))
    f = open(logfn, 'r')
    lines = f.readlines()
    for line in lines:
        if not line.startswith('Training Round'):
            continue
        items = line.strip().split(sep=' ')

        epoch = int(items[2][:-1])
        x_epoch.append(epoch)

        loss = float(items[5][:-1])
        y_loss_train.append(loss)

        train_time = float(items[10][1:])
        train_time_list.append(train_time)

    # Count average TTpS
    avgTTpS = sum(train_time_list) / len(train_time_list)
    print('Average TTpS: %.4f sec' % avgTTpS)

    # Plot training loss curve
    print('Plotting loss curve.')
    plt.plot(x_epoch, y_loss_train, c='purple', label='Train Loss', alpha=0.8)
    plt.title('Epoch - Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    # plt.show()
    figpath = '{}.png'.format(path2FileNameWithoutExt(logfn))
    plt.savefig(figpath)
    print('Loss curve saved to {}'.format(figpath))

    print('All analysis tasks finished.')


# by RoshanRane in https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
def plot_grad_flow(named_parameters):
    """
    Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing/exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as "plot_grad_flow(model.named_parameters())" to
        visualize the gradient flow.
    """
    avg_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.requires_grad and (p.grad is not None) and ('bias' not in n):
            layers.append(n)
            avg_grads.append(p.grad.cpu().abs().mean())
            max_grads.append(p.grad.cpu().abs().max())

    plt.figure(figsize=(7, 20))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color='c')
    plt.bar(np.arange(len(avg_grads)), avg_grads, alpha=0.1, lw=1, color='b')
    plt.hlines(0, 0, len(avg_grads)+1, lw=2, color='k')
    plt.xticks(range(0, len(avg_grads), 1), layers, rotation='vertical')
    plt.xlim(left=0, right=len(avg_grads))
    plt.ylim(bottom=-0.001, top=0.5)   # zoom in on the lower gradient regions
    plt.xlabel('Layers')
    plt.ylabel('Average Gradient')
    plt.title('Gradient flow')
    plt.grid(True)
    plt.legend([Line2D([0], [0], color='c', lw=4),
                Line2D([0], [0], color='b', lw=4),
                Line2D([0], [0], color='k', lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(5)
    plt.ioff()


# Test
if __name__ == '__main__':
    # print(haversine((40.4944, -74.2655), (40.9196, -73.6957)))  # 67.39581283189828
    # trainLog2LossCurve(logfn='../res/Gallat_retrain/20210522_14_44_12.log')
    # trainLog2LossCurve(logfn='../res/GallatExt_pretrain/low_dimension/best/20210518_15_20_40.log')
    trainLog2LossCurve(logfn='../res/20210530_05_30_19.log')
