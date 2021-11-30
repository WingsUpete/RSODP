"""
A baseline model called AR - Auto-regressive: Predict with the considerations of the past p records
"""
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AutoReg


def AR(data_train, pred_start_at, pred_end_at, lag):
    autoregressive = AutoReg(data_train, lags=lag).fit()
    print(autoregressive.summary())
    print(autoregressive.params)

    pred = autoregressive.predict(start=pred_start_at, end=pred_end_at)
    return pred


if __name__ == '__main__':
    dt = np.array([i for i in range(10)]).astype('float')
    dt_split_ind = 8
    dTrain = dt[:dt_split_ind]
    dTrain[3] = 3.2
    dTest = dt[dt_split_ind:]

    res = AR(data_train=dTrain,  pred_start_at=0, pred_end_at=9, lag=2)
    print('train:\t%s', list(dTrain))
    print('res\t%s', list(res))
    # plt.plot(res, color='purple')
    # plt.plot(dTest, color='red')
    # plt.show()
