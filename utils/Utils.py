"""
Utility functions
"""
import math
import torch

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


def filter_with_threshold(x: torch.Tensor, threshold: torch.Tensor):
    """
    Filter out values below the threshold (they will become the threshold)
    :param x: a tensor
    :param threshold: single-value tensor containing the threshold
    :return: filtered tensor
    """
    return torch.max(x, threshold)


def RMSE(y_pred: torch.Tensor, y_true: torch.Tensor, threshold=ZERO_TENSOR):
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


def MAE(y_pred, y_true, threshold=ZERO_TENSOR):
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


def MAPE(y_pred, y_true, threshold=ZERO_TENSOR):
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


# Test
if __name__ == '__main__':
    # print(haversine((40.4944, -74.2655), (40.9196, -73.6957)))  # 67.39581283189828
    print(haversine(
        (40.9196, -74.2655),
        (40.9196, -73.6957))
    )
