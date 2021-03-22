"""
Package the data into a specified folder, which will contain:
1. Passenger Request Data separated in hours (1.csv, 2.csv, 3.csv, ...)
2. Grid Info grid_info.json (minLat, maxLat, minLng, maxLng, girdH, gridW, latLen, latGridNum, gridLat, ..., gridNum)
3. Adjacency Matrix (in form of tensor) for DDW Graph with grids as nodes adjacency_matrix.pt
"""
import argparse
import os
import math
import json
import torch
import pandas as pd


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


def path2FileNameWithoutExt(path):
    """
    get file name without extension from path
    :param path: file path
    :return: file name without extension
    """
    return os.path.splitext(path)[0]


def splitData(fPath, folder):
    df = pd.read_csv(fPath)
    df['request time'] = pd.to_datetime(df['request time'])
    minT, maxT = df['request time'].min(), df['request time'].max()
    totalH = round((maxT - minT) / pd.Timedelta(hours=1))
    lowT, upT = minT, minT + pd.Timedelta(hours=1)
    print('Dataframe prepared. Total hours = {}.'.format(totalH))
    for i in range(totalH):
        curH = i + 1
        print('\r-> Splitting hour-wise data No.{}/{}.'.format(curH, totalH), end='\r')
        mask = ((df['request time'] >= lowT) & (df['request time'] < upT)).values
        df_split = df.iloc[mask]
        df_split.to_csv(os.path.join(folder, '{}.csv'.format(curH)), index=False)
        lowT += pd.Timedelta(hours=1)
        upT += pd.Timedelta(hours=1)
    print('Data splitting complete.')


def getGridInfo(minLat, maxLat, minLng, maxLng, refGridW=2.5, refGridH=2.5):
    """
    :param minLat: lower boundary of region's latitude
    :param maxLat: upper boundary of region's latitude
    :param minLng: lower boundary of region's longitude
    :param maxLng: upper boundary of region's longitude
    :param refGridW: reference width of a grid in km, will auto-adjust later
    :param refGridH: reference height of a grid in km, will auto-adjust later
    :return: grid_info dictionary
    """
    grid_info = {
        'minLat': minLat,
        'maxLat': maxLat,
        'minLng': minLng,
        'maxLng': maxLng
    }
    grid_info['latLen'] = haversine((grid_info['minLat'], grid_info['maxLng']),
                                    (grid_info['maxLat'], grid_info['maxLng']))
    grid_info['latGridNum'] = round(grid_info['latLen'] / refGridH)
    grid_info['gridH'] = grid_info['latLen'] / grid_info['latGridNum']
    grid_info['gridLat'] = (grid_info['maxLat'] - grid_info['minLat']) / grid_info['latGridNum']

    grid_info['lngLen'] = haversine((grid_info['maxLat'], grid_info['minLng']),
                                    (grid_info['maxLat'], grid_info['maxLng']))
    grid_info['lngGridNum'] = round(grid_info['lngLen'] / refGridW)
    grid_info['gridW'] = grid_info['lngLen'] / grid_info['lngGridNum']
    grid_info['gridLng'] = (grid_info['maxLng'] - grid_info['minLng']) / grid_info['lngGridNum']

    grid_info['gridNum'] = grid_info['latGridNum'] * grid_info['lngGridNum']
    print('Grid info retrieved.')
    return grid_info


def saveGridInfo(grid_info, fPath):
    with open(fPath, 'w') as f:
        json.dump(grid_info, f)
    print('grid_info saved to {}'.format(fPath))


def makeGridNodes(grid_info):
    leftLng = grid_info['minLng'] + grid_info['gridLng'] / 2
    midLat = grid_info['maxLat'] - grid_info['gridLat'] / 2
    grid_nodes = []
    for i in range(grid_info['latGridNum']):
        midLng = leftLng
        for j in range(grid_info['lngGridNum']):
            grid_nodes.append((midLat, midLng))
            midLng += grid_info['gridLng']
        midLat -= grid_info['gridLat']
    print('Grid nodes generated.')
    return grid_nodes


def getAdjacencyMatrix(grid_nodes):
    adjacency_matrix = torch.zeros((len(grid_nodes), len(grid_nodes)))
    for i in range(len(grid_nodes)):
        for j in range(len(grid_nodes)):
            adjacency_matrix[i][j] = haversine(grid_nodes[i], grid_nodes[j])
    print('Adjacency Matrix generated.')
    return adjacency_matrix


def saveAdjacencyMatrix(adjacency_matrix, fPath):
    torch.save(adjacency_matrix, fPath)
    print('Adjacency matrix saved to {}'.format(fPath))


if __name__ == '__main__':
    """
    Usage Example:
        python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='ny2016_0101to0331.csv',
                        help='The input request data file to be handled, default={}'.format('ny2016_0101to0331.csv'))
    parser.add_argument('--minLat', type=float, default=40.4944,
                        help='The minimum latitude for the grids, default={}'.format(40.4944))
    parser.add_argument('--maxLat', type=float, default=40.9196,
                        help='The maximum latitude for the grids, default={}'.format(40.9196))
    parser.add_argument('--minLng', type=float, default=-74.2655,
                        help='The minimum longitude for the grids, default={}'.format(-74.2655))
    parser.add_argument('--maxLng', type=float, default=-73.6957,
                        help='The minimum latitude for the grids, default={}'.format(-73.6957))
    parser.add_argument('--refGridH', type=float, default=2.5,
                        help='The reference height for the grids, default={}, final grid height might be different'.format(
                            2.5))
    parser.add_argument('--refGridW', type=float, default=2.5,
                        help='The reference height for the grids, default={}, final grid width might be different'.format(
                            2.5))
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.data):
        print('Data file path {} is invalid.'.format(FLAGS.data))
        exit(-1)

    folderName = path2FileNameWithoutExt(FLAGS.data)
    if not os.path.isdir(folderName):
        os.mkdir(folderName)

    # 1
    splitData(FLAGS.data, folderName)

    # 2
    gridInfo = getGridInfo(FLAGS.minLat, FLAGS.maxLat, FLAGS.minLng, FLAGS.maxLng, FLAGS.refGridH, FLAGS.refGridW)
    saveGridInfo(gridInfo, os.path.join(folderName, 'grid_info.json'))
    # print(json.load(open(os.path.join(folderName, 'grid_info.json'))))    # Load Example

    # 3
    gridNodes = makeGridNodes(gridInfo)
    adjacencyMatrix = getAdjacencyMatrix(gridNodes)
    saveAdjacencyMatrix(adjacencyMatrix, os.path.join(folderName, 'adjacency_matrix.pt'))
    # print(torch.load(os.path.join(folderName, 'adjacency_matrix.pt')))  # Load Example
