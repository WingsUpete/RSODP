"""
Package the data into a specified folder, which will contain:
1. grid_info.json (minLat, maxLat, minLng, maxLng, girdH, gridW, latLen, latGridNum, gridLat, ..., gridNum)
2. GeoGraph.dgl which stores the geographical graph GeoGraph
3. Passenger Request Data separated in hours ((1/request.csv, 1/GDVQ.npy, 1/FBGraphs.dgl), ...)
"""
import argparse
import os
import sys
import math
import json
import numpy as np
import pandas as pd
import torch
import multiprocessing

stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import dgl
sys.stderr.close()
sys.stderr = stderr

EPSILON = 1e-12
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


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
    """
    Generate a grid node coordinate list where each grid represents a node
    :param grid_info: Grid Map Information
    :return: A Grid Coordinate List st. in each position (Grid ID) a (latitude, longitude) pair is stored
    """
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


def pushGraphEdge(gSrc: list, gDst: list, wList, src, dst, weight):
    gSrc.append(src)
    gDst.append(dst)
    if wList is not None and weight is not None:
        wList.append(weight)
        return gSrc, gDst, wList
    else:
        return gSrc, gDst


def getGeoGraph(grid_nodes, L):
    """
    Get RTc data object with the given information
    :param grid_nodes: Grid Coordinate List storing the coordinates of each grid/node
    :param L: Threshold distance for geographical neighborhood decision making
    :return: Geographical Neighborhood DGL Graph
    """
    adjacency_matrix = np.zeros((len(grid_nodes), len(grid_nodes)))
    RSrcList, RDstList = [], []
    TcMat = np.zeros((len(grid_nodes), len(grid_nodes)))
    for i in range(len(grid_nodes)):
        for j in range(len(grid_nodes)):
            adjacency_matrix[i][j] = haversine(grid_nodes[i], grid_nodes[j])
            if i != j and adjacency_matrix[i][j] <= L:
                # if i->j is small enough, j is i's geographical neighbor, j should propagate its features to i, so j->i
                RSrcList, RDstList = pushGraphEdge(RSrcList, RDstList, None, j, i, None)
                TcMat[j][i] = 1 / (adjacency_matrix[i][j] + EPSILON)
    TcSum = np.sum(TcMat, axis=0)
    for ni in range(len(grid_nodes)):
        if TcSum[ni] == 0:
            continue
        for nj in range(len(grid_nodes)):
            TcMat[nj][ni] /= TcSum[ni]

    # Transform node data TcMat to edge data TcEdges
    TcEdges = []
    for i in range(len(RSrcList)):
        TcEdges.append([TcMat[RSrcList[i]][RDstList[i]]])

    GeoGraph = dgl.graph((RSrcList, RDstList), num_nodes=len(grid_nodes))
    GeoGraph.edata['pre_w'] = torch.Tensor(TcEdges)

    print('Geographical info generated.')
    return GeoGraph


def saveGeoGraph(geoG, fPath):
    dgl.save_graphs(fPath, geoG)
    print('Geographical info saved to {}'.format(fPath))


def handleRequestData(i, totalH, folder, lowT, df_split, export_requests, grid_nodes, grid_info):
    curH = i + 1
    print('-> Splitting hour-wise data No.{}/{}.'.format(curH, totalH))

    # Folder for this split of data
    curDir = os.path.join(folder, str(curH))
    if not os.path.isdir(curDir):
        os.mkdir(curDir)

    dayOfWeek = lowT.weekday()  # Mon: 0, ..., Sun: 6

    GDVQ = {}

    # Save request.csv
    if export_requests:
        df_split.to_csv(os.path.join(curDir, 'request.csv'), index=False)

    # Get request matrix G
    request_matrix = np.zeros((len(grid_nodes), len(grid_nodes)))
    nRequests = 0
    for split_i in range(len(df_split)):
        curData = df_split.iloc[split_i]
        srcRow, srcCol, srcID = inWhichGrid((curData['src lat'], curData['src lng']), grid_info)
        dstRow, dstCol, dstID = inWhichGrid((curData['dst lat'], curData['dst lng']), grid_info)
        request_matrix[srcID][dstID] += curData['volume']
        nRequests += curData['volume']
    GDVQ['G'] = request_matrix.astype(np.float32)

    # Get Feature Matrix V
    feature_vectors = []
    query_feature_vectors = []
    inDs = np.sum(request_matrix, axis=0)  # Col-wise: Total number of nodes pointing to current node = In Degree
    outDs = np.sum(request_matrix, axis=1)  # Row-wise: Total number of nodes current node points to = Out Degree
    GDVQ['D'] = outDs.astype(np.float32)

    for vi in range(len(grid_nodes)):
        viRow, viCol = ID2Coord(vi, grid_info)
        query_feature_vector = [
            viRow / grid_info['latGridNum'],
            viCol / grid_info['lngGridNum'],
            vi / grid_info['gridNum'],
            torch.sigmoid(torch.Tensor([curH])).item(),
            dayOfWeek / 7
        ]
        query_feature_vectors.append(query_feature_vector)
        feature_vector = [
            query_feature_vector[0],
            query_feature_vector[1],
            outDs[vi] / nRequests,
            inDs[vi] / nRequests,
            query_feature_vector[2],
            query_feature_vector[3],
            query_feature_vector[4]
        ]
        feature_vectors.append(feature_vector)
    feature_matrix = np.array(feature_vectors)
    query_feature_matrix = np.array(query_feature_vectors)
    GDVQ['V'] = feature_matrix.astype(np.float32)
    GDVQ['Q'] = query_feature_matrix.astype(np.float32)

    # Save GDVQ as GDVQ.npy
    np.save(os.path.join(curDir, 'GDVQ.npy'), GDVQ)

    # Get Psi (Forward Neighborhood) and Phi (Backward Neighborhood)
    PaSrcList, PaDstList = [], []  # Psi & a
    PbSrcList, PbDstList = [], []  # Phi & b
    PaMat = np.zeros((len(grid_nodes), len(grid_nodes)))
    PbMat = np.zeros((len(grid_nodes), len(grid_nodes)))
    for rmi in range(len(grid_nodes)):
        for rmj in range(len(grid_nodes)):
            if request_matrix[rmi][rmj] > 0:  # In this case, rmi == rmj is valid
                # rmi -> rmj, rmj is rmi's forward neighbor, rmi is rmj's backward neighbor
                # forward neighborhood: features of rmj should propagate to rmi, thus rmj->rmi
                # backward neighborhood: features of rmi should propagate to rmj, thus rmi->rmj
                PaSrcList, PaDstList = pushGraphEdge(PaSrcList, PaDstList, None, rmj, rmi, None)
                PaMat[rmj][rmi] = request_matrix[rmi][rmj] + EPSILON
                PbSrcList, PbDstList = pushGraphEdge(PbSrcList, PbDstList, None, rmi, rmj, None)
                PbMat[rmi][rmj] = request_matrix[rmi][rmj] + EPSILON
    PaSum = np.sum(PaMat, axis=0)
    for ni in range(len(grid_nodes)):
        if PaSum[ni] == 0:
            continue
        for nj in range(len(grid_nodes)):
            PaMat[nj][ni] /= PaSum[ni]
    PbSum = np.sum(PbMat, axis=0)
    for ni in range(len(grid_nodes)):
        if PbSum[ni] == 0:
            continue
        for nj in range(len(grid_nodes)):
            PbMat[nj][ni] /= PbSum[ni]

    # Transform node data Mat to edge data Edges
    PaEdges = []
    for paei in range(len(PaSrcList)):
        PaEdges.append([PaMat[PaSrcList[paei]][PaDstList[paei]]])
    PbEdges = []
    for pbei in range(len(PbSrcList)):
        PbEdges.append([PbMat[PbSrcList[pbei]][PbDstList[pbei]]])

    FNGraph = dgl.graph((PaSrcList, PaDstList), num_nodes=len(grid_nodes))
    FNGraph.edata['pre_w'] = torch.Tensor(PaEdges)
    BNGraph = dgl.graph((PbSrcList, PbDstList), num_nodes=len(grid_nodes))
    BNGraph.edata['pre_w'] = torch.Tensor(PbEdges)

    # Save two neighborhood graphs
    dgl.save_graphs(os.path.join(curDir, 'FBGraphs.dgl'), [FNGraph, BNGraph])


def splitData(fPath, folder, grid_nodes, grid_info, export_requests=1, num_workers=10):
    """
    Split data in hours (request.csv, GDVQ.npy) of each DDW Snapshot Graph
    :param fPath: The path of request data file
    :param folder: The path of the working directory/folder
    :param grid_nodes: Grid Coordinate List storing the coordinates of each grid/node
    :param grid_info: Grid Map Information
    :param export_requests: whether the split requests should be exported if space is enough
    :return: nothing
    """
    df = pd.read_csv(fPath)
    df['request time'] = pd.to_datetime(df['request time'])
    minT, maxT = df['request time'].min(), df['request time'].max()
    totalH = round((maxT - minT) / pd.Timedelta(hours=1))
    lowT, upT = minT, minT + pd.Timedelta(hours=1)
    print('Dataframe prepared. Total hours = {}.'.format(totalH))

    req_info = {
        'name': path2FileNameWithoutExt(fPath),
        'minT': minT.strftime(DATE_FORMAT),
        'maxT': maxT.strftime(DATE_FORMAT),
        'totalH': totalH
    }
    req_info_path = os.path.join(folder, 'req_info.json')
    with open(req_info_path, 'w') as f:
        json.dump(req_info, f)
    print('requests info saved to {}'.format(req_info_path))

    pool = multiprocessing.Pool(processes=num_workers)
    for i in range(totalH):
        # Filter data
        mask = ((df['request time'] >= lowT) & (df['request time'] < upT)).values
        df_split = df.iloc[mask]

        # Handle data
        pool.apply_async(handleRequestData, args=(i, totalH, folder, lowT, df_split, export_requests, grid_nodes, grid_info))
        # handleRequestData(i, totalH, folder, lowT, df_split, export_requests, grid_nodes, grid_info)    # DEBUG

        lowT += pd.Timedelta(hours=1)
        upT += pd.Timedelta(hours=1)

    pool.close()
    pool.join()

    print('Data splitting complete.')


def inWhichGrid(coord, grid_info):
    """
    Specify which grid it is in for a given coordinate
    :param coord: (latitude, longitude)
    :param grid_info: grid_info dictionary
    :return: row, column, grid ID
    """
    lat, lng = coord
    row = math.floor((grid_info['maxLat'] - lat) / grid_info['gridLat'])
    col = math.floor((lng - grid_info['minLng']) / grid_info['gridLng'])
    gridID = row * grid_info['lngGridNum'] + col
    return row, col, gridID


def ID2Coord(gridID, grid_info):
    """
    Given a grid ID, decide the coordinate of that grid
    :param gridID: Given Grid ID
    :param grid_info: Grid map information
    :return: Grid Map Coordinates of that grid in format of (row, col)
    """
    row = math.floor(gridID / grid_info['lngGridNum'])
    col = gridID - row * grid_info['lngGridNum']
    return row, col


if __name__ == '__main__':
    """
    Usage Example:
        python DataPackager.py -d ny2016_0101to0331.csv --minLat 40.4944 --maxLat 40.9196 --minLng -74.2655 --maxLng -73.6957 --refGridH 2.5 --refGridW 2.5 -er 1 -od ../data/ -c 10
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
    parser.add_argument('-er', '--exportRequests', type=int, default=1,
                        help='Whether the split requests should be exported, default={}'.format(1))
    parser.add_argument('-od', '--outDir', type=str, default='./',
                        help='Where the data should be exported, default={}'.format('""'))
    parser.add_argument('-c', '--cores', type=int, default=10,
                        help='How many cores should we use for paralleling , default={}'.format(10))
    FLAGS, unparsed = parser.parse_known_args()

    if not os.path.isfile(FLAGS.data):
        print('Data file path {} is invalid.'.format(FLAGS.data))
        exit(-1)

    if not os.path.isdir(FLAGS.outDir):
        print('Output path {} is invalid.'.format(FLAGS.outDir))
        exit(-2)

    folderName = path2FileNameWithoutExt(FLAGS.data)
    folderName = os.path.join(FLAGS.outDir, folderName)
    if not os.path.isdir(folderName):
        os.mkdir(folderName)

    # 1
    gridInfo = getGridInfo(FLAGS.minLat, FLAGS.maxLat, FLAGS.minLng, FLAGS.maxLng, FLAGS.refGridH, FLAGS.refGridW)
    saveGridInfo(gridInfo, os.path.join(folderName, 'grid_info.json'))
    # print(json.load(open(os.path.join(folderName, 'grid_info.json'))))    # Load Example

    # 2
    gridNodes = makeGridNodes(gridInfo)
    geoGraph = getGeoGraph(gridNodes, max(gridInfo['gridH'], gridInfo['gridW']) * 1.05)
    saveGeoGraph(geoGraph, os.path.join(folderName, 'GeoGraph.dgl'))
    # print(dgl.load_graphs(os.path.join(folderName, 'GeoGraph.dgl')))  # Load Example

    # 3
    splitData(FLAGS.data, folderName, gridNodes, gridInfo, FLAGS.exportRequests == 1, num_workers=FLAGS.cores)
    # print(np.load('GVQ.npy', allow_pickle=True).item())  # Load Example
    # (fg, bg), _ = dgl.load_graphs('FBGraphs.dgl')  # Load Example
