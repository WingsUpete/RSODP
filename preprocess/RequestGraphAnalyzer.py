import os
import argparse
import json


def analyze(data_dir='../data/ny2016_0101to0331/'):
    req_info = json.load(open(os.path.join(data_dir, 'req_info.json')))
    total_H = req_info['totalH']


if __name__ == '__main__':
    """
        Usage Example:
            python DataAnalysis.py -d ny2016_0101to0331.csv
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data_dir', type=str, default='../data/ny2016_0101to0331/',
                        help='The input data directory to be analyzed, default={}'.format('../data/ny2016_0101to0331/'))
    FLAGS, unparsed = parser.parse_known_args()

    # DO STH
    data_dir = FLAGS.data_dir
    if os.path.isdir(data_dir):
        analyze(data_dir=data_dir)
