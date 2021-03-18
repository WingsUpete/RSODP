"""
Analyze the data on various aspects
"""
import argparse
import pandas as pd
import os


def statistics(file):
    df = pd.read_csv(file)
    df['request time'] = pd.to_datetime(df['request time'])

    minT, maxT = df['request time'].min(), df['request time'].max()
    tSpan = (maxT - minT).days
    print('time span: [{}, {}] => {} days'.format(minT, maxT, tSpan))

    minLat, maxLat = df[['src lat', 'dst lat']].min().min(), df[['src lat', 'dst lat']].max().max()
    minLng, maxLng = df[['src lng', 'dst lng']].min().min(), df[['src lng', 'dst lng']].max().max()
    print('latitude ∈ [{}, {}]'.format(minLat, maxLat))
    print('longitude ∈ [{}, {}]'.format(minLng, maxLng))


if __name__ == '__main__':
    """
        Usage Example:
            python DataAnalysis.py -d ny2016_0101to0331.csv
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', type=str, default='ny2016_0101to0331.csv',
                        help='The input data file to be analyzed, default={}'.format('ny2016_0101to0331.csv'))
    FLAGS, unparsed = parser.parse_known_args()

    # DO STH
    fileName = FLAGS.data
    if os.path.isfile(fileName):
        statistics(fileName)
