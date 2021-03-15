# Conversion of New York Yellow Taxi Trip Data (2016): https://www.kaggle.com/vishnurapps/newyork-taxi-demand
import argparse
import pandas as pd
import numpy as np


def convertData(inFile, outFile, startDate, endDate):
    df = pd.read_csv(inFile)
    # missingStat(df)
    df_out = df[['tpep_pickup_datetime',
                 'pickup_latitude', 'pickup_longitude',
                 'dropoff_latitude', 'dropoff_longitude',
                 'passenger_count']].rename(columns={
        'tpep_pickup_datetime': 'request time',
        'pickup_latitude': 'src lat',
        'pickup_longitude': 'src lng',
        'dropoff_latitude': 'dst lat',
        'dropoff_longitude': 'dst lng',
        'passenger_count': 'volume'
    })
    # Select a day of data
    df_out['request time'] = pd.to_datetime(df_out['request time'])
    mask = ((df_out['request time'] >= startDate) & (df_out['request time'] < endDate)).values
    df_out = df_out.iloc[mask]
    # Filter abnormal data: In New York, any coordinate as 0 is abnormal
    mask = (df_out['src lat'] * df_out['src lng'] * df_out['dst lat'] * df_out['dst lng'] != 0).values
    df_out = df_out.iloc[mask]
    # Sort by Date
    df_out.sort_values(by=['request time'], inplace=True)
    df_out.to_csv(outFile, index=False)


def missingStat(df):
    nullSheet = df.isnull().sum()
    nNull = np.sum(nullSheet)
    total = np.prod(df.shape)
    print('\nMissing Count:')
    print(nullSheet)
    print('Missing Percentage = %.2f / %.2f = %.2f%%\n' % (float(nNull), float(total), nNull / total * 100))


if __name__ == '__main__':
    """
    Usage Example:
        python ny2016.py -i yellow_tripdata_2016-03.csv -o ny2016.csv -sd 2016-03-06 -ed 2016-03-13
    """
    # Command Line Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, default='yellow_tripdata_2016-03.csv',
                        help='The input data file to be converted, default={}'.format('yellow_tripdata_2016-03.csv'))
    parser.add_argument('-o', '--output', type=str, default='ny2016.csv',
                        help='The output converted data file, default={}'.format('ny2016.csv'))
    parser.add_argument('-sd', '--startDate', type=str, default='2016-03-06',
                        help='The start date to filter (inclusive), default={}'.format('2016-03-06'))
    parser.add_argument('-ed', '--endDate', type=str, default='2016-03-12',
                        help='The end date to filter (exclusive), default={}'.format('2016-03-12'))
    FLAGS, unparsed = parser.parse_known_args()

    convertData(FLAGS.input, FLAGS.output, FLAGS.startDate, FLAGS.endDate)
