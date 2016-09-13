import numpy as np
import pandas as pd


def preprocess(df, t_range):
    stations = sorted(set(df['STATION_NAME']))
    df['DATE_fmt'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df.set_index(['DATE_fmt'])
    data_per_station = dict()
    for t in ['TMAX', 'TMIN', 'TAVG']:
        df[t] = df[t].apply(lambda x: np.nan if x < t_range[0] else x)
        df[t] = df[t].apply(lambda x: np.nan if x > t_range[1] else x)
    for st in stations:
        data_per_station[st] = df[df['STATION_NAME'] == st]
    return df, stations, data_per_station
