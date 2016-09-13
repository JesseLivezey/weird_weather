import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def preprocess(df, t_range=None):
    if t_range is None:
        t_range = [-50, 150]
    stations = sorted(set(df['STATION_NAME']))
    df['DATE_fmt'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    data_per_station = dict()
    for t in ['TMAX', 'TMIN', 'TAVG']:
        df[t] = df[t].apply(lambda x: np.nan if x < t_range[0] else x)
        df[t] = df[t].apply(lambda x: np.nan if x > t_range[1] else x)
    for st in stations:
        data_per_station[st] = df[df['STATION_NAME'] == st]
    return df, stations, data_per_station

def plot_station_all_time(df, station, t_range=None):
    if t_range is None:
        t_range = [-50, 150]
    ax = df.plot('DATE_fmt', ['TMIN', 'TMAX'], zorder=1)
    for t, c in zip(['TMIN', 'TMAX'], ['red', 'yellow']):
        mean = df[t].mean()
        rolling = df[t].rolling(window=30, min_periods=10, center=True).median()
        time = np.linspace(*ax.get_xlim(), num=rolling.shape[0], endpoint=True)
        y = mean*np.ones_like(time)
        ax.plot(time, y, c=c, zorder=10)
        ax.plot(time, rolling, c=c, zorder=10)
    plt.ylim(t_range)
    plt.title(station)
    plt.xlabel('Date')