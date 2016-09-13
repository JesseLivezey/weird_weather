import numpy as np
import pandas as pd
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams.update({'figure.autolayout': True})

def preprocess(df, t_range=None):
    if t_range is None:
        t_range = [-50, 150]
    stations = sorted(set(df['STATION_NAME']))
    df['DATE_fmt'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df = df.set_index(['DATE_fmt'])
    df['year'] = df.index.year
    df['day'] = df.index.dayofyear
    for t in ['TMAX', 'TMIN', 'TAVG']:
        df[t] = df[t].apply(lambda x: np.nan if x < t_range[0] else x)
        df[t] = df[t].apply(lambda x: np.nan if x > t_range[1] else x)
    return df, stations

def single_station_data(df, station):
    return df[df['STATION_NAME'] == station]

def plot_station_all_time(df, stations, t_range=None):
    if t_range is None:
        t_range = [-50, 150]
    for st in stations:
        data = single_station_data(df, st)
        ax = data.plot(y=['TMIN', 'TMAX'], zorder=1)
        for t, c in zip(['TMIN', 'TMAX'], ['red', 'yellow']):
            mean = data[t].mean()
            rolling = data[t].rolling(window=30, min_periods=10,
                                      center=True).median()
            time = np.linspace(*ax.get_xlim(), num=rolling.shape[0], endpoint=True)
            y = mean*np.ones_like(time)
            ax.plot(time, y, c=c, zorder=10)
            ax.plot(time, rolling, c=c, zorder=10)
        plt.ylim(t_range)
        plt.title(st)
        plt.xlabel('Date')
        
def plot_annual_temperature(df, stations, t_range=None):
    if t_range is None:
        t_range = [0, 100]
    time = np.linspace(1, 365, num=365)
    f, axes = plt.subplots(len(stations), 1,
                           sharex=True,
                           figsize=(6, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = ' '.join(st.split(' ')[:-2])
        name = name.replace('METROPOLITAN', 'MET.')
        data = single_station_data(df, st)
        mean = np.zeros(365)
        delta = np.zeros(365)
        for day in range(365):
            temps = data[['TMIN', 'TMAX']].loc[data['day'] == day+1]
            mean[day] = np.nanmean(temps.values)
            delta[day] = np.nanmean((temps['TMAX']-temps['TMIN']).values)
        ax.fill_between(time, mean+delta/2., mean-delta/2., facecolor='red')
        ax.plot(time, mean, c='black')
        ax.set_ylim(t_range)
        ax.set_xlim([0, 366])
        ax.set_title(name)
        #ax.set_xlabel('Day of year')
        ax.set_xticks([15, 105, 258, 350])
        ax.set_xticklabels(['Jan.', 'April', 'Sept.', 'Dec.'])