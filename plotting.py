import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt

import utils


def plot_stations_all_time(df, stations, t_range=None):
    """
    Plot all min and max temp data for all stations.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    if t_range is None:
        t_range = [-20, 120]
    f, axes = plt.subplots(len(stations), 1,
                           figsize=(12, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)
        if ii == 0:
            legend = True
        else:
            legend = False
        time = matplotlib.dates.date2num(data.index.date)
        for p, c in zip(['TMIN', 'TMAX'], ['blue', 'red']):
            ax.plot_date(time, data[p], c=c, fmt='-', alpha=.5)
            mean = data[p].mean()
            rolling = data[p].rolling(window=30, min_periods=10,
                                      center=True).median()
            y = mean*np.ones_like(time)
            ax.plot_date(time, y, c=c, zorder=10, fmt='-')
            ax.plot_date(time, rolling, c=c, zorder=10, fmt='-')
        ax.set_ylim(t_range)
        ax.set_title(name)
        ax.set_ylabel('Temp.')
    ax.set_xlabel('Year')
    return f
        
def plot_annual_temperature(df, stations, t_range=None):
    """
    Plot mean annual temperature variations for all stations.
    
    Parameters
    ----------
    
    Returns
    -------
    
    """
    if t_range is None:
        t_range = [0, 100]
    time = np.linspace(1, 365, num=365)
    f, axes = plt.subplots(len(stations), 1,
                           sharex=True,
                           figsize=(6, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)
        mean = np.zeros(365)
        delta = np.zeros(365)
        for day in range(365):
            temps = data[['TMIN', 'TMAX']].loc[data['day'] == day+1]
            mean[day] = np.nanmean(temps.values)
            delta[day] = np.nanmean((temps['TMAX']-temps['TMIN']).values)
        ax.fill_between(time, mean+delta/2., mean-delta/2., facecolor='red',
                        alpha=.5)
        ax.plot(time, mean, c='black')
        ax.set_ylim(t_range)
        ax.set_xlim([0, 366])
        ax.set_title(name)
        #ax.set_xlabel('Day of year')
        ax.set_xticks([79, 172, 265, 344])
        ax.set_xticklabels(['March 20', 'June 21', 'Sept. 22', 'Dec. 21'])
    return f