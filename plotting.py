import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import utils


def plot_stations_all_time(df, stations, t_range=None):
    """
    Plot all min and max temp data for all stations.
    
    Parameters
    ----------
    df : dataframe
        Data for all stations.
    stations : list
        List of stations to include in plot.
    t_range : list, optional
        Values to clip temperatures to.
    
    Returns
    -------
    f : figure
        Matplotlib figure.
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
    df : dataframe
        Data for all stations.
    stations : list
        List of stations to include in plot.
    t_range : list, optional
        Values to clip temperatures to.
    
    Returns
    -------
    f : figure
        Matplotlib figure.
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

def plot_annual_daily_comparison(df, stations):
    """
    Plot annual vs. daily temperature variations for all stations.
    
    Parameters
    ----------
    df : dataframe
        Data for all stations.
    stations : list
        List of stations to include in plot.
    
    Returns
    -------
    f : figure
        Matplotlib figure.
    """
    colors = matplotlib.cm.get_cmap('plasma')
    colors = [colors(v) for v in np.linspace(0, 1, len(stations))]
    f = plt.figure(figsize=(6, 6))
    ax = f.gca()
    x_max = 0.
    y_max = 0.
    for ii, st in enumerate(stations):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)
        daily_delta = data['TMAX'] - data['TMIN']
        years = sorted(set(data.index.year))
        days = data.index.dayofyear
        if days[0] > 1:
            years = years[1:]
        if days[-1] < 365:
            years = years[:-1]
        annual_delta = np.zeros(len(years))
        for jj, year in enumerate(years):
            min_t = data['TMIN'].loc[data.index.year == year].min()
            min_t = min(min_t,
                        data['TMAX'].loc[data.index.year == year].min())
            max_t = data['TMIN'].loc[data.index.year == year].max()
            max_t = max(max_t,
                        data['TMAX'].loc[data.index.year == year].max())
            annual_delta[jj] = max_t - min_t
        e = Ellipse(xy=[annual_delta.mean(), np.nanmean(daily_delta)],
                    height=2*np.nanstd(daily_delta),
                    width=2*annual_delta.std())
        x_max = max(x_max, annual_delta.mean() + 2*annual_delta.std())
        y_max = max(y_max, np.nanmean(daily_delta) + 2*np.nanstd(daily_delta))
        ax.add_artist(e)
        e.set_facecolor(colors[ii])
        e.set_alpha(.5)
        e.set_clip_box(ax.bbox)
        ax.plot(0, 0, c=colors[ii], label=name)
    ax.set_xlim([50, x_max])
    ax.set_ylim([0, y_max])
    ax.legend(loc='lower right', prop={'size': 12})
    ax.set_xlabel('Annual temp. swing')
    ax.set_ylabel('Daily temp. swing')
    return f
