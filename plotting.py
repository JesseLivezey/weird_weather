import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from importlib import reload
import utils
reload(utils)


def plot_annual_jacket_crossings(df, stations, temp):
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
    days = np.linspace(1, 365, num=365)
    f, axes = plt.subplots(len(stations), 1,
                           sharex=True,
                           figsize=(6, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)
        cross, years = utils.annual_jacket_crossing(data, temp)
        mean = np.nanmean(cross, axis=0)
        frac_cross = (mean > .5).sum()/float(mean.shape[0])
        ax.fill_between(days, np.zeros_like(mean), mean, facecolor='blue',
                        alpha=.5)
        ax.text(7.5, .65, '{}% of days\nP>.5\n@{} deg.'.format(np.rint(100*frac_cross).astype(int),
                                                                temp),
                bbox={'facecolor':'white', 'alpha':0.5, 'pad':5})
        ax.axhline(.5, c='black')
        ax.set_ylim([0, 1])
        ax.set_xlim([0, 366])
        ax.set_title(name)
        ax.set_yticks(np.linspace(0, 1, 5))
        ax.set_xticks([79, 172, 265, 344])
        ax.set_xticklabels(['March 20', 'June 21', 'Sept. 22', 'Dec. 21'])
        ax.set_ylabel('P(jacket crossing)')
        ax.grid()
    return f

def plot_daily_fluctuations(df, stations):
    """
    Plot daily fluctuations for TMAX and TMIN.
    
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
    f, axes = plt.subplots(len(stations), 1,
                           sharex=True,
                           figsize=(5, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)

        max_data, years = utils.annual_data(data, 'TMAX')
        max_data -= max_data.mean(axis=0, keepdims=True)
        hist, bins = np.histogram(max_data.flatten(), bins=60,
                                  range=[-30, 30], density=True)
        ax.step(bins[:-1], hist, 'r', where='mid', label='Daily max')

        min_data, years = utils.annual_data(data, 'TMIN')
        min_data -= min_data.mean(axis=0, keepdims=True)
        hist, bins = np.histogram(min_data.flatten(), bins=60,
                                  range=[-30, 30], density=True)
        ax.step(bins[:-1], hist, 'b', where='mid', label='Daily min')

        ax.set_title(name)
        ax.set_ylabel('prob. density')
        ax.set_ylim([0, .15])
        ax.set_yticks(np.arange(0, .16, .05))
        ax.grid()
    axes[0].legend(loc='best', ncol=2)
    ax.set_xlabel('Deviation from mean daily temperature')

    return f


def plot_annual_power_spectrum(df, stations):
    """
    Plot annual temperature powerspectrum.
    
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
    f, axes = plt.subplots(len(stations), 1,
                           sharex=True,
                           figsize=(6, 2*len(stations)))
    for ii, (st, ax) in enumerate(zip(stations, axes)):
        name = utils.short_name(st)
        data = utils.single_station_data(df, st)
        freqs, tmin_power = utils.mean_annual_powerspectrum(data, 'TMIN')
        freqs, tmax_power = utils.mean_annual_powerspectrum(data, 'TMAX')
        ax.loglog(freqs, tmin_power, c='blue')
        ax.loglog(freqs, tmin_power, '.', c='blue', alpha=.5)
        ax.loglog(freqs, tmax_power, c='red')
        ax.loglog(freqs, tmax_power, '.', c='red', alpha=.5)
        ax.set_title(name)
        ax.set_ylabel('Temp.')
        ax.axvline(12, c='black', linestyle='--', label='Monthly fluctuations')
        ax.axvline(52, c='black', label='Weekly fluctuations')
        ax.set_ylim([1e1, 1e4])
        ax.set_xlim([1e0, 2e2])
        ax.grid()
    axes[0].plot(0, 0, 'r-', label='Daily max')
    axes[0].plot(0, 0, 'b-', label='Daily min')
    leg = axes[0].legend(loc='best', ncol=2)
    axes[-1].set_xlabel('Cycles/year')

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
    f, ax = plt.subplots(1)
    x_max = 0.
    y_max = 0.
    x_min = np.inf
    y_min = np.inf
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
        ax.plot(annual_delta.mean(), np.nanmean(daily_delta), 'o', c=colors[ii])
        x_max = max(x_max, annual_delta.mean() + 1.5*annual_delta.std())
        y_max = max(y_max, np.nanmean(daily_delta) + 1.5*np.nanstd(daily_delta))
        x_min = min(x_min, annual_delta.mean() - 1.5*annual_delta.std())
        y_min = min(y_min, np.nanmean(daily_delta) - 1.5*np.nanstd(daily_delta))
        ax.add_artist(e)
        e.set_facecolor(colors[ii])
        e.set_alpha(.5)
        e.set_clip_box(ax.bbox)
        ax.plot(0, 0, c=colors[ii], label=name)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    leg = ax.legend(loc='best', prop={'size': 12}, ncol=2)
    leg.get_frame().set_alpha(0.5)
    ax.set_xlabel('Annual temp. swing')
    ax.set_ylabel('Daily temp. swing')
    f.set_size_inches(8., 8.*(y_max-y_min)/(x_max-x_min))
    plt.grid()
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
        ax.grid()
        ax.set_ylabel('Temp.')
    return f

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
