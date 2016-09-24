import numpy as np
import pandas as pd
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt

rcParams.update({'figure.autolayout': True})

def preprocess(df, t_range=None):
    """
    Take a NOAA dataframe and do a little cleanup.
    
    Parameters
    ----------
    df : dataframe
        Source data. Expected to be read from NOAA csv.
    t_range : list
        Min and max values to clip temperatures.
        
    Returns
    -------
    df : dataframe
        Cleaned dataframe.
    stations : list
        List of station names included in dataframe.
    """
        
    if t_range is None:
        t_range = [-20, 120]
    stations = sorted(set(df['STATION_NAME']))
    df['DATE_fmt'] = pd.to_datetime(df['DATE'], format='%Y%m%d')
    df = df.set_index(['DATE_fmt'])
    df['year'] = df.index.year
    df['day'] = df.index.dayofyear
    for t in ['TMAX', 'TMIN', 'TAVG']:
        df[t] = df[t].apply(lambda x: np.nan if x < t_range[0] else x)
        df[t] = df[t].apply(lambda x: np.nan if x > t_range[1] else x)
    notnull = df['TMIN'].notnull() & df['TMAX'].notnull()
    df = df.loc[notnull]
    return df, stations

def single_station_data(df, station):
    """
    Return the data from df corresponding to station.
    """
    return df[df['STATION_NAME'] == station]

def short_name(name):
    """
    Return a shortened station name. Useful for
    plot titles.
    """
    name = ' '.join(name.split(' ')[:-2])
    name = name.replace('METROPOLITAN', 'MET.')
    return name