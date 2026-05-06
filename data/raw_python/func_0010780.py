def daily_from_hourly(df):
    """Aggregates data (hourly to daily values) according to the characteristics
    of each variable (e.g., average for temperature, sum for precipitation)

    Args:
        df: dataframe including time series with one hour time steps

    Returns:
        dataframe (daily)
    """

    df_daily = pd.DataFrame()

    if 'temp' in df:
        df_daily['temp'] = df.temp.resample('D').mean()
        df_daily['tmin'] = df.temp.groupby(df.temp.index.date).min()
        df_daily['tmax'] = df.temp.groupby(df.temp.index.date).max()

    if 'precip' in df:
        df_daily['precip'] = df.precip.resample('D').sum()

    if 'glob' in df:
        df_daily['glob'] = df.glob.resample('D').mean()

    if 'hum' in df:
        df_daily['hum'] = df.hum.resample('D').mean()

    if 'hum' in df:
        df_daily['hum_min'] = df.hum.groupby(df.hum.index.date).min()

    if 'hum' in df:
        df_daily['hum_max'] = df.hum.groupby(df.hum.index.date).max()

    if 'wind' in df:
        df_daily['wind'] = df.wind.resample('D').mean()

    if 'ssd' in df:
        df_daily['ssd'] = df.ssd.resample('D').sum() / 60  # minutes to hours

    df_daily.index.name = None
    return df_daily