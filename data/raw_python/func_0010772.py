def get_shift_by_data(temp_hourly, lon, lat, time_zone):
    '''function to get max temp shift (monthly) by hourly data
    
    Parameters
    ----
    hourly_data_obs : observed hourly data 
    lat :             latitude in DezDeg
    lon :             longitude in DezDeg
    time_zone:        timezone
    '''
    daily_index = temp_hourly.resample('D').mean().index
    sun_times = melodist.util.get_sun_times(daily_index, lon, lat, time_zone)

    idxmax = temp_hourly.groupby(temp_hourly.index.date).idxmax()
    idxmax.index = pd.to_datetime(idxmax.index)
    max_temp_hour_obs = idxmax.dropna().apply(lambda d: d.hour)
    max_temp_hour_pot = sun_times.sunnoon
    max_delta = max_temp_hour_obs - max_temp_hour_pot
    mean_monthly_delta = max_delta.groupby(max_delta.index.month).mean()

    return mean_monthly_delta