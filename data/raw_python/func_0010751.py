def fit_cosine_function(wind):
    """fits a cosine function to observed hourly windspeed data

    Args:
        wind: observed hourly windspeed data
        
    Returns:
        parameters needed to generate diurnal features of windspeed using a cosine function
    """
    wind_daily = wind.groupby(wind.index.date).mean()
    wind_daily_hourly = pd.Series(index=wind.index, data=wind_daily.loc[wind.index.date].values)  # daily values evenly distributed over the hours

    df = pd.DataFrame(data=dict(daily=wind_daily_hourly, hourly=wind)).dropna(how='any')
    x = np.array([df.daily, df.index.hour])
    popt, pcov = scipy.optimize.curve_fit(_cosine_function, x, df.hourly)

    return popt