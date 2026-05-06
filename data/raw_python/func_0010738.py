def bristow_campbell(tmin, tmax, pot_rad_daily, A, C):
    """calculates potential shortwave radiation based on minimum and maximum temperature

    This routine calculates global radiation as described in:
    Bristow, Keith L., and Gaylon S. Campbell: On the relationship between
    incoming solar radiation and daily maximum and minimum temperature.
    Agricultural and forest meteorology 31.2 (1984): 159-166.

    Args:
        daily_data: time series (daily data) including at least minimum and maximum temeprature
        pot_rad_daily: mean potential daily radiation
        A: parameter A of the Bristow-Campbell model
        C: parameter C of the Bristow-Campbell model
    Returns:
        series of potential shortwave radiation
    """

    assert tmin.index.equals(tmax.index)

    temp = pd.DataFrame(data=dict(tmin=tmin, tmax=tmax))
    temp = temp.reindex(pd.DatetimeIndex(start=temp.index[0], end=temp.index[-1], freq='D'))
    temp['tmin_nextday'] = temp.tmin
    temp.tmin_nextday.iloc[:-1] = temp.tmin.iloc[1:].values

    temp = temp.loc[tmin.index]
    pot_rad_daily = pot_rad_daily.loc[tmin.index]

    dT = temp.tmax - (temp.tmin + temp.tmin_nextday) / 2

    dT_m_avg = dT.groupby(dT.index.month).mean()
    B = 0.036 * np.exp(-0.154 * dT_m_avg[temp.index.month])
    B.index = temp.index

    if isinstance(A, pd.Series):
        months = temp.index.month
        A = A.loc[months].values
        C = C.loc[months].values

    transmissivity = A * (1 - np.exp(-B * dT**C))
    R0 = transmissivity * pot_rad_daily

    return R0