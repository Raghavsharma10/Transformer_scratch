def disaggregate_radiation(data_daily,
                           sun_times=None,
                           pot_rad=None,
                           method='pot_rad',
                           angstr_a=0.25,
                           angstr_b=0.5,
                           bristcamp_a=0.75,
                           bristcamp_c=2.4,
                           mean_course=None):
    """general function for radiation disaggregation

    Args:
        daily_data: daily values
        sun_times: daily dataframe including results of the util.sun_times function
        pot_rad: hourly dataframe including potential radiation
        method: keyword specifying the disaggregation method to be used
        angstr_a: parameter a of the Angstrom model (intercept)
        angstr_b: parameter b of the Angstrom model (slope)
        mean_course: monthly values of the mean hourly radiation course
        
    Returns:
        Disaggregated hourly values of shortwave radiation.
    """
    # check if disaggregation method has a valid value
    if method not in ('pot_rad', 'pot_rad_via_ssd', 'pot_rad_via_bc', 'mean_course'):
        raise ValueError('Invalid option')

    glob_disagg = pd.Series(index=melodist.util.hourly_index(data_daily.index))

    if method == 'mean_course':
        assert mean_course is not None

        pot_rad = pd.Series(index=glob_disagg.index)
        pot_rad[:] = mean_course.unstack().loc[list(zip(pot_rad.index.month, pot_rad.index.hour))].values
    else:
        assert pot_rad is not None

    pot_rad_daily = pot_rad.resample('D').mean()

    if method in ('pot_rad', 'mean_course'):
        globalrad = data_daily.glob
    elif method == 'pot_rad_via_ssd':
        # in this case use the Angstrom model
        globalrad = pd.Series(index=data_daily.index, data=0.)
        dates = sun_times.index[sun_times.daylength > 0]  # account for polar nights
        globalrad[dates] = angstroem(data_daily.ssd[dates], sun_times.daylength[dates],
                                     pot_rad_daily[dates], angstr_a, angstr_b)
    elif method == 'pot_rad_via_bc':
        # using data from Bristow-Campbell model
        globalrad = bristow_campbell(data_daily.tmin, data_daily.tmax, pot_rad_daily, bristcamp_a, bristcamp_c)

    globalrad_equal = globalrad.reindex(pot_rad.index, method='ffill')  # hourly values (replicate daily mean value for each hour)
    pot_rad_daily_equal = pot_rad_daily.reindex(pot_rad.index, method='ffill')
    glob_disagg = pot_rad / pot_rad_daily_equal * globalrad_equal
    glob_disagg[glob_disagg < 1e-2] = 0.

    return glob_disagg