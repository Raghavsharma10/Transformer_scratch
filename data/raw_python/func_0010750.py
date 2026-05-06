def disaggregate_wind(wind_daily, method='equal', a=None, b=None, t_shift=None):
    """general function for windspeed disaggregation

    Args:
        wind_daily: daily values
        method: keyword specifying the disaggregation method to be used
        a: parameter a for the cosine function
        b: parameter b for the cosine function
        t_shift: parameter t_shift for the cosine function
        
    Returns:
        Disaggregated hourly values of windspeed.
    """
    assert method in ('equal', 'cosine', 'random'), 'Invalid method'

    wind_eq = melodist.distribute_equally(wind_daily)

    if method == 'equal':
        wind_disagg = wind_eq
    elif method == 'cosine':
        assert None not in (a, b, t_shift)
        wind_disagg = _cosine_function(np.array([wind_eq.values, wind_eq.index.hour]), a, b, t_shift)
    elif method == 'random':
        wind_disagg = wind_eq * (-np.log(np.random.rand(len(wind_eq))))**0.3

    return wind_disagg