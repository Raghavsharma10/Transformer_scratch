def disagg_prec(dailyData,
                method='equal',
                cascade_options=None,
                hourly_data_obs=None,
                zerodiv="uniform",
                shift=0):
    """The disaggregation function for precipitation.

    Parameters
    ----------
    dailyData : pd.Series
        daily data
    method : str
        method to disaggregate
    cascade_options : cascade object
        including statistical parameters for the cascade model
    hourly_data_obs : pd.Series
        observed hourly data of master station
    zerodiv : str
        method to deal with zero division by key "uniform" --> uniform
        distribution
    shift : int
        shifts the precipitation data by shift (int) steps (eg +7 for
        7:00 to 6:00)
    """

    if method not in ('equal', 'cascade', 'masterstation'):
        raise ValueError('Invalid option')

    if method == 'equal':
        precip_disagg = melodist.distribute_equally(dailyData.precip,
                                                    divide=True)
    elif method == 'masterstation':
        precip_disagg = precip_master_station(dailyData,
                                              hourly_data_obs,
                                              zerodiv)
    elif method == 'cascade':
        assert cascade_options is not None
        precip_disagg = disagg_prec_cascade(dailyData,
                                            cascade_options,
                                            shift=shift)

    return precip_disagg