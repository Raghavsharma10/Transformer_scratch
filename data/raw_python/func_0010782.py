def precip_master_station(precip_daily,
                          master_precip_hourly,
                          zerodiv):
    """Disaggregate precipitation based on the patterns of a master station

    Parameters
    -----------
    precip_daily : pd.Series
        daily data
    master_precip_hourly :  pd.Series
        observed hourly data of the master station
    zerodiv : str
        method to deal with zero division by key "uniform" --> uniform
        distribution
    """

    precip_hourly = pd.Series(index=melodist.util.hourly_index(precip_daily.index))

    # set some parameters for cosine function
    for index_d, precip in precip_daily.iteritems():

        # get hourly data of the day
        index = index_d.date().isoformat()
        precip_h = master_precip_hourly[index]

        # calc rel values and multiply by daily sums
        # check for zero division
        if precip_h.sum() != 0 and precip_h.sum() != np.isnan(precip_h.sum()):
            precip_h_rel = (precip_h / precip_h.sum()) * precip

        else:
            # uniform option will preserve daily data by uniform distr
            if zerodiv == 'uniform':
                precip_h_rel = (1/24) * precip

            else:
                precip_h_rel = 0

        # write the disaggregated day to data
        precip_hourly[index] = precip_h_rel

    return precip_hourly