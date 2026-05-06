def build_casc(ObsData, hourly=True,level=9,
               months=None,
               avg_stats=True,
               percentile=50):
    '''Builds the cascade statistics of observed data for disaggregation

    Parameters
    -----------
    ObsData : pd.Series
        hourly=True -> hourly obs data
        else -> 5min data (disaggregation level=9 (default), 10, 11)
    
    months : numpy array of ints
        Months for each seasons to be used for statistics (array of
        numpy array, default=1-12, e.g., [np.arange(12) + 1])
    avg_stats : bool
        average statistics for all levels True/False (default=True)
    percentile : int, float
        percentile for splitting the dataset in small and high
        intensities (default=50)

    Returns
    -------
    list_seasonal_casc :
        list holding the results
    '''

    list_seasonal_casc = list()

    if months is None:
        months = [np.arange(12) + 1]

    # Parameter estimation for each season
    for cur_months in months:
        vdn = seasonal_subset(ObsData, cur_months)
        if len(ObsData.precip[np.isnan(ObsData.precip)]) > 0:
            ObsData.precip[np.isnan(ObsData.precip)] = 0

        casc_opt = melodist.cascade.CascadeStatistics()
        casc_opt.percentile = percentile
        list_casc_opt = list()
        
        count = 0
        
        if hourly:
            aggre_level = 5
        else:
            aggre_level =  level
        
        thresholds = np.zeros(aggre_level) #np.array([0., 0., 0., 0., 0.])
        
        for i in range(0, aggre_level):
            # aggregate the data
            casc_opt_i, vdn = aggregate_precipitation(vdn, hourly, \
                                percentile=percentile)
            thresholds[i] = casc_opt_i.threshold
            copy_of_casc_opt_i = copy.copy(casc_opt_i)
            list_casc_opt.append(copy_of_casc_opt_i)
            n_vdn = len(vdn)
            casc_opt_i * n_vdn  # level related weighting
            casc_opt + casc_opt_i  # add to total statistics
            count = count + n_vdn
        casc_opt * (1. / count)  # transfer weighted matrices to probabilities
        casc_opt.threshold = thresholds
        
        # statistics object
        if avg_stats:
            # in this case, the average statistics will be applied for all levels likewise
            stat_obj = casc_opt
        else:
            # for longer time series, separate statistics might be more appropriate
            # level dependent statistics will be assumed
            stat_obj = list_casc_opt

        list_seasonal_casc.append(stat_obj)

    return list_seasonal_casc