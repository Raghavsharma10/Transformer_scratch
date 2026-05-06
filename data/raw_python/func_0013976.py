def calculate_imf_steadiness(inst, steady_window=15, min_window_frac=0.75,
                             max_clock_angle_std=90.0/np.pi, max_bmag_cv=0.5):
    """ Calculate IMF steadiness using clock angle standard deviation and
    the coefficient of variation of the IMF magnitude in the GSM Y-Z plane

    Parameters
    -----------
    inst : pysat.Instrument
        Instrument with OMNI HRO data
    steady_window : int
        Window for calculating running statistical moments in min (default=15)
    min_window_frac : float
        Minimum fraction of points in a window for steadiness to be calculated
        (default=0.75)
    max_clock_angle_std : float
        Maximum standard deviation of the clock angle in degrees (default=22.5)
    max_bmag_cv : float
        Maximum coefficient of variation of the IMF magnitude in the GSM
        Y-Z plane (default=0.5)
    """

    # We are not going to interpolate through missing values
    sample_rate = int(inst.tag[0])
    max_wnum = np.floor(steady_window / sample_rate)
    if max_wnum != steady_window / sample_rate:
        steady_window = max_wnum * sample_rate
        print("WARNING: sample rate is not a factor of the statistical window")
        print("new statistical window is {:.1f}".format(steady_window))

    min_wnum = int(np.ceil(max_wnum * min_window_frac))

    # Calculate the running coefficient of variation of the BYZ magnitude
    byz_mean = inst['BYZ_GSM'].rolling(min_periods=min_wnum, center=True,
                                       window=steady_window).mean()
    byz_std = inst['BYZ_GSM'].rolling(min_periods=min_wnum, center=True,
                                      window=steady_window).std()
    inst['BYZ_CV'] = pds.Series(byz_std / byz_mean, index=inst.data.index)

    # Calculate the running circular standard deviation of the clock angle
    circ_kwargs = {'high':360.0, 'low':0.0}
    ca = inst['clock_angle'][~np.isnan(inst['clock_angle'])]
    ca_std = inst['clock_angle'].rolling(min_periods=min_wnum,
                                         window=steady_window, \
                center=True).apply(pysat.utils.nan_circstd, kwargs=circ_kwargs)
    inst['clock_angle_std'] = pds.Series(ca_std, index=inst.data.index)

    # Determine how long the clock angle and IMF magnitude are steady
    imf_steady = np.zeros(shape=inst.data.index.shape)

    steady = False
    for i,cv in enumerate(inst.data['BYZ_CV']):
        if steady:
            del_min = int((inst.data.index[i] -
                           inst.data.index[i-1]).total_seconds() / 60.0)
            if np.isnan(cv) or np.isnan(ca_std[i]) or del_min > sample_rate:
                # Reset the steadiness flag if fill values are encountered, or
                # if an entry is missing
                steady = False

        if cv <= max_bmag_cv and ca_std[i] <= max_clock_angle_std:
            # Steadiness conditions have been met
            if steady:
                imf_steady[i] = imf_steady[i-1]

            imf_steady[i] += sample_rate
            steady = True

    inst['IMF_Steady'] = pds.Series(imf_steady, index=inst.data.index)
    return