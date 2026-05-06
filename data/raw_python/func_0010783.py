def aggregate_precipitation(vec_data,hourly=True, percentile=50):
    """Aggregates highly resolved precipitation data and creates statistics

    Parameters
    ----------
    vec_data : pd.Series
        hourly (hourly=True) OR 5-min values 

    Returns
    -------
    output : cascade object
        representing statistics of the cascade model
    """
    cascade_opt = cascade.CascadeStatistics()
    cascade_opt.percentile = percentile

    # length of input time series
    n_in = len(vec_data)
    n_out = np.floor(n_in/2)

    # alternative:
    # 1st step: new time series
    vec_time = vec_data.index
    vdn0 = []
    vtn0 = []
    j = 0
    for i in range(0, n_in):
        if np.mod(i, 2) != 0:
            vdn0.append(vec_data.precip.values[i-1] + vec_data.precip.values[i])
            vtn0.append(vec_time[i])
            j = j+1

    vdn = pd.DataFrame(index=vtn0, data={'precip': vdn0})

    # length of new time series
    n_out = len(vdn)

    # series of box types:
    vbtype = np.zeros((n_out, ), dtype=np.int)

    # fields for empirical probabilities
    # counts
    nb = np.zeros((2, 4))
    nbxx = np.zeros((2, 4))

    # class boundaries for histograms
    # wclassbounds = np.linspace(0, 1, num=8)
    wlower = np.array([0,
                       0.1429,
                       0.2857,
                       0.4286,
                       0.5714,
                       0.7143,
                       0.8571])  # wclassbounds[0:7]
    wupper = np.array([0.1429,
                       0.2857,
                       0.4286,
                       0.5714,
                       0.7143,
                       0.8571,
                       1.0])  # wclassbounds[1:8]

    # evaluate mean rainfall intensity for wet boxes
    # these values should be determined during the aggregation phase!!!!!
    # mean volume threshold
    meanvol = np.percentile(vdn.precip[vdn.precip > 0.],
                            cascade_opt.percentile)  # np.mean(vdn.precip[vdn.precip>0.])
    cascade_opt.threshold = np.array([meanvol])

    # 2nd step: classify boxes at the upper level
    for i in range(0, n_out):
        if vdn.precip.values[i] > 0.:  # rain?
            if i == 0:  # only starting or isolated
                if vdn.precip.values[i+1] > 0.:
                    vbtype[i] = cascade.BoxTypes.starting
                else:
                    vbtype[i] = cascade.BoxTypes.isolated
            elif i == n_out-1:  # only ending or isolated
                if vdn.precip.values[i-1] > 0.:
                    vbtype[i] = cascade.BoxTypes.ending
                else:
                    vbtype[i] = cascade.BoxTypes.isolated
            else:  # neither at at the end nor at the beginning
                if vdn.precip.values[i-1] == 0. and vdn.precip.values[i+1] == 0.:
                    vbtype[i] = cascade.BoxTypes.isolated
                if vdn.precip.values[i-1] == 0. and vdn.precip.values[i+1] > 0.:
                    vbtype[i] = cascade.BoxTypes.starting
                if vdn.precip.values[i-1] > 0. and vdn.precip.values[i+1] > 0.:
                    vbtype[i] = cascade.BoxTypes.enclosed
                if vdn.precip.values[i-1] > 0. and vdn.precip.values[i+1] == 0.:
                    vbtype[i] = cascade.BoxTypes.ending
        else:
            vbtype[i] = cascade.BoxTypes.dry  # no rain

    # 3rd step: examine branching
    j = 0
    for i in range(0, n_in):
        if np.mod(i, 2) != 0:
            if vdn.precip.values[j] > 0:
                if vdn.precip.values[j] > meanvol:
                    belowabove = 1  # above mean
                else:
                    belowabove = 0  # below mean

                nb[belowabove, vbtype[j]-1] += 1

                if vec_data.precip.values[i-1] > 0 and vec_data.precip.values[i] == 0:
                    # P(1/0)
                    cascade_opt.p10[belowabove, vbtype[j]-1] += 1

                if vec_data.precip.values[i-1] == 0 and vec_data.precip.values[i] > 0:
                    # P(0/1)
                    cascade_opt.p01[belowabove, vbtype[j]-1] += 1

                if vec_data.precip.values[i-1] > 0 and vec_data.precip.values[i] > 0:
                    # P(x/x)
                    cascade_opt.pxx[belowabove, vbtype[j]-1] += 1

                    nbxx[belowabove, vbtype[j]-1] += 1

                    # weights
                    r1 = vec_data.precip.values[i-1]
                    r2 = vec_data.precip.values[i]
                    wxxval = r1 / (r1 + r2)

                    # Test
                    if abs(r1+r2-vdn.precip.values[j]) > 1.E-3:
                        print('i=' + str(i) + ', j=' + str(j) +
                              ', r1=' + str(r1) + ", r2=" + str(r2) +
                              ", Summe=" + str(vdn.precip.values[j]))
                        print(vec_data.index[i])
                        print(vdn.index[j])
                        print('error')
                        return cascade_opt, vdn

                    for k in range(0, 7):
                        if wxxval > wlower[k] and wxxval <= wupper[k]:
                            cascade_opt.wxx[k, belowabove, vbtype[j]-1] += 1
                            break
            j = j + 1

    # 4th step: transform counts to percentages
    cascade_opt.p01 = cascade_opt.p01 / nb
    cascade_opt.p10 = cascade_opt.p10 / nb
    cascade_opt.pxx = cascade_opt.pxx / nb

    with np.errstate(divide='ignore', invalid='ignore'):  # do not issue warnings here when dividing by zero, this is handled below
        for k in range(0, 7):
            cascade_opt.wxx[k, :, :] = cascade_opt.wxx[k, :, :] / nbxx[:, :]

    # In some cases, the time series are too short for deriving statistics.

    if (np.isnan(cascade_opt.p01).any() or
            np.isnan(cascade_opt.p10).any() or
            np.isnan(cascade_opt.pxx).any()):
        print("ERROR (branching probabilities):")
        print("Invalid statistics. Default values will be returned. "
              "Try to use longer time series or apply statistics "
              "derived for another station.")
        cascade_opt.fill_with_sample_data()

    # For some box types, the corresponding probabilities might yield nan.
    # If this happens, nan values will be replaced by 1/7 in order to provide
    # valid values for disaggregation.
    if np.isnan(cascade_opt.wxx).any():
        print("Warning (weighting probabilities):")
        print("The derived cascade statistics are not valid as some "
              "probabilities are undefined! ", end="")
        print("Try to use longer time series that might be more "
              "appropriate for deriving statistics. ", end="")
        print("As a workaround, default values according to equally "
              "distributed probabilities ", end="")
        print("will be applied...", end="")
        cascade_opt.wxx[np.isnan(cascade_opt.wxx)] = 1.0 / 7.0
        wxx = np.zeros((2, 4))
        for k in range(0, 7):
            wxx[:, :] += cascade_opt.wxx[k, :, :]
        if wxx.any() > 1.001 or wxx.any() < 0.999:
            print("failed! Using default values!")
            cascade_opt.fill_with_sample_data()
        else:
            print("OK!")

    return cascade_opt, vdn