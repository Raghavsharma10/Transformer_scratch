def GetStars(campaign, module, model='nPLD', **kwargs):
    '''
    Returns de-trended light curves for all stars on a given module in
    a given campaign.

    '''

    # Get the channel numbers
    channels = Channels(module)
    assert channels is not None, "No channels available on this module."

    # Get the EPIC numbers
    all = GetK2Campaign(campaign)
    stars = np.array([s[0] for s in all if s[2] in channels and
                      os.path.exists(
        os.path.join(EVEREST_DAT, 'k2', 'c%02d' % int(campaign),
                     ('%09d' % s[0])[:4] + '00000',
                     ('%09d' % s[0])[4:], model + '.npz'))], dtype=int)
    N = len(stars)
    assert N > 0, "No light curves found for campaign %d, module %d." % (
        campaign, module)

    # Loop over all stars and store the fluxes in a list
    fluxes = []
    errors = []
    kpars = []

    for n in range(N):

        # De-trended light curve file name
        nf = os.path.join(EVEREST_DAT, 'k2', 'c%02d' % int(campaign),
                          ('%09d' % stars[n])[:4] + '00000',
                          ('%09d' % stars[n])[4:], model + '.npz')

        # Get the data
        data = np.load(nf)
        t = data['time']
        if n == 0:
            time = t
            breakpoints = data['breakpoints']

        # Get de-trended light curve
        y = data['fraw'] - data['model']
        err = data['fraw_err']

        # De-weight outliers and bad timestamps
        m = np.array(list(set(np.concatenate([data['outmask'], data['badmask'],
                                              data['nanmask'],
                                              data['transitmask']]))),
                     dtype=int)

        # Interpolate over the outliers
        y = np.interp(t, np.delete(t, m), np.delete(y, m))
        err = np.interp(t, np.delete(t, m), np.delete(err, m))

        # Append to our running lists
        fluxes.append(y)
        errors.append(err)
        kpars.append(data['kernel_params'])

    return time, breakpoints, np.array(fluxes), \
           np.array(errors), np.array(kpars)