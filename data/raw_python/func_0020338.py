def GetCBVs(campaign, model='nPLD', clobber=False, **kwargs):
    '''
    Computes the CBVs for a given campaign.

    :param int campaign: The campaign number
    :param str model: The name of the :py:obj:`everest` model. Default `nPLD`
    :param bool clobber: Overwrite existing files? Default `False`

    '''

    # Initialize logging?
    if len(logging.getLogger().handlers) == 0:
        InitLog(file_name=None, screen_level=logging.DEBUG)
    log.info('Computing CBVs for campaign %d...' % (campaign))

    # Output path
    path = os.path.join(EVEREST_DAT, 'k2', 'cbv', 'c%02d' % campaign)
    if not os.path.exists(path):
        os.makedirs(path)

    # Get the design matrix
    xfile = os.path.join(path, 'X.npz')
    if clobber or not os.path.exists(xfile):

        log.info('Obtaining light curves...')
        time = None
        for module in range(2, 25):

            # Get the light curves
            lcfile = os.path.join(path, '%d.npz' % module)
            if clobber or not os.path.exists(lcfile):
                try:
                    time, breakpoints, fluxes, errors, kpars = GetStars(
                        campaign, module, model=model, **kwargs)
                except AssertionError:
                    continue
                np.savez(lcfile, time=time, breakpoints=breakpoints,
                         fluxes=fluxes, errors=errors, kpars=kpars)

            # Load the light curves
            lcs = np.load(lcfile)
            if time is None:
                time = lcs['time']
                breakpoints = lcs['breakpoints']
                fluxes = lcs['fluxes']
                errors = lcs['errors']
                kpars = lcs['kpars']
            else:
                fluxes = np.vstack([fluxes, lcs['fluxes']])
                errors = np.vstack([errors, lcs['errors']])
                kpars = np.vstack([kpars, lcs['kpars']])

        # Compute the design matrix
        log.info('Running SysRem...')
        X = np.ones((len(time), 1 + kwargs.get('ncbv', 5)))

        # Loop over the segments
        new_fluxes = np.zeros_like(fluxes)
        for b in range(len(breakpoints)):

            # Get the current segment's indices
            inds = GetChunk(time, breakpoints, b)

            # Update the error arrays with the white GP component
            for j in range(len(errors)):
                errors[j] = np.sqrt(errors[j] ** 2 + kpars[j][0] ** 2)

            # Get de-trended fluxes
            X[inds, 1:] = SysRem(time[inds], fluxes[:, inds],
                                 errors[:, inds], **kwargs).T

        # Save
        np.savez(xfile, X=X, time=time, breakpoints=breakpoints)

    else:

        # Load from disk
        data = np.load(xfile)
        X = data['X'][()]
        time = data['time'][()]
        breakpoints = data['breakpoints'][()]

    # Plot
    plotfile = os.path.join(path, 'X.pdf')
    if clobber or not os.path.exists(plotfile):
        fig, ax = pl.subplots(2, 3, figsize=(12, 8))
        fig.subplots_adjust(left=0.05, right=0.95)
        ax = ax.flatten()
        for axis in ax:
            axis.set_xticks([])
            axis.set_yticks([])
        for b in range(len(breakpoints)):
            inds = GetChunk(time, breakpoints, b)
            for n in range(min(6, X.shape[1])):
                ax[n].plot(time[inds], X[inds, n])
                ax[n].set_title(n, fontsize=14)
        fig.savefig(plotfile, bbox_inches='tight')

    return X