def FitCBVs(model):
    '''
    Fits the CBV design matrix to the de-trended flux of a given target. This
    is called internally whenever the user accesses the :py:attr:`fcor`
    attribute.

    :param model: An instance of the :py:obj:`everest` model for the target

    '''

    # Get cbvs?
    if model.XCBV is None:
        GetTargetCBVs(model)

    # The number of CBVs to use
    ncbv = model.cbv_num

    # Need to treat short and long cadences differently
    if model.cadence == 'lc':

        # Loop over all the light curve segments
        m = [None for b in range(len(model.breakpoints))]
        weights = [None for b in range(len(model.breakpoints))]
        for b in range(len(model.breakpoints)):

            # Get the indices for this light curve segment
            inds = model.get_chunk(b, pad=False)
            masked_inds = model.get_masked_chunk(b, pad=False)

            # Regress
            mX = model.XCBV[masked_inds, :ncbv + 1]
            A = np.dot(mX.T, mX)
            B = np.dot(mX.T, model.flux[masked_inds])
            try:
                weights[b] = np.linalg.solve(A, B)
            except np.linalg.linalg.LinAlgError:
                # Singular matrix
                log.warn('Singular matrix!')
                weights[b] = np.zeros(mX.shape[1])

            m[b] = np.dot(model.XCBV[inds, :ncbv + 1], weights[b])

            # Vertical alignment
            if b == 0:
                m[b] -= np.nanmedian(m[b])
            else:
                # Match the first finite model point on either side of the
                # break
                # We could consider something more elaborate in the future
                i0 = -1 - np.argmax([np.isfinite(m[b - 1][-i])
                                     for i in range(1, len(m[b - 1]) - 1)])
                i1 = np.argmax([np.isfinite(m[b][i])
                                for i in range(len(m[b]))])
                m[b] += (m[b - 1][i0] - m[b][i1])

        # Join model and normalize
        m = np.concatenate(m)
        m -= np.nanmedian(m)

    else:

        # Interpolate over outliers so we don't have to worry
        # about masking the arrays below
        flux = Interpolate(model.time, model.mask, model.flux)

        # Get downbinned light curve
        newsize = len(model.time) // 30
        time = Downbin(model.time, newsize, operation='mean')
        flux = Downbin(flux, newsize, operation='mean')

        # Get LC breakpoints
        breakpoints = list(Breakpoints(
            model.ID, season=model.season, cadence='lc'))
        breakpoints += [len(time) - 1]

        # Loop over all the light curve segments
        m = [None for b in range(len(breakpoints))]
        weights = [None for b in range(len(breakpoints))]
        for b in range(len(breakpoints)):

            # Get the indices for this light curve segment
            M = np.arange(len(time))
            if b > 0:
                inds = M[(M > breakpoints[b - 1]) & (M <= breakpoints[b])]
            else:
                inds = M[M <= breakpoints[b]]

            # Regress
            A = np.dot(model.XCBV[inds, :ncbv + 1].T,
                       model.XCBV[inds, :ncbv + 1])
            B = np.dot(model.XCBV[inds, :ncbv + 1].T, flux[inds])
            weights[b] = np.linalg.solve(A, B)
            m[b] = np.dot(model.XCBV[inds, :ncbv + 1], weights[b])

            # Vertical alignment
            if b == 0:
                m[b] -= np.nanmedian(m[b])
            else:
                # Match the first finite model point on either side of the
                # break
                # We could consider something more elaborate in the future
                i0 = -1 - np.argmax([np.isfinite(m[b - 1][-i])
                                     for i in range(1, len(m[b - 1]) - 1)])
                i1 = np.argmax([np.isfinite(m[b][i])
                                for i in range(len(m[b]))])
                m[b] += (m[b - 1][i0] - m[b][i1])

        # Join model and normalize
        m = np.concatenate(m)
        m -= np.nanmedian(m)

        # Finally, interpolate back to short cadence
        m = np.interp(model.time, time, m)

    return m