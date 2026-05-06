def _cont_norm_running_quantile_regions_mp(wl, fluxes, ivars, q, delta_lambda,
                                           ranges, n_proc=2, verbose=False):
    """
    Perform continuum normalization using running quantile, for spectrum
    that comes in chunks.

    The same as _cont_norm_running_quantile_regions(),
    but using multi-processing.

    Bo Zhang (NAOC)
    """
    print("contnorm.py: continuum norm using running quantile")
    print("Taking spectra in %s chunks" % len(ranges))
    # nstars = fluxes.shape[0]
    nchunks = len(ranges)
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for i in xrange(nchunks):
        chunk = ranges[i, :]
        start = chunk[0]
        stop = chunk[1]
        if verbose:
            print('@Bo Zhang: Going to normalize Chunk [%d/%d], pixel:[%d, %d] ...'
                  % (i+1, nchunks, start, stop))
        output = _cont_norm_running_quantile_mp(
            wl[start:stop], fluxes[:, start:stop],
            ivars[:, start:stop], q, delta_lambda,
            n_proc=n_proc, verbose=verbose)
        norm_fluxes[:, start:stop] = output[0]
        norm_ivars[:, start:stop] = output[1]
    return norm_fluxes, norm_ivars