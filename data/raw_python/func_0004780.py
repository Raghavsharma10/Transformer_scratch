def _cont_norm_running_quantile_regions(wl, fluxes, ivars, q, delta_lambda,
                                        ranges, verbose=True):
    """ Perform continuum normalization using running quantile, for spectrum
    that comes in chunks
    """
    print("contnorm.py: continuum norm using running quantile")
    print("Taking spectra in %s chunks" % len(ranges))
    nstars = fluxes.shape[0]
    norm_fluxes = np.zeros(fluxes.shape)
    norm_ivars = np.zeros(ivars.shape)
    for chunk in ranges:
        start = chunk[0]
        stop = chunk[1]
        output = _cont_norm_running_quantile(
                wl[start:stop], fluxes[:,start:stop],
                ivars[:,start:stop], q, delta_lambda)
        norm_fluxes[:,start:stop] = output[0]
        norm_ivars[:,start:stop] = output[1]
    return norm_fluxes, norm_ivars