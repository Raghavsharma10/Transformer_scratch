def _cont_norm_running_quantile_mp(wl, fluxes, ivars, q, delta_lambda,
                                   n_proc=2, verbose=False):
    """
    The same as _cont_norm_running_quantile() above,
    but using multi-processing.

    Bo Zhang (NAOC)
    """
    nStar = fluxes.shape[0]

    # start mp.Pool
    mp_results = []
    pool = mp.Pool(processes=n_proc)
    for i in xrange(nStar):
        mp_results.append(pool.apply_async(\
            _find_cont_running_quantile,
            (wl, fluxes[i, :].reshape((1, -1)), ivars[i, :].reshape((1, -1)),
             q, delta_lambda), {'verbose': False}))
        if verbose:
            print('@Bo Zhang: continuum normalizing star [%d/%d] ...'\
                  % (i + 1, nStar))
    # close mp.Pool
    pool.close()
    pool.join()

    # reshape results --> cont
    cont = np.zeros_like(fluxes)
    for i in xrange(nStar):
        cont[i, :] = mp_results[i].get() #.flatten()
    norm_fluxes = np.ones(fluxes.shape)
    norm_fluxes[cont!=0] = fluxes[cont!=0] / cont[cont!=0]
    norm_ivars = cont**2 * ivars

    print('@Bo Zhang: continuum normalization finished!')
    return norm_fluxes, norm_ivars