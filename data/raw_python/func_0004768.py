def estimate_noise(fluxes, contmask):
    """ Estimate the scatter in a region of the spectrum
    taken to be continuum """
    nstars = fluxes.shape[0]
    scatter = np.zeros(nstars)
    for i,spec in enumerate(fluxes): 
        cont = spec[contmask]
        scatter[i] = stats.funcs.mad_std(cont)
    return scatter