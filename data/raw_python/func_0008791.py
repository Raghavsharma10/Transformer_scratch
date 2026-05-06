def nan_acf(noise):
    """
    Calculate the autocorrelation function of the noise
    where the noise is a 2d array that may contain nans


    Parameters
    ----------
    noise : 2d-array
        Noise image.

    Returns
    -------
    acf : 2d-array
        The ACF.
    """
    corr = np.zeros(noise.shape)
    ix,jx = noise.shape
    for i in range(ix):
        si_min = slice(i, None, None)
        si_max = slice(None, ix-i, None)
        for j in range(jx):
            sj_min = slice(j, None, None)
            sj_max = slice(None, jx-j, None)
            if np.all(np.isnan(noise[si_min, sj_min])) or np.all(np.isnan(noise[si_max, sj_max])):
                corr[i, j] = np.nan
            else:
                corr[i, j] = np.nansum(noise[si_min, sj_min] * noise[si_max, sj_max])
    # return the normalised acf
    return corr / np.nanmax(corr)