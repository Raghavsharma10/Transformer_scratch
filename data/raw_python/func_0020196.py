def CDPP(flux, mask=[], cadence='lc'):
    '''
    Compute the proxy 6-hr CDPP metric.

    :param array_like flux: The flux array to compute the CDPP for
    :param array_like mask: The indices to be masked
    :param str cadence: The light curve cadence. Default `lc`

    '''

    # 13 cadences is 6.5 hours
    rmswin = 13
    # Smooth the data on a 2 day timescale
    svgwin = 49

    # If short cadence, need to downbin
    if cadence == 'sc':
        newsize = len(flux) // 30
        flux = Downbin(flux, newsize, operation='mean')

    flux_savgol = SavGol(np.delete(flux, mask), win=svgwin)
    if len(flux_savgol):
        return Scatter(flux_savgol / np.nanmedian(flux_savgol),
                       remove_outliers=True, win=rmswin)
    else:
        return np.nan