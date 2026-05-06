def corrcoef(time, crossf, integration_window=0.):
    """
    Calculate the correlation coefficient for given auto- and crosscorrelation
    functions. Standard settings yield the zero lag correlation coefficient.
    Setting integration_window > 0 yields the correlation coefficient of
    integrated auto- and crosscorrelation functions. The correlation coefficient
    between a zero signal with any other signal is defined as 0.


    Parameters
    ----------
    time : numpy.ndarray
        1 dim array of times corresponding to signal.
    crossf : numpy.ndarray 
        Crosscorrelation functions, 1st axis first unit, 2nd axis second unit,
        3rd axis times.
    integration_window: float
        Size of the integration window.


    Returns
    -------
    cc : numpy.ndarray
        2 dim array of correlation coefficient between two units.

    """

    N = len(crossf)
    cc = np.zeros(np.shape(crossf)[:-1])
    tbin = abs(time[1] - time[0])
    lim = int(integration_window / tbin)
    
    if len(time)%2 == 0:
        mid = len(time)/2-1
    else:
        mid = np.floor(len(time)/2.)
    
    for i in range(N):
        ai = np.sum(crossf[i, i][mid - lim:mid + lim + 1])
        offset_autoi = np.mean(crossf[i,i][:mid-1])
        for j in range(N):
            cij = np.sum(crossf[i, j][mid - lim:mid + lim + 1])
            offset_cross = np.mean(crossf[i,j][:mid-1])
            aj = np.sum(crossf[j, j][mid - lim:mid + lim + 1])
            offset_autoj = np.mean(crossf[j,j][:mid-1])
            if ai > 0. and aj > 0.:
                cc[i, j] = (cij-offset_cross) / np.sqrt((ai-offset_autoi) * \
                    (aj-offset_autoj))
            else:
                cc[i, j] = 0.
    
    return cc