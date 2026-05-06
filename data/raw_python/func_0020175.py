def Scatter(y, win=13, remove_outliers=False):
    '''
    Return the scatter in ppm based on the median running standard deviation
    for a window size of :py:obj:`win` = 13 cadences (for K2, this
    is ~6.5 hours, as in VJ14).

    :param ndarray y: The array whose CDPP is to be computed
    :param int win: The window size in cadences. Default `13`
    :param bool remove_outliers: Clip outliers at 5 sigma before computing \
           the CDPP? Default `False`

    '''

    if remove_outliers:
        # Remove 5-sigma outliers from data
        # smoothed on a 1 day timescale
        if len(y) >= 50:
            ys = y - Smooth(y, 50)
        else:
            ys = y
        M = np.nanmedian(ys)
        MAD = 1.4826 * np.nanmedian(np.abs(ys - M))
        out = []
        for i, _ in enumerate(y):
            if (ys[i] > M + 5 * MAD) or (ys[i] < M - 5 * MAD):
                out.append(i)
        out = np.array(out, dtype=int)
        y = np.delete(y, out)
    if len(y):
        return 1.e6 * np.nanmedian([np.std(yi) / np.sqrt(win)
                                    for yi in Chunks(y, win, all=True)])
    else:
        return np.nan