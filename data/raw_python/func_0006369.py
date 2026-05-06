def select_good_pixel_region(hits, col_span, row_span, min_cut_threshold=0.2, max_cut_threshold=2.0):
    '''Takes the hit array and masks all pixels with a certain occupancy.

    Parameters
    ----------
    hits : array like
        If dim > 2 the additional dimensions are summed up.
    min_cut_threshold : float
        A number to specify the minimum threshold, which pixel to take. Pixels are masked if
        occupancy < min_cut_threshold * np.ma.median(occupancy)
        0 means that no pixels are masked
    max_cut_threshold : float
        A number to specify the maximum threshold, which pixel to take. Pixels are masked if
        occupancy > max_cut_threshold * np.ma.median(occupancy)
        Can be set to None that no pixels are masked by max_cut_threshold

    Returns
    -------
    numpy.ma.array, shape=(80,336)
        The hits array with masked pixels.
    '''
    hits = np.sum(hits, axis=(-1)).astype('u8')
    mask = np.ones(shape=(80, 336), dtype=np.uint8)

    mask[min(col_span):max(col_span) + 1, min(row_span):max(row_span) + 1] = 0

    ma = np.ma.masked_where(mask, hits)
    if max_cut_threshold is not None:
        return np.ma.masked_where(np.logical_or(ma < min_cut_threshold * np.ma.median(ma), ma > max_cut_threshold * np.ma.median(ma)), ma)
    else:
        return np.ma.masked_where(ma < min_cut_threshold * np.ma.median(ma), ma)