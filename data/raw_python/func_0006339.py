def generate_threshold_mask(hist):
    '''Masking array elements when equal 0.0 or greater than 10 times the median

    Parameters
    ----------
    hist : array_like
        Input data.

    Returns
    -------
    masked array
        Returns copy of the array with masked elements.
    '''
    masked_array = np.ma.masked_values(hist, 0)
    masked_array = np.ma.masked_greater(masked_array, 10 * np.ma.median(hist))
    logging.info('Masking %d pixel(s)', np.ma.count_masked(masked_array))
    return np.ma.getmaskarray(masked_array)