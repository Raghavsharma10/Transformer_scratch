def djs_iterstat(InputArr, MaxIter=10, SigRej=3.0,
                 Max=None, Min=None, Mask=None, lineno=None):
    """
    Iterative sigma-clipping.

    Parameters
    ----------
    InputArr : `numpy.ndarray`
        Input image array.

    MaxIter, SigRej : see `clean`

    Max, Min : float
        Max and min values for clipping.

    Mask : `numpy.ndarray`
        Mask array to indicate pixels to reject, in addition to clipping.
        Pixels where mask is zero will be rejected.
        If not given, all pixels will be used.

    lineno : int or None
        Line number to be used in log and/or warning messages.

    Returns
    -------
    FMean, FSig, FMedian, NPix : float
        Mean, sigma, and median of final result.

    NIter : int
        Number of performed clipping iterations

    BMask : `numpy.ndarray`
        Logical image mask from the final iteration.

    """
    NGood = InputArr.size
    ArrShape = InputArr.shape
    if NGood == 0:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - No data points given' + imrow)
        return 0, 0, 0, 0, 0, None
    if NGood == 1:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - Only one data point; '
                    'cannot compute stats{0}'.format(imrow))
        return 0, 0, 0, 0, 0, None
    if np.unique(InputArr).size == 1:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - Only one value in data; '
                    'cannot compute stats{0}'.format(imrow))
        return 0, 0, 0, 0, 0, None

    # Determine Max and Min
    if Max is None:
        Max = InputArr.max()
    if Min is None:
        Min = InputArr.min()

    # Use all pixels if no mask is provided
    if Mask is None:
        Mask = np.ones(ArrShape, dtype=np.byte)
    else:
        Mask = Mask.copy()

    # Reject those above Max and those below Min
    Mask[InputArr > Max] = 0
    Mask[InputArr < Min] = 0

    FMean = np.sum(1.0 * InputArr * Mask) / NGood
    FSig  = np.sqrt(np.sum((1.0 * InputArr - FMean) ** 2 * Mask) / (NGood - 1))

    NLast = -1
    Iter  = 0
    NGood = np.sum(Mask)
    if NGood < 2:
        imrow = _write_row_number(lineno=lineno, offset=0, pad=1)
        LOG.warning('djs_iterstat - No good data points; '
                    'cannot compute stats{0}'.format(imrow))
        return 0, 0, 0, 0, 0, None

    SaveMask = Mask.copy()
    if Iter >= MaxIter:  # to support MaxIter=0
        NLast = NGood

    while (Iter < MaxIter) and (NLast != NGood) and (NGood >= 2):
        LoVal = FMean - SigRej * FSig
        HiVal = FMean + SigRej * FSig

        Mask[InputArr < LoVal] = 0
        Mask[InputArr > HiVal] = 0
        NLast = NGood
        npix = np.sum(Mask)

        if npix >= 2:
            FMean = np.sum(1.0 * InputArr * Mask) / npix
            FSig = np.sqrt(np.sum(
                (1.0 * InputArr - FMean) ** 2 * Mask) / (npix - 1))
            SaveMask = Mask.copy()  # last mask used for computation of mean
            NGood = npix
            Iter += 1
        else:
            break

    logical_mask = SaveMask.astype(np.bool)

    if NLast > 1:
        FMedian = np.median(InputArr[logical_mask])
        NLast = NGood
    else:
        FMedian = FMean

    return FMean, FSig, FMedian, NLast, Iter, logical_mask