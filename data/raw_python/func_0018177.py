def perform_correction(image, output, stat="pmode1", maxiter=15, sigrej=2.0,
                       lower=None, upper=None, binwidth=0.3,
                       mask=None, dqbits=None,
                       rpt_clean=0, atol=0.01, clobber=False, verbose=True):
    """
    Clean each input image.

    Parameters
    ----------
    image : str
        Input image name.

    output : str
        Output image name.

    mask : `numpy.ndarray`
        Mask array.

    maxiter, sigrej, clobber
        See :func:`clean`.

    dqbits : int, str, or None
        Data quality bits to be considered as "good" (or "bad").
        See :func:`clean` for more details.

    rpt_clean : int
        An integer indicating how many *additional* times stripe cleaning
        should be performed on the input image. Default = 0.

    atol : float, None
        The threshold for maximum absolute value of bias stripe correction
        below which repeated cleanings can stop. When `atol` is `None`
        cleaning will be repeated `rpt_clean` number of times.
        Default = 0.01 [e].

    verbose : bool
        Print informational messages. Default = True.

    """
    # construct the frame to be cleaned, including the
    # associated data stuctures needed for cleaning
    frame = StripeArray(image)

    # combine user mask with image's DQ array:
    mask = _mergeUserMaskAndDQ(frame.dq, mask, dqbits)

    # Do the stripe cleaning
    Success, NUpdRows, NMaxIter, Bkgrnd, STDDEVCorr, MaxCorr, Nrpt = clean_streak(
        frame, stat=stat, maxiter=maxiter, sigrej=sigrej,
        lower=lower, upper=upper, binwidth=binwidth, mask=mask,
        rpt_clean=rpt_clean, atol=atol, verbose=verbose
    )

    if Success:
        if verbose:
            LOG.info('perform_correction - =====  Overall statistics for '
                     'de-stripe corrections:  =====')

        if (STDDEVCorr > 1.5*0.9):
            LOG.warning('perform_correction - STDDEV of applied de-stripe '
                        'corrections ({:.3g}) exceeds\nknown bias striping '
                        'STDDEV of 0.9e (see ISR ACS 2011-05) more than '
                        '1.5 times.'.format(STDDEVCorr))

        elif verbose:
            LOG.info('perform_correction - STDDEV of applied de-stripe '
                     'corrections {:.3g}.'.format(STDDEVCorr))

        if verbose:
            LOG.info('perform_correction - Estimated background: '
                     '{:.5g}.'.format(Bkgrnd))
            LOG.info('perform_correction - Maximum applied correction: '
                     '{:.3g}.'.format(MaxCorr))
            LOG.info('perform_correction - Effective number of clipping '
                     'iterations: {}.'.format(NMaxIter))
            LOG.info('perform_correction - Effective number of additional '
                     '(repeated) cleanings: {}.'.format(Nrpt))
            LOG.info('perform_correction - Total number of corrected rows: '
                     '{}.'.format(NUpdRows))

    frame.write_corrected(output, clobber=clobber)
    frame.close()