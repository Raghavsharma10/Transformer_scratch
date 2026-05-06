def extract_flatfield(prihdr, scihdu):
    """Extract flatfield data from ``PFLTFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    invflat : ndarray or `None`
        Inverse flatfield, if any. Multiply this to apply ``FLATCORR``.

    """
    for ff in ['DFLTFILE', 'LFLTFILE']:
        vv = prihdr.get(ff, 'N/A')
        if vv != 'N/A':
            warnings.warn('{0}={1} is not accounted for'.format(ff, vv),
                          AstropyUserWarning)

    flatfile = prihdr.get('PFLTFILE', 'N/A')

    if flatfile == 'N/A':
        return None

    flatfile = from_irafpath(flatfile)
    ampstring = prihdr['CCDAMP']

    with fits.open(flatfile) as hduflat:
        if ampstring == 'ABCD':
            invflat = np.concatenate(
                (1 / hduflat['sci', 1].data,
                 1 / hduflat['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            invflat = 1 / extract_ref(scihdu, hduflat['sci', 2])
        else:
            invflat = 1 / extract_ref(scihdu, hduflat['sci', 1])

    return invflat