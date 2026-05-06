def extract_dark(prihdr, scihdu):
    """Extract superdark data from ``DARKFILE`` or ``DRKCFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    dark : ndarray or `None`
        Superdark, if any. Subtract this to apply ``DARKCORR``.

    """
    if prihdr.get('PCTECORR', 'OMIT') == 'COMPLETE':
        darkfile = prihdr.get('DRKCFILE', 'N/A')
    else:
        darkfile = prihdr.get('DARKFILE', 'N/A')

    if darkfile == 'N/A':
        return None

    darkfile = from_irafpath(darkfile)
    ampstring = prihdr['CCDAMP']

    # Calculate DARKTIME
    exptime = prihdr.get('EXPTIME', 0.0)
    flashdur = prihdr.get('FLASHDUR', 0.0)
    darktime = exptime + flashdur
    if exptime > 0:  # Not BIAS
        darktime += 3.0

    with fits.open(darkfile) as hdudark:
        if ampstring == 'ABCD':
            dark = np.concatenate(
                (hdudark['sci', 1].data,
                 hdudark['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            dark = extract_ref(scihdu, hdudark['sci', 2])
        else:
            dark = extract_ref(scihdu, hdudark['sci', 1])

    dark = dark * darktime

    return dark