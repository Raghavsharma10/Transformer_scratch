def extract_flash(prihdr, scihdu):
    """Extract postflash data from ``FLSHFILE``.

    Parameters
    ----------
    prihdr : obj
        FITS primary header HDU.

    scihdu : obj
        Extension HDU of the science image.
        This is only used to extract subarray data.

    Returns
    -------
    flash : ndarray or `None`
        Postflash, if any. Subtract this to apply ``FLSHCORR``.

    """
    flshfile = prihdr.get('FLSHFILE', 'N/A')
    flashsta = prihdr.get('FLASHSTA', 'N/A')
    flashdur = prihdr.get('FLASHDUR', 0.0)

    if flshfile == 'N/A' or flashdur <= 0:
        return None

    if flashsta != 'SUCCESSFUL':
        warnings.warn('Flash status is {0}'.format(flashsta),
                      AstropyUserWarning)

    flshfile = from_irafpath(flshfile)
    ampstring = prihdr['CCDAMP']

    with fits.open(flshfile) as hduflash:
        if ampstring == 'ABCD':
            flash = np.concatenate(
                (hduflash['sci', 1].data,
                 hduflash['sci', 2].data[::-1, :]), axis=1)
        elif ampstring in ('A', 'B', 'AB'):
            flash = extract_ref(scihdu, hduflash['sci', 2])
        else:
            flash = extract_ref(scihdu, hduflash['sci', 1])

    flash = flash * flashdur

    return flash