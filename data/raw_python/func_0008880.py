def expand(datafile, outfile=None):
    """
    Expand and interpolate the given data file using the given method.
    Datafile can be a filename or an HDUList

    It is assumed that the file has been compressed and that there are `BN_?` keywords in the
    fits header that describe how the compression was done.

    Parameters
    ----------
    datafile : str or HDUList
        filename or HDUList of file to work on

    outfile : str
        filename to write to (default = None)

    Returns
    -------
    hdulist : HDUList
        HDUList of the expanded data.

    See Also
    --------
    :func:`AegeanTools.fits_interp.compress`

    """
    hdulist = load_file_or_hdu(datafile)

    header = hdulist[0].header
    data = hdulist[0].data
    # Check for the required key words, only expand if they exist
    if not all(a in header for a in ['BN_CFAC', 'BN_NPX1', 'BN_NPX2', 'BN_RPX1', 'BN_RPX2']):
        return hdulist

    factor = header['BN_CFAC']
    (gx, gy) = np.mgrid[0:header['BN_NPX2'], 0:header['BN_NPX1']]
    # fix the last column of the grid to account for residuals
    lcx = header['BN_RPX2']
    lcy = header['BN_RPX1']

    rows = (np.arange(data.shape[0]) + int(lcx/factor))*factor
    cols = (np.arange(data.shape[1]) + int(lcy/factor))*factor

    # Do the interpolation
    hdulist[0].data = np.array(RegularGridInterpolator((rows,cols), data)((gx, gy)), dtype=np.float32)

    # update the fits keywords so that the WCS is correct
    header['CRPIX1'] = (header['CRPIX1'] - 1) * factor + 1
    header['CRPIX2'] = (header['CRPIX2'] - 1) * factor + 1

    if 'CDELT1' in header:
        header['CDELT1'] /= factor
    elif 'CD1_1' in header:
        header['CD1_1'] /= factor
    else:
        logging.error("Error: Can't find CD1_1 or CDELT1")
        return None

    if 'CDELT2' in header:
        header['CDELT2'] /= factor
    elif "CD2_2" in header:
        header['CD2_2'] /= factor
    else:
        logging.error("Error: Can't find CDELT2 or CD2_2")
        return None

    header['HISTORY'] = 'Expanded by factor {0}'.format(factor)

    # don't need these any more so delete them.
    del header['BN_CFAC'], header['BN_NPX1'], header['BN_NPX2'], header['BN_RPX1'], header['BN_RPX2']
    hdulist[0].header = header
    if outfile is not None:
        hdulist.writeto(outfile, overwrite=True)
        logging.info("Wrote: {0}".format(outfile))
    return hdulist