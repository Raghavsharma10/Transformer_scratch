def hdr_vals_for_overscan(root):
    """Retrieve header keyword values from RAW and SPT
    FITS files to pass on to :func:`check_oscntab` and
    :func:`check_overscan`.

    Parameters
    ----------
    root : str
        Rootname of the observation. Can be relative path
        to the file excluding the type of FITS file and
        extension, e.g., '/my/path/jxxxxxxxq'.

    Returns
    -------
    ccdamp : str
        Amplifiers used to read out the CCDs.

    xstart : int
        Starting column of the readout in detector
        coordinates.

    ystart : int
        Starting row of the readout in detector
        coordinates.

    xsize : int
        Number of columns in the readout.

    ysize : int
        Number of rows in the readout.

    """
    with fits.open(root + '_spt.fits') as hdu:
        spthdr = hdu[0].header
    with fits.open(root + '_raw.fits') as hdu:
        prihdr = hdu[0].header
    xstart = spthdr['SS_A1CRN']
    ystart = spthdr['SS_A2CRN']
    xsize = spthdr['SS_A1SZE']
    ysize = spthdr['SS_A2SZE']
    ccdamp = prihdr['CCDAMP']

    return ccdamp, xstart, ystart, xsize, ysize