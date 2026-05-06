def write_fits(data, header, file_name):
    """
    Combine data and a fits header to write a fits file.

    Parameters
    ----------
    data : numpy.ndarray
        The data to be written.

    header : astropy.io.fits.hduheader
        The header for the fits file.

    file_name : string
        The file to write

    Returns
    -------
    None
    """
    hdu = fits.PrimaryHDU(data)
    hdu.header = header
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(file_name, overwrite=True)
    logging.info("Wrote {0}".format(file_name))
    return