def write_moc_fits(moc, filename, **kwargs):
    """Write a MOC as a FITS file.

    Any additional keyword arguments are passed to the
    astropy.io.fits.HDUList.writeto method.
    """

    tbhdu = write_moc_fits_hdu(moc)
    prihdr = fits.Header()
    prihdu = fits.PrimaryHDU(header=prihdr)
    hdulist = fits.HDUList([prihdu, tbhdu])
    hdulist.writeto(filename, **kwargs)