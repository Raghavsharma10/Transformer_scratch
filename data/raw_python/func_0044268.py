def read_moc_fits(moc, filename, include_meta=False, **kwargs):
    """Read data from a FITS file into a MOC.

    Any additional keyword arguments are passed to the
    astropy.io.fits.open method.
    """

    hl = fits.open(filename, mode='readonly', **kwargs)

    read_moc_fits_hdu(moc, hl[1], include_meta)