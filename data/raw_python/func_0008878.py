def load_file_or_hdu(filename):
    """
    Load a file from disk and return an HDUList
    If filename is already an HDUList return that instead

    Parameters
    ----------
    filename : str or HDUList
        File or HDU to be loaded

    Returns
    -------
    hdulist : HDUList
    """
    if isinstance(filename, fits.HDUList):
        hdulist = filename
    else:
        hdulist = fits.open(filename, ignore_missing_end=True)
    return hdulist