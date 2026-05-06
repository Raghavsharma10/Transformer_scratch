def get_aux_files(basename):
    """
    Look for and return all the aux files that are associated witht this filename.
    Will look for:
    background (_bkg.fits)
    rms        (_rms.fits)
    mask       (.mim)
    catalogue  (_comp.fits)
    psf map    (_psf.fits)

    will return filenames if they exist, or None where they do not.

    Parameters
    ----------
    basename : str
        The name/path of the input image.

    Returns
    -------
    aux : dict
        Dict of filenames or None with keys (bkg, rms, mask, cat, psf)
    """
    base = os.path.splitext(basename)[0]
    files = {"bkg": base + "_bkg.fits",
             "rms": base + "_rms.fits",
             "mask": base + ".mim",
             "cat": base + "_comp.fits",
             "psf": base + "_psf.fits"}

    for k in files.keys():
        if not os.path.exists(files[k]):
            files[k] = None
    return files