def loadPng(varNumVol, tplPngSize, strPathPng):
    """Load PNG files.

    Parameters
    ----------
    varNumVol : float
        Number of volumes, i.e. number of time points in all runs.
    tplPngSize : tuple
        Shape of the stimulus image (i.e. png).
    strPathPng: str
        Path to the folder cointaining the png files.
    Returns
    -------
    aryPngData : 2d numpy array, shape [png_x, png_y, n_vols]
        Stack of stimulus data.

    """
    print('------Load PNGs')
    # Create list of png files to load:
    lstPngPaths = [None] * varNumVol
    for idx01 in range(0, varNumVol):
        lstPngPaths[idx01] = (strPathPng + str(idx01) + '.png')

    # Load png files. The png data will be saved in a numpy array of the
    # following order: aryPngData[x-pixel, y-pixel, PngNumber]. The
    # sp.misc.imread function actually contains three values per pixel (RGB),
    # but since the stimuli are black-and-white, any one of these is sufficient
    # and we discard the others.
    aryPngData = np.zeros((tplPngSize[0],
                           tplPngSize[1],
                           varNumVol))
    for idx01 in range(0, varNumVol):
        aryPngData[:, :, idx01] = np.array(Image.open(lstPngPaths[idx01]))

    # Convert RGB values (0 to 255) to integer ones and zeros:
    aryPngData = (aryPngData > 0).astype(int)

    return aryPngData