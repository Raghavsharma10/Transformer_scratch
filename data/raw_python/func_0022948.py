def read_png(filename):
    """Read a PNG file to RGB8 or RGBA8

    Unlike imread, this requires no external dependencies.

    Parameters
    ----------
    filename : str
        File to read.

    Returns
    -------
    data : array
        Image data.

    See also
    --------
    write_png, imread, imsave
    """
    x = Reader(filename)
    try:
        alpha = x.asDirect()[3]['alpha']
        if alpha:
            y = x.asRGBA8()[2]
            n = 4
        else:
            y = x.asRGB8()[2]
            n = 3
        y = np.array([yy for yy in y], np.uint8)
    finally:
        x.file.close()
    y.shape = (y.shape[0], y.shape[1] // n, n)
    return y