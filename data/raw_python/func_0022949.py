def write_png(filename, data):
    """Write a PNG file

    Unlike imsave, this requires no external dependencies.

    Parameters
    ----------
    filename : str
        File to save to.
    data : array
        Image data.

    See also
    --------
    read_png, imread, imsave
    """
    data = np.asarray(data)
    if not data.ndim == 3 and data.shape[-1] in (3, 4):
        raise ValueError('data must be a 3D array with last dimension 3 or 4')
    with open(filename, 'wb') as f:
        f.write(_make_png(data))