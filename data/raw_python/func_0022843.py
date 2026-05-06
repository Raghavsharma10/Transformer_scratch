def resize(image, shape, kind='linear'):
    """Resize an image

    Parameters
    ----------
    image : ndarray
        Array of shape (N, M, ...).
    shape : tuple
        2-element shape.
    kind : str
        Interpolation, either "linear" or "nearest".

    Returns
    -------
    scaled_image : ndarray
        New image, will have dtype np.float64.
    """
    image = np.array(image, float)
    shape = np.array(shape, int)
    if shape.ndim != 1 or shape.size != 2:
        raise ValueError('shape must have two elements')
    if image.ndim < 2:
        raise ValueError('image must have two dimensions')
    if not isinstance(kind, string_types) or kind not in ('nearest', 'linear'):
        raise ValueError('mode must be "nearest" or "linear"')

    r = np.linspace(0, image.shape[0] - 1, shape[0])
    c = np.linspace(0, image.shape[1] - 1, shape[1])
    if kind == 'linear':
        r_0 = np.floor(r).astype(int)
        c_0 = np.floor(c).astype(int)
        r_1 = r_0 + 1
        c_1 = c_0 + 1

        top = (r_1 - r)[:, np.newaxis]
        bot = (r - r_0)[:, np.newaxis]
        lef = (c - c_0)[np.newaxis, :]
        rig = (c_1 - c)[np.newaxis, :]

        c_1 = np.minimum(c_1, image.shape[1] - 1)
        r_1 = np.minimum(r_1, image.shape[0] - 1)
        for arr in (top, bot, lef, rig):
            arr.shape = arr.shape + (1,) * (image.ndim - 2)
        out = top * rig * image[r_0][:, c_0, ...]
        out += bot * rig * image[r_1][:, c_0, ...]
        out += top * lef * image[r_0][:, c_1, ...]
        out += bot * lef * image[r_1][:, c_1, ...]
    else:  # kind == 'nearest'
        r = np.round(r).astype(int)
        c = np.round(c).astype(int)
        out = image[r][:, c, ...]
    return out