def zoom_to_shape(data, shape, dtype=None):
    """
    Zoom data to specific shape.
    """
    import scipy
    import scipy.ndimage

    zoomd = np.array(shape) / np.array(data.shape, dtype=np.double)
    import warnings

    datares = scipy.ndimage.interpolation.zoom(data, zoomd, order=0, mode="reflect")

    if datares.shape != shape:
        logger.warning("Zoom with different output shape")
    dataout = np.zeros(shape, dtype=dtype)
    shpmin = np.minimum(dataout.shape, shape)

    dataout[: shpmin[0], : shpmin[1], : shpmin[2]] = datares[
        : shpmin[0], : shpmin[1], : shpmin[2]
    ]
    return datares