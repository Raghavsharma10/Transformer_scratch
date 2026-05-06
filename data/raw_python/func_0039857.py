def invert(image):
    """Return an inverted image of the same dtype.

    Assumes the full range of the input dtype is in use and
    that no negative values are present in the input image.

    :param image: :class:`jicbioimage.core.image.Image`
    :returns: inverted image of the same dtype as the input
    """
    if image.dtype == bool:
        return np.logical_not(image)
    maximum = np.iinfo(image.dtype).max
    maximum_array = np.ones(image.shape, dtype=image.dtype) * maximum
    return maximum_array - image