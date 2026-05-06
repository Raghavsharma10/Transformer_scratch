def remove_small_objects(image, min_size=50, connectivity=1):
    """Remove small objects from an boolean image.

    :param image: boolean numpy array or :class:`jicbioimage.core.image.Image`
    :returns: boolean :class:`jicbioimage.core.image.Image`
    """
    return skimage.morphology.remove_small_objects(image,
                                                   min_size=min_size,
                                                   connectivity=connectivity)