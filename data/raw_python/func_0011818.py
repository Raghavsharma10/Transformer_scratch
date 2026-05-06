def mat_to_surface(mat: np.ndarray, transformer=to_24bit_gray):
    """Can be used to create a pygame.Surface from a 2d numpy array.

    By default a grey image with scaled colors is returned, but using the
    transformer argument any transformation can be used.

    :param mat: the matrix to create the surface of.
    :type mat: np.ndarray

    :param transformer: function that transforms the matrix to a valid color
        matrix, i.e. it must have 3dimension, were the 3rd dimension are the color
        channels. For each channel a value between 0 and 255 is allowed
    :type transformer: Callable[np.ndarray[np.ndarray]]"""

    return pygame.pixelcopy.make_surface(transformer(mat.transpose()) 
        if transformer is not None else mat.transpose())