def ensure_array(array):
    """
    Assert that the given array is an Array subclass (or numpy array).

    If the given array is a numpy.ndarray an appropriate NumpyArrayAdapter
    instance is created, otherwise the passed array must be a subclass of
    :class:`Array` else a TypeError will be raised.

    """
    if not isinstance(array, Array):
        if isinstance(array, np.ndarray):
            array = NumpyArrayAdapter(array)
        elif np.isscalar(array):
            array = ConstantArray([], array)
        else:
            raise TypeError('The given array should be a `biggus.Array` '
                            'instance, got {}.'.format(type(array)))
    return array