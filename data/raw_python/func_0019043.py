def array2mask(cls, array=None, **kwargs):
        """Create a new mask object based on the given |numpy.ndarray|
        and return it."""
        kwargs['dtype'] = bool
        if array is None:
            return numpy.ndarray.__new__(cls, 0, **kwargs)
        return numpy.asarray(array, **kwargs).view(cls)