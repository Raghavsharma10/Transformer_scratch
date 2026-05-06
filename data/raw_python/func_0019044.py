def new(cls, variable, **kwargs):
        """Return a new |DefaultMask| object associated with the
        given |Variable| object."""
        return cls.array2mask(numpy.full(variable.shape, True))