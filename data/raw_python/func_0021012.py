def fromkeys(cls, seq, value=None, **kwargs):
        """Create a new dictionary with keys from *seq* and values set to
        *value*.

        .. note::
            :func:`fromkeys` is a class method that returns a new dictionary.
            It is possible to specify additional keyword arguments to be passed
            to :func:`__init__` of the new object.
        """
        values = ((key, value) for key in seq)
        return cls(values, **kwargs)