def fromkeys(cls, seq, value=None, **kwargs):
        """
        Create a new collection with keys from *seq* and values set to
        *value*. The keyword arguments are passed to the persistent ``Dict``.
        """
        other = cls(**kwargs)
        other.update(((key, value) for key in seq))

        return other