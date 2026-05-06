def fromkeys(cls, iterable, value, **kwargs):
        """
        Return a new pqict mapping keys from an iterable to the same value.

        """
        return cls(((k, value) for k in iterable), **kwargs)