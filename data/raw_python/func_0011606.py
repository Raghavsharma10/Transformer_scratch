def r(self, **kwargs):
        """
        Resolve the object.

        This will always succeed, since, if a lookup fails, an Empty
        instance will be returned farther upstream.
        """
        # by using kwargs we ensure that usage of positional arguments, as if
        # this object were another kind of function, will fail-fast and raise
        # a TypeError
        kwargs.pop('default', None)
        if kwargs:
            raise TypeError(
                "Unexpected argument: {}".format(repr(next(iter(kwargs))))
            )
        return self._obj