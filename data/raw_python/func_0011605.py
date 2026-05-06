def r(self, **kwargs):
        """
        Resolve the object.

        This returns default (if present) or fails on an Empty.
        """
        # by using kwargs we ensure that usage of positional arguments, as if
        # this object were another kind of function, will fail-fast and raise
        # a TypeError
        if 'default' in kwargs:
            default = kwargs.pop('default')
            if kwargs:
                raise TypeError(
                    "Unexpected argument: {}".format(repr(next(iter(kwargs))))
                )
            return default
        else:
            raise JSaneException(
                "Key does not exist: {}".format(repr(self._key_name))
            )