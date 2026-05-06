def fromkeys(cls, iterable, value=None):
        # TODO : type: (Iterable, Union[Any, Callable]) -> DictWrapper
        # https://github.com/python/mypy/issues/2254
        """Create a new d from

        Args:
            iterable: Iterable containing keys
            value: value to associate with each key.
            If callable, will be value[key]

        Returns: new DictWrapper

        Example:

            >>> from ww import d
            >>> sorted(d.fromkeys('123', value=4).items())
            [('1', 4), ('2', 4), ('3', 4)]
            >>> sorted(d.fromkeys(range(3), value=lambda e:e**2).items())
            [(0, 0), (1, 1), (2, 4)]
        """
        if not callable(value):
            return cls(dict.fromkeys(iterable, value))

        return cls((key, value(key)) for key in iterable)