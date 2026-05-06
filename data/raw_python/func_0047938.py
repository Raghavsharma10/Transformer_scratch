def update(self, other=None, **kwargs):
        """D.update([other, ]**kwargs) -> None.
        Update D From dict/iterable ``other`` and ``kwargs``.
        If ``other`` present and has a .keys() method, does:
            for k in other: D[k] = other[k]
        If ``other`` present and lacks .keys() method, does:
            for (k, v) in other: D[k] = v
        In either case, this is followed by: for k in kwargs: D[k] = kwargs[k]

        >>> dc = Dictator()
        >>> dc['1'] = 'abc'
        >>> dc['2'] = 'def'
        >>> dc.values()
        ['def', 'abc']
        >>> dc.update({'3': 'ghi'}, name='Keys')
        >>> dc.values()
        ['Keys', 'ghi', 'def', 'abc']
        >>> dc.clear()

        :param other: dict/iterable with .keys() function.
        :param kwargs: key/value pairs
        """
        logger.debug('call update %s', other)
        if other:
            if hasattr(other, 'keys'):
                for key in other.keys():
                    self.set(key, other[key])
            else:
                for (key, value) in other:
                    self.set(key, value)

        if kwargs:
            for key, value in six.iteritems(kwargs):
                self.set(key, value)