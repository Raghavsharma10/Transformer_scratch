def pop(self, key, default=None):
        """Remove and return the last item of the list ``key``.
        If key doesn't exists it return ``default``.

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2, 3, 4]
        >>> dc.pop('l0')
        ['1', '2', '3', '4']
        >>> dc.pop('l1', 'empty')
        'empty'

        :param key: key name to pop
        :type key: str
        :param default: default value if key doesn't exist
        :type default: Any
        :return: value associated with given key or None or ``default``
        :rtype: Any
        """
        logger.debug('call pop %s', key)
        value = self.get(key)
        self._redis.delete(key)
        return value or default