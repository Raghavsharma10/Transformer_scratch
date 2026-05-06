def keys(self, pattern=None):
        """Returns a list of keys matching ``pattern``.
        By default return all keys.

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2, 3, 4]
        >>> dc['s0'] = 'string value'
        >>> dc.keys()
        ['l0', 's0']
        >>> dc.keys('h*')
        []
        >>> dc.clear()

        :param pattern: key pattern
        :type pattern: str
        :return: list of keys in db
        :rtype: list of str
        """
        logger.debug('call pop %s', pattern)
        if pattern is None:
            pattern = '*'
        return self._redis.keys(pattern=pattern)