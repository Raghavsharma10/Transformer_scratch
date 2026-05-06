def iterkeys(self, match=None, count=1):
        """Return an iterator over the db's keys.
        ``match`` allows for filtering the keys by pattern.
        ``count`` allows for hint the minimum number of returns.

        >>> dc = Dictator()
        >>> dc['1'] = 'abc'
        >>> dc['2'] = 'def'
        >>> dc['3'] = 'ghi'
        >>> itr = dc.iterkeys()
        >>> type(itr)
        <type 'generator'>
        >>> list(reversed([item for item in itr]))
        ['1', '2', '3']
        >>> dc.clear()

        :param match: pattern to filter keys
        :type match: str
        :param count: minimum number of returns
        :type count: int
        :return: iterator over key.
        :rtype: generator
        """
        logger.debug('call iterkeys %s', match)
        if match is None:
            match = '*'
        for key in self._redis.scan_iter(match=match, count=count):
            yield key