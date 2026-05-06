def iteritems(self, match=None, count=1):
        """Return an iterator over the db's (key, value) pairs.
        ``match`` allows for filtering the keys by pattern.
        ``count`` allows for hint the minimum number of returns.

        >>> dc = Dictator()
        >>> dc['1'] = 'abc'
        >>> dc['2'] = 'def'
        >>> dc['3'] = 'ghi'
        >>> itr = dc.iteritems()
        >>> type(itr)
        <type 'generator'>
        >>> list(reversed([item for item in itr]))
        [('1', 'abc'), ('2', 'def'), ('3', 'ghi')]
        >>> dc.clear()

        :param match: pattern to filter keys
        :type match: str
        :param count: minimum number of returns
        :type count: int
        :return: iterator over key, value pairs.
        :rtype: generator
        """
        logger.debug('call iteritems %s', match)
        if match is None:
            match = '*'
        for key in self._redis.scan_iter(match=match, count=count):
            yield key, self.get(key)