def items(self):
        """Return list of tuples of keys and values in db

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2, 3, 4]
        >>> dc.items()
        [('l0', ['1', '2', '3', '4'])]
        >>> dc.clear()

        :return: list of (key, value) pairs
        :rtype: list of tuple
        """
        logger.debug('call items')
        return [(key, self.get(key)) for key in self.keys()]