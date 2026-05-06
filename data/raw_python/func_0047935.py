def values(self):
        """Return list of values in db

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2, 3, 4]
        >>> dc.items()
        [('l0', ['1', '2', '3', '4'])]
        >>> dc.clear()

        :return: list of tuple
        :rtype: list
        """
        logger.debug('call values')
        return [self.get(key) for key in self.keys()]