def copy(self):
        """Convert ``Dictator`` to standard ``dict`` object

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2]
        >>> dc['1'] = 'abc'
        >>> d = dc.copy()
        >>> type(d)
        dict
        >>> d
        {'l0': ['1', '2'], '1': 'abc'}
        >>> dc.clear()

        :return: Python's dict object
        :rtype: dict
        """
        logger.debug('call to_dict')
        return {key: self.get(key) for key in self.keys()}