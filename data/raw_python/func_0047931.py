def get(self, key, default=None):
        """Return the value at key ``key``, or default value ``default``
        which is None by default.

        >>> dc = Dictator()
        >>> dc['l0'] = [1, 2, 3, 4]
        >>> dc.get('l0')
        ['1', '2', '3', '4']
        >>> dc['l0']
        ['1', '2', '3', '4']
        >>> dc.clear()

        :param key: key of value to return
        :type key: str
        :param default: value of any type to return of key doesn't exist.
        :type default: Any
        :return: value of given key
        :rtype: Any
        """
        try:
            value = self.__getitem__(key)
        except KeyError:
            value = None

        # Py3 Redis compatibiility
        if isinstance(value, bytes):
            value = value.decode()
        return value or default