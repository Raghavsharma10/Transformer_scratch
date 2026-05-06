def additem(self, key, value):
        """
        Add a new item. Raises ``KeyError`` if key is already in the pqdict.

        """
        if key in self._position:
            raise KeyError('%s is already in the queue' % repr(key))
        self[key] = value