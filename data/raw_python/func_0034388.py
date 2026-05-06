def _grabData(self, key):
        """
        Tries to find the existing document in the database, if it is found,
        then the objects _data is set to that document, and this returns
        `True`, otherwise this will return `False`

        :param key: The primary key of the object we're looking for
        :type key: Str

        :return: True if a document was found, otherwise False
        :rtype: Boolean
        """
        rawCursor = r.table(self.table).get(key).run(self._conn)
        if rawCursor:
            self._data = rawCursor
            self._new = False
            return True
        else:
            return False