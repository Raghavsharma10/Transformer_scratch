def match(self, row):
        """
        Returns True if the field matches the glob expression of this simple condition. Returns False otherwise.

        :param dict row: The row.

        :rtype: bool
        """
        return fnmatch.fnmatchcase(row[self._field], self._expression)