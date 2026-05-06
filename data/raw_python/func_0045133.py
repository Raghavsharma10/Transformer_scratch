def match(self, row):
        """
        Returns True if the field matches the regular expression of this simple condition. Returns False otherwise.

        :param dict row: The row.

        :rtype: bool
        """
        if re.search(self._expression, row[self._field]):
            return True

        return False