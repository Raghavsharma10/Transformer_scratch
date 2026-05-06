def match(self, row):
        """
        Returns True if the field is in the list of conditions. Returns False otherwise.

        :param dict row: The row.

        :rtype: bool
        """
        if row[self._field] in self._values:
            return True

        for condition in self._conditions:
            if condition.match(row):
                return True

        return False