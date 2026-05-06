def match(self, row):
        """
        Returns True if the row matches one or more child conditions. Returns False otherwise.

        :param dict row: The row.

        :rtype: bool
        """
        for condition in self._conditions:
            if condition.match(row):
                return True

        return False