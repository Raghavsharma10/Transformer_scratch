def _equal(self, row1, row2):
        """
        Returns True if two rows are identical excluding start and end date. Returns False otherwise.

        :param dict[str,T] row1: The first row.
        :param dict[str,T] row2: The second row.

        :rtype: bool
        """
        for key in row1.keys():
            if key not in [self._key_start_date, self._key_end_date]:
                if row1[key] != row2[key]:
                    return False

        return True