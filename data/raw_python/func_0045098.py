def _derive_distinct_intervals(self, rows):
        """
        Returns the set of distinct intervals in a row set.

        :param list[dict[str,T]] rows: The rows set.

        :rtype: set[(int,int)]
        """
        ret = set()
        for row in rows:
            self._add_interval(ret, (row[self._key_start_date], row[self._key_end_date]))

        return ret