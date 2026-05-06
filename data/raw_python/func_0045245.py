def get_rows(self, sort=False):
        """
        Returns the rows of this Type2Helper.

        :param bool sort: If True the rows are sorted by the pseudo key.
        """
        ret = []
        for _, rows in sorted(self._rows.items()) if sort else self._rows.items():
            self._rows_int2date(rows)
            ret.extend(rows)

        return ret