def enumerate(self, name, start=1):
        """
        Enumerates all rows such that the pseudo key and the ordinal number are a unique key.

        :param str name: The key holding the ordinal number.
        :param int start: The start of the ordinal numbers. Foreach pseudo key the first row has this ordinal number.
        """
        for pseudo_key, rows in self._rows.items():
            rows = self._rows_sort(rows)
            ordinal = start
            for row in rows:
                row[name] = ordinal
                ordinal += 1
            self._rows[pseudo_key] = rows