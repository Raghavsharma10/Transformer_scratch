def _additional_rows_date2int(self, keys, rows):
        """
        Replaces start and end dates of the additional date intervals in the row set with their integer representation

        :param list[tuple[str,str]] keys: The other keys with start and end date.
        :param list[dict[str,T]] rows: The list of rows.

        :rtype: list[dict[str,T]]
        """
        for row in rows:
            for key_start_date, key_end_date in keys:
                if key_start_date not in [self._key_start_date, self._key_end_date]:
                    row[key_start_date] = self._date2int(row[key_start_date])
                if key_end_date not in [self._key_start_date, self._key_end_date]:
                    row[key_end_date] = self._date2int(row[key_end_date])