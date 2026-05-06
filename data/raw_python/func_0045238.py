def _rows_date2int(self, rows):
        """
        Replaces start and end dates in a row set with their integer representation

        :param list[dict[str,T]] rows: The list of rows.
        """
        for row in rows:
            # Determine the type of dates based on the first start date.
            if not self._date_type:
                self._date_type = self._get_date_type(row[self._key_start_date])

            # Convert dates to integers.
            row[self._key_start_date] = self._date2int(row[self._key_start_date])
            row[self._key_end_date] = self._date2int(row[self._key_end_date])