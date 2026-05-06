def _rows_int2date(self, rows):
        """
        Replaces start and end dates in the row set with their integer representation

        :param list[dict[str,T]] rows: The list of rows.
        """
        for row in rows:
            if self._date_type == 'str':
                row[self._key_start_date] = datetime.date.fromordinal(row[self._key_start_date]).isoformat()
                row[self._key_end_date] = datetime.date.fromordinal(row[self._key_end_date]).isoformat()
            elif self._date_type == 'date':
                row[self._key_start_date] = datetime.date.fromordinal(row[self._key_start_date])
                row[self._key_end_date] = datetime.date.fromordinal(row[self._key_end_date])
            elif self._date_type == 'int':
                # Nothing to do.
                pass
            else:
                raise ValueError('Unexpected date type {0!s}'.format(self._date_type))