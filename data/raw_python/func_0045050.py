def get_id(self, natural_key, date, enhancement=None):
        """
        Returns the technical ID for a natural key at a date or None if the given natural key is not valid.

        :param T natural_key: The natural key.
        :param str date: The date in ISO 8601 (YYYY-MM-DD) format.
        :param T enhancement: Enhancement data of the dimension row.

        :rtype: int|None
        """
        if not date:
            return None

        # If the natural key is known return the technical ID immediately.
        if natural_key in self._map:
            for row in self._map[natural_key]:
                if row[0] <= date <= row[1]:
                    return row[2]

        # The natural key is not in the map of this dimension. Call a stored procedure for translating the natural key
        # to a technical key.
        self.pre_call_stored_procedure()
        success = False
        try:
            row = self.call_stored_procedure(natural_key, date, enhancement)
            # Convert dates to strings in ISO 8601 format.
            if isinstance(row[self._key_date_start], datetime.date):
                row[self._key_date_start] = row[self._key_date_start].isoformat()
            if isinstance(row[self._key_date_end], datetime.date):
                row[self._key_date_end] = row[self._key_date_end].isoformat()
            success = True
        finally:
            self.post_call_stored_procedure(success)

        # Make sure the natural key is in the map.
        if natural_key not in self._map:
            self._map[natural_key] = []

        if row[self._key_key]:
            self._map[natural_key].append((row[self._key_date_start],
                                           row[self._key_date_end],
                                           row[self._key_key]))
        else:
            self._map[natural_key].append((date, date, None))

        return row[self._key_key]