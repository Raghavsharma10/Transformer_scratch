def _intersection(self, keys, rows):
        """
        Computes the intersection of the date intervals of two or more reference data sets. If the intersection is empty
        the row is removed from the group.

        :param list[tuple[str,str]] keys: The other keys with start and end date.
        :param list[dict[str,T]] rows: The list of rows.

        :rtype: list[dict[str,T]]
        """
        # If there are no other keys with start and end date (i.e. nothing to merge) return immediately.
        if not keys:
            return rows

        ret = list()
        for row in rows:
            start_date = row[self._key_start_date]
            end_date = row[self._key_end_date]
            for key_start_date, key_end_date in keys:
                start_date, end_date = Type2JoinHelper._intersect(start_date,
                                                                  end_date,
                                                                  row[key_start_date],
                                                                  row[key_end_date])
                if not start_date:
                    break
                if key_start_date not in [self._key_start_date, self._key_end_date]:
                    del row[key_start_date]
                if key_end_date not in [self._key_start_date, self._key_end_date]:
                    del row[key_end_date]

            if start_date:
                row[self._key_start_date] = start_date
                row[self._key_end_date] = end_date
                ret.append(row)

        return ret