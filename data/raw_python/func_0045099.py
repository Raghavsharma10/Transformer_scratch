def condense(self):
        """
        Condense the data set to the distinct intervals based on the pseudo key.
        """
        for pseudo_key, rows in self._rows.items():
            tmp1 = []
            intervals = sorted(self._derive_distinct_intervals(rows))
            for interval in intervals:
                tmp2 = dict(zip(self._pseudo_key, pseudo_key))
                tmp2[self._key_start_date] = interval[0]
                tmp2[self._key_end_date] = interval[1]
                tmp1.append(tmp2)

            self._rows[pseudo_key] = tmp1