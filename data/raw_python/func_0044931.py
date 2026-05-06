def merge(self, keys):
        """
        Merges the join on pseudo keys of two or more reference data sets.

        :param list[tuple[str,str]] keys: For each data set the keys of the start and end date.
        """
        deletes = []
        for pseudo_key, rows in self._rows.items():
            self._additional_rows_date2int(keys, rows)
            rows = self._intersection(keys, rows)
            if rows:
                rows = self._rows_sort(rows)
                self._rows[pseudo_key] = self._merge_adjacent_rows(rows)
            else:
                deletes.append(pseudo_key)

        for pseudo_key in deletes:
            del self._rows[pseudo_key]