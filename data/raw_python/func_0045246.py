def prepare_data(self, rows):
        """
        Sets and prepares the rows. The rows are stored in groups in a dictionary. A group is a list of rows with the
        same pseudo key. The key in the dictionary is a tuple with the values of the pseudo key.

        :param list[dict] rows: The rows
        """
        self._rows = dict()
        for row in copy.copy(rows) if self.copy else rows:
            pseudo_key = self._get_pseudo_key(row)
            if pseudo_key not in self._rows:
                self._rows[pseudo_key] = list()
            self._rows[pseudo_key].append(row)

        # Convert begin and end dates to integers.
        self._date_type = None
        for pseudo_key, rows in self._rows.items():
            self._rows_date2int(rows)