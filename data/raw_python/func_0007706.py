def _read_as_table(self):
        """
        Read the data contained in all entries as a list of
        lists containing all of the data

        :return: list of dicts containing all tabular data
        """
        rows = list()

        for row in self._rows:
            rows.append([row[i].get() for i in range(self.num_of_columns)])

        return rows