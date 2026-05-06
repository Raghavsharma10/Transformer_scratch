def validate_integrity(self):
        """
        Validate the integrity of the DataFrame. This checks that the indexes, column names and internal data are not
        corrupted. Will raise an error if there is a problem.

        :return: nothing
        """
        self._validate_columns(self._columns)
        self._validate_index(self._index)
        self._validate_data()