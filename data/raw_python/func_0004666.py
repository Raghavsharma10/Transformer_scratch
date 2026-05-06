def append_rows(self, indexes, values, new_cols=True):
        """
        Appends rows of values to the end of the data. If there are new columns in the values and new_cols is True
        they will be added. Be very careful with this function as for sort DataFrames it will not enforce sort order. 
        Use this only for speed when needed, be careful.

        :param indexes: list of indexes
        :param values: dictionary of values where the key is the column name and the value is a list
        :param new_cols: if True add new columns in values, if False ignore
        :return: nothing
        """

        # check that the values data is less than or equal to the length of the indexes
        for column in values:
            if len(values[column]) > len(indexes):
                raise ValueError('length of %s column in values is longer than indexes' % column)

        # check the indexes are not duplicates
        combined_index = self._index + indexes
        if len(set(combined_index)) != len(combined_index):
            raise IndexError('duplicate indexes in DataFrames')

        if new_cols:
            for col in values:
                if col not in self._columns:
                    self._add_column(col)

        # append index value
        self._index.extend(indexes)

        # add data values, if not in values then use None
        for c, col in enumerate(self._columns):
            self._data[c].extend(values.get(col, [None] * len(indexes)))
        self._pad_data()