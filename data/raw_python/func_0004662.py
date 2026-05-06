def set_row(self, index, values):
        """
        Sets the values of the columns in a single row.

        :param index: index value
        :param values: dict with the keys as the column names and the values what to set that column to
        :return: nothing
        """
        if self._sort:
            exists, i = sorted_exists(self._index, index)
            if not exists:
                self._insert_row(i, index)
        else:
            try:
                i = self._index.index(index)
            except ValueError:  # new row
                i = len(self._index)
                self._add_row(index)
        if isinstance(values, dict):
            if not (set(values.keys()).issubset(self._columns)):
                raise ValueError('keys of values are not all in existing columns')
            for c, column in enumerate(self._columns):
                self._data[c][i] = values.get(column, self._data[c][i])
        else:
            raise TypeError('cannot handle values of this type.')