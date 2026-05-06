def append_row(self, index, values, new_cols=True):
        """
        Appends a row of values to the end of the data. If there are new columns in the values and new_cols is True
        they will be added. Be very careful with this function as for sort DataFrames it will not enforce sort order. 
        Use this only for speed when needed, be careful.

        :param index: value of the index
        :param values: dictionary of values
        :param new_cols: if True add new columns in values, if False ignore
        :return: nothing
        """

        if index in self._index:
            raise IndexError('index already in DataFrame')

        if new_cols:
            for col in values:
                if col not in self._columns:
                    self._add_column(col)

        # append index value
        self._index.append(index)

        # add data values, if not in values then use None
        for c, col in enumerate(self._columns):
            self._data[c].append(values.get(col, None))